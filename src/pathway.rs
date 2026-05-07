//! `NPC` pathway scoring through fixed representative spectra.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use mass_spectrometry::prelude::{
    FlashCosineThresholdIndex, GenericSpectrum, SearchState, SpectraIndexBuilder,
};
use rayon::prelude::*;

use crate::{
    cli::ScanArgs,
    model::{LoadedRecord, Metric, PathwayPrediction, PathwayScore, SimilarityConfig},
    progress::ScanProgress,
};

/// Score query spectra against fixed `NPC` pathway representatives.
pub fn score_pathway_representatives(
    args: &ScanArgs,
    progress: &ScanProgress,
    config: &SimilarityConfig,
    peak_count: usize,
    records: &[LoadedRecord],
    spectra: &[GenericSpectrum],
    query_ids: &[usize],
) -> Result<Option<(Vec<PathwayScore>, Vec<PathwayPrediction>)>> {
    if config.metric != Metric::Cosine || args.pathway_representatives_per_class == 0 {
        return Ok(None);
    }

    let representatives =
        select_pathway_representatives(records, args.pathway_representatives_per_class);
    if representatives.is_empty() {
        return Ok(Some((Vec::new(), Vec::new())));
    }
    let reference_spectra = representatives
        .iter()
        .map(|representative| spectra[representative.record_index].clone())
        .collect::<Vec<_>>();

    let mut builder = FlashCosineThresholdIndex::<f64>::builder()
        .mz_power(config.mz_power)
        .intensity_power(config.intensity_power)
        .mz_tolerance(args.mz_tolerance)
        .score_threshold(0.0)
        .parallel();
    if let Some(pepmass_tolerance) = args.pepmass_tolerance {
        builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    }
    let config_name = config.name();
    let index_progress = progress.spinner(format!(
        "building pathway representative index for {config_name} top {peak_count} peaks"
    ));
    let index = builder.build(&reference_spectra)?;
    index_progress.finish();
    let pathways = representative_pathways(&representatives);
    let task = progress.bar(
        u64::try_from(query_ids.len()).unwrap_or(u64::MAX),
        format!("pathway scoring {config_name} top {peak_count} peaks"),
    );

    let rows = query_ids
        .par_iter()
        .map_init(
            || index.new_search_state(),
            |state, &query_index| {
                let rows = score_query_pathways(
                    args,
                    config,
                    &config_name,
                    peak_count,
                    records,
                    spectra,
                    query_index,
                    &representatives,
                    &pathways,
                    &index,
                    state,
                );
                task.inc(1);
                rows
            },
        )
        .collect::<Result<Vec<_>>>()?;
    task.finish();

    let mut scores = Vec::new();
    let mut predictions = Vec::new();
    for (query_scores, prediction) in rows {
        scores.extend(query_scores);
        predictions.push(prediction);
    }
    Ok(Some((scores, predictions)))
}

/// Pick the first `m` labeled records from each pathway.
fn select_pathway_representatives(
    records: &[LoadedRecord],
    representatives_per_class: usize,
) -> Vec<PathwayRepresentative> {
    let mut selected_per_pathway = BTreeMap::<String, usize>::new();
    let mut representatives = Vec::new();
    for (record_index, record) in records.iter().enumerate() {
        let labels = pathway_labels(record.npc_pathway.as_deref());
        let mut selected_pathways = Vec::new();
        for pathway in labels {
            let selected = selected_per_pathway.entry(pathway.clone()).or_default();
            if *selected >= representatives_per_class {
                continue;
            }
            selected_pathways.push(pathway);
            *selected += 1;
        }
        if !selected_pathways.is_empty() {
            representatives.push(PathwayRepresentative {
                record_index,
                pathways: selected_pathways,
            });
        }
    }
    representatives
}

/// Split a raw `NPC_PATHWAY` field into unique pathway labels.
fn pathway_labels(raw: Option<&str>) -> Vec<String> {
    raw.into_iter()
        .flat_map(|labels| labels.split('|'))
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Count representatives per pathway.
fn representative_pathways(representatives: &[PathwayRepresentative]) -> BTreeMap<String, usize> {
    let mut pathways = BTreeMap::new();
    for representative in representatives {
        for pathway in &representative.pathways {
            *pathways.entry(pathway.clone()).or_default() += 1;
        }
    }
    pathways
}

/// Score one query against all pathway representative groups.
#[allow(clippy::too_many_arguments)]
fn score_query_pathways(
    args: &ScanArgs,
    config: &SimilarityConfig,
    config_name: &str,
    peak_count: usize,
    records: &[LoadedRecord],
    spectra: &[GenericSpectrum],
    query_index: usize,
    representatives: &[PathwayRepresentative],
    pathways: &BTreeMap<String, usize>,
    index: &FlashCosineThresholdIndex<f64>,
    state: &mut SearchState,
) -> Result<(Vec<PathwayScore>, PathwayPrediction)> {
    let query = records
        .get(query_index)
        .with_context(|| format!("query index {query_index} is out of bounds"))?;
    let mut scored_pathways = pathways
        .iter()
        .map(|(pathway, &representative_count)| {
            (
                pathway.clone(),
                PathwayAggregate {
                    representatives: representative_count,
                    score: 0.0,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let hits =
        index.search_top_k_with_state(&spectra[query_index], representatives.len(), state)?;
    for hit in hits {
        let representative_index =
            usize::try_from(hit.spectrum_id).context("representative index does not fit usize")?;
        let representative = representatives.get(representative_index).with_context(|| {
            format!("representative index {representative_index} is out of bounds")
        })?;
        if representative.record_index == query_index {
            continue;
        }
        for pathway_name in &representative.pathways {
            if let Some(pathway) = scored_pathways.get_mut(pathway_name) {
                pathway.score += hit.score;
            }
        }
    }

    let mut best_pathway = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut score_rows = Vec::with_capacity(scored_pathways.len());
    for (candidate_pathway, aggregate) in scored_pathways {
        if aggregate.score > best_score {
            best_score = aggregate.score;
            best_pathway = Some(candidate_pathway.clone());
        }
        score_rows.push(PathwayScore {
            dataset: args.dataset.as_str().to_string(),
            config: config_name.to_string(),
            peak_count,
            query_index,
            query_id: query.id.clone(),
            query_npc_pathway: query.npc_pathway.clone(),
            candidate_npc_pathway: candidate_pathway,
            representatives: aggregate.representatives,
            score: aggregate.score,
        });
    }

    let predicted_score = if best_pathway.is_some() {
        best_score
    } else {
        0.0
    };
    let is_correct =
        pathway_prediction_is_correct(query.npc_pathway.as_deref(), best_pathway.as_deref());
    let prediction = PathwayPrediction {
        dataset: args.dataset.as_str().to_string(),
        config: config.name(),
        peak_count,
        query_index,
        query_id: query.id.clone(),
        query_npc_pathway: query.npc_pathway.clone(),
        predicted_npc_pathway: best_pathway,
        predicted_score,
        is_correct,
        candidate_pathways: score_rows.len(),
    };

    Ok((score_rows, prediction))
}

/// Return whether a predicted pathway is among the query pathway labels.
fn pathway_prediction_is_correct(expected: Option<&str>, predicted: Option<&str>) -> Option<bool> {
    let predicted = predicted?;
    let expected_labels = pathway_labels(expected);
    if expected_labels.is_empty() {
        return None;
    }
    Some(expected_labels.iter().any(|expected| expected == predicted))
}

/// Selected pathway representative metadata.
struct PathwayRepresentative {
    /// Original record index in the loaded dataset.
    record_index: usize,
    /// Pathways represented by the record.
    pathways: Vec<String>,
}

/// Accumulated score for one candidate pathway.
struct PathwayAggregate {
    /// Number of representative spectra assigned to the pathway.
    representatives: usize,
    /// Sum of query cosine similarities to representatives.
    score: f64,
}

#[cfg(test)]
/// Unit tests for pathway label handling.
mod tests {
    use anyhow::Result;
    use mass_spectrometry::prelude::GenericSpectrum;

    use crate::model::LoadedRecord;

    use super::{
        pathway_labels, pathway_prediction_is_correct, representative_pathways,
        select_pathway_representatives,
    };

    #[test]
    /// Pipe-separated `NPC` pathway values are treated as multiple labels.
    fn pathway_labels_split_pipe_separated_values() {
        assert_eq!(
            pathway_labels(Some(" Terpenoids | Alkaloids || Alkaloids ")),
            vec!["Alkaloids".to_string(), "Terpenoids".to_string()]
        );
        assert!(pathway_labels(None).is_empty());
    }

    #[test]
    /// Multi-label representatives count toward every selected pathway quota.
    fn representatives_include_each_multilabel_pathway() -> Result<()> {
        let records = vec![
            labeled_record("first", Some("pathway-b|pathway-a"))?,
            labeled_record("second", Some("pathway-b"))?,
            labeled_record("third", Some("pathway-c"))?,
        ];

        let representatives = select_pathway_representatives(&records, 1);

        assert_eq!(representatives.len(), 2);
        assert_eq!(representatives[0].record_index, 0);
        assert_eq!(
            representatives[0].pathways,
            vec!["pathway-a".to_string(), "pathway-b".to_string()]
        );
        assert_eq!(representatives[1].record_index, 2);
        assert_eq!(representatives[1].pathways, vec!["pathway-c".to_string()]);

        let counts = representative_pathways(&representatives);
        assert_eq!(counts.get("pathway-a"), Some(&1));
        assert_eq!(counts.get("pathway-b"), Some(&1));
        assert_eq!(counts.get("pathway-c"), Some(&1));
        Ok(())
    }

    #[test]
    /// A prediction is correct when it matches any query pathway label.
    fn prediction_correctness_accepts_any_query_label() {
        assert_eq!(
            pathway_prediction_is_correct(Some("pathway-a|pathway-b"), Some("pathway-b")),
            Some(true)
        );
        assert_eq!(
            pathway_prediction_is_correct(Some("pathway-a|pathway-b"), Some("pathway-c")),
            Some(false)
        );
        assert_eq!(pathway_prediction_is_correct(None, Some("pathway-a")), None);
        assert_eq!(
            pathway_prediction_is_correct(Some(""), Some("pathway-a")),
            None
        );
        assert_eq!(pathway_prediction_is_correct(Some("pathway-a"), None), None);
    }

    /// Build one minimal labeled record for pathway tests.
    fn labeled_record(id: &str, npc_pathway: Option<&str>) -> Result<LoadedRecord> {
        Ok(LoadedRecord {
            id: id.to_string(),
            name: None,
            npc_pathway: npc_pathway.map(ToOwned::to_owned),
            spectrum: GenericSpectrum::try_with_capacity(100.0, 0)?,
        })
    }
}
