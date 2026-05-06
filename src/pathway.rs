//! `NPC` pathway scoring through fixed representative spectra.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use mass_spectrometry::prelude::{
    FlashCosineThresholdIndex, GenericSpectrum, SearchState, SpectraIndexBuilder,
};
use rayon::prelude::*;

use crate::{
    cli::ScanArgs,
    model::{LoadedRecord, Metric, PathwayPrediction, PathwayScore, SimilarityConfig},
    progress::progress_bar,
};

/// Score query spectra against fixed `NPC` pathway representatives.
pub fn score_pathway_representatives(
    args: &ScanArgs,
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
    let index = builder.build(&reference_spectra)?;
    let pathways = representative_pathways(&representatives);
    let config_name = config.name();
    let progress = progress_bar(
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
                progress.inc(1);
                rows
            },
        )
        .collect::<Result<Vec<_>>>()?;
    progress.finish_and_clear();

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
        let Some(pathway) = &record.npc_pathway else {
            continue;
        };
        let selected = selected_per_pathway.entry(pathway.clone()).or_default();
        if *selected >= representatives_per_class {
            continue;
        }
        representatives.push(PathwayRepresentative {
            record_index,
            pathway: pathway.clone(),
        });
        *selected += 1;
    }
    representatives
}

/// Count representatives per pathway.
fn representative_pathways(representatives: &[PathwayRepresentative]) -> BTreeMap<String, usize> {
    let mut pathways = BTreeMap::new();
    for representative in representatives {
        *pathways.entry(representative.pathway.clone()).or_default() += 1;
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
        if let Some(pathway) = scored_pathways.get_mut(&representative.pathway) {
            pathway.score += hit.score;
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
    let is_correct = query
        .npc_pathway
        .as_ref()
        .zip(best_pathway.as_ref())
        .map(|(expected, predicted)| expected == predicted);
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

/// Selected pathway representative metadata.
struct PathwayRepresentative {
    /// Original record index in the loaded dataset.
    record_index: usize,
    /// Pathway represented by the record.
    pathway: String,
}

/// Accumulated score for one candidate pathway.
struct PathwayAggregate {
    /// Number of representative spectra assigned to the pathway.
    representatives: usize,
    /// Sum of query cosine similarities to representatives.
    score: f64,
}
