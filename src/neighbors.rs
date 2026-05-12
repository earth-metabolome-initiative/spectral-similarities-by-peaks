//! Similarity index construction and top-neighbor row collection.

use anyhow::{Context, Result};
use mass_spectrometry::prelude::{
    FlashCosineSelfSimilarityIndex, FlashCosineThresholdIndex, FlashEntropyIndex,
    FlashSearchResult, GenericSpectrum, SearchState, SimilarityComputationError,
    SpectraIndexBuilder, Spectrum, TopKSearchState,
};
use rayon::prelude::*;

use crate::{
    cli::ScanArgs,
    model::{LoadedRecord, Metric, NeighborHit, SimilarityConfig},
    progress::ScanProgress,
};

/// Inputs for one top-neighbor search batch.
pub struct SearchBatch<'a> {
    /// Scan arguments that configure tolerances and retained neighbors.
    pub args: &'a ScanArgs,
    /// Shared scan progress reporter.
    pub progress: &'a ScanProgress,
    /// Similarity configuration being evaluated.
    pub config: &'a SimilarityConfig,
    /// Retained top-intensity peak count.
    pub peak_count: usize,
    /// Loaded metadata records.
    pub records: &'a [LoadedRecord],
    /// Prepared spectra aligned to `records`.
    pub spectra: &'a [GenericSpectrum<f32>],
    /// Query row ids for this batch.
    pub query_ids: &'a [usize],
    /// Reference row ids for this batch.
    pub reference_ids: &'a [usize],
}

/// Compute retained non-self neighbor hits for one similarity configuration.
pub fn compute_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    match batch.config.metric {
        Metric::Cosine => compute_cosine_neighbors(batch),
        Metric::ModifiedCosine => compute_modified_cosine_neighbors(batch),
        Metric::Entropy => compute_entropy_neighbors(batch),
        Metric::ModifiedEntropy => compute_modified_entropy_neighbors(batch),
    }
}

/// Compute cosine neighbors using the crate-provided self-similarity row index.
fn compute_cosine_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    if !uses_full_reference_panel(batch.reference_ids, batch.records.len()) {
        return compute_cosine_reference_neighbors(batch);
    }

    let mut builder = FlashCosineSelfSimilarityIndex::<f32>::builder()
        .mz_power(batch.config.mz_power)
        .intensity_power(batch.config.intensity_power)
        .mz_tolerance(batch.args.mz_tolerance)
        .score_threshold(batch.args.score_threshold)
        .top_k(batch.args.neighbors)
        .parallel();
    let pepmass_tolerance = batch
        .args
        .pepmass_tolerance
        .unwrap_or_else(|| broad_pepmass_tolerance(batch.spectra));
    builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    let config_name = batch.config.name();
    let index_progress = batch.progress.spinner(format!(
        "building {config_name} self-similarity index for top {} peaks",
        batch.peak_count
    ));
    let index = builder.build(batch.spectra)?;
    index_progress.finish();
    let query_ids = batch
        .query_ids
        .iter()
        .map(|&query_id| u32::try_from(query_id).context("query index does not fit u32"))
        .collect::<Result<Vec<_>>>()?;

    let scoring_progress = batch.progress.bar(
        u64::try_from(query_ids.len()).unwrap_or(u64::MAX),
        format!("scoring {config_name} top {} peaks", batch.peak_count),
    );
    let chunks = index
        .rows()
        .ids(&query_ids)
        .into_par_iter()
        .map(|row| -> Result<Vec<NeighborHit>> {
            let (_query_id, hits) = row.context("computing cosine self-similarity row")?;
            let hits = hits
                .into_iter()
                .map(|hit| {
                    usize::try_from(hit.spectrum_id).context("target index does not fit usize")?;
                    Ok(NeighborHit { score: hit.score })
                })
                .collect::<Result<Vec<_>>>();
            scoring_progress.inc(1);
            hits
        })
        .collect::<Result<Vec<_>>>()?;
    scoring_progress.finish();

    Ok(chunks.into_iter().flatten().collect())
}

/// Compute cosine neighbors against a fixed sampled reference panel.
fn compute_cosine_reference_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    let reference_spectra = reference_spectra(batch.spectra, batch.reference_ids);
    compute_cosine_neighbors_against_references(batch, &reference_spectra)
}

/// Compute cosine neighbors against already selected reference spectra.
fn compute_cosine_neighbors_against_references(
    batch: &SearchBatch<'_>,
    reference_spectra: &[GenericSpectrum<f32>],
) -> Result<Vec<NeighborHit>> {
    let mut builder = FlashCosineThresholdIndex::<f32>::builder()
        .mz_power(batch.config.mz_power)
        .intensity_power(batch.config.intensity_power)
        .mz_tolerance(batch.args.mz_tolerance)
        .score_threshold(batch.args.score_threshold)
        .parallel();
    if let Some(pepmass_tolerance) = batch.args.pepmass_tolerance {
        builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    }
    let config_name = batch.config.name();
    let index_progress = batch.progress.spinner(format!(
        "building {config_name} reference index for top {} peaks",
        batch.peak_count
    ));
    let index = builder.build(reference_spectra)?;
    index_progress.finish();
    collect_external_neighbors(
        batch,
        || index.new_search_state(),
        |query, state, top_k_state, emit| {
            index.for_each_top_k_with_state(
                query,
                batch.args.neighbors + 1,
                state,
                top_k_state,
                emit,
            )
        },
    )
}

/// Compute modified-cosine neighbors with the modified top-k index path.
fn compute_modified_cosine_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    if uses_full_reference_panel(batch.reference_ids, batch.records.len()) {
        return compute_modified_cosine_neighbors_against_references(batch, batch.spectra);
    }
    let reference_spectra = reference_spectra(batch.spectra, batch.reference_ids);
    compute_modified_cosine_neighbors_against_references(batch, &reference_spectra)
}

/// Compute modified-cosine neighbors against already selected reference spectra.
fn compute_modified_cosine_neighbors_against_references(
    batch: &SearchBatch<'_>,
    reference_spectra: &[GenericSpectrum<f32>],
) -> Result<Vec<NeighborHit>> {
    let mut builder = FlashCosineThresholdIndex::<f32>::builder()
        .mz_power(batch.config.mz_power)
        .intensity_power(batch.config.intensity_power)
        .mz_tolerance(batch.args.mz_tolerance)
        .score_threshold(batch.args.score_threshold)
        .parallel();
    if let Some(pepmass_tolerance) = batch.args.pepmass_tolerance {
        builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    }
    let config_name = batch.config.name();
    let index_progress = batch.progress.spinner(format!(
        "building {config_name} reference index for top {} peaks",
        batch.peak_count
    ));
    let index = builder.build(reference_spectra)?;
    index_progress.finish();
    collect_external_neighbors(
        batch,
        || index.new_search_state(),
        |query, state, top_k_state, emit| {
            index.for_each_modified_top_k_with_state(
                query,
                batch.args.neighbors + 1,
                state,
                top_k_state,
                emit,
            )
        },
    )
}

/// Return a broad precursor tolerance that preserves no-filter semantics.
fn broad_pepmass_tolerance(spectra: &[GenericSpectrum<f32>]) -> f64 {
    let mut min_precursor_mz = f64::INFINITY;
    let mut max_precursor_mz = f64::NEG_INFINITY;
    for spectrum in spectra {
        let precursor_mz = f64::from(spectrum.precursor_mz());
        min_precursor_mz = min_precursor_mz.min(precursor_mz);
        max_precursor_mz = max_precursor_mz.max(precursor_mz);
    }
    (max_precursor_mz - min_precursor_mz).max(0.0)
}

/// Compute entropy neighbors with the entropy index top-k API.
fn compute_entropy_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    if !uses_full_reference_panel(batch.reference_ids, batch.records.len()) {
        return compute_entropy_reference_neighbors(batch);
    }

    let mut builder = FlashEntropyIndex::<f32>::builder()
        .mz_power(batch.config.mz_power)
        .intensity_power(batch.config.intensity_power)
        .mz_tolerance(batch.args.mz_tolerance)
        .weighted(batch.config.entropy_weighted)
        .parallel();
    if let Some(pepmass_tolerance) = batch.args.pepmass_tolerance {
        builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    }
    let config_name = batch.config.name();
    let index_progress = batch.progress.spinner(format!(
        "building {config_name} index for top {} peaks",
        batch.peak_count
    ));
    let index = builder.build(batch.spectra)?;
    index_progress.finish();
    collect_indexed_neighbors(
        batch,
        || index.new_search_state(),
        |query_id, state, top_k_state, emit| {
            index.for_each_top_k_threshold_indexed_with_state(
                query_id,
                batch.args.neighbors + 1,
                batch.args.score_threshold,
                state,
                top_k_state,
                emit,
            )
        },
    )
}

/// Compute entropy neighbors against a fixed sampled reference panel.
fn compute_entropy_reference_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    let reference_spectra = reference_spectra(batch.spectra, batch.reference_ids);
    compute_entropy_neighbors_against_references(batch, &reference_spectra)
}

/// Compute entropy neighbors against already selected reference spectra.
fn compute_entropy_neighbors_against_references(
    batch: &SearchBatch<'_>,
    reference_spectra: &[GenericSpectrum<f32>],
) -> Result<Vec<NeighborHit>> {
    let mut builder = FlashEntropyIndex::<f32>::builder()
        .mz_power(batch.config.mz_power)
        .intensity_power(batch.config.intensity_power)
        .mz_tolerance(batch.args.mz_tolerance)
        .weighted(batch.config.entropy_weighted)
        .parallel();
    if let Some(pepmass_tolerance) = batch.args.pepmass_tolerance {
        builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    }
    let config_name = batch.config.name();
    let index_progress = batch.progress.spinner(format!(
        "building {config_name} reference index for top {} peaks",
        batch.peak_count
    ));
    let index = builder.build(reference_spectra)?;
    index_progress.finish();
    collect_external_neighbors(
        batch,
        || index.new_search_state(),
        |query, state, top_k_state, emit| {
            index.for_each_top_k_threshold_with_state(
                query,
                batch.args.neighbors + 1,
                batch.args.score_threshold,
                state,
                top_k_state,
                emit,
            )
        },
    )
}

/// Compute modified-entropy neighbors with the modified top-k index path.
fn compute_modified_entropy_neighbors(batch: &SearchBatch<'_>) -> Result<Vec<NeighborHit>> {
    if uses_full_reference_panel(batch.reference_ids, batch.records.len()) {
        return compute_modified_entropy_neighbors_against_references(batch, batch.spectra);
    }
    let reference_spectra = reference_spectra(batch.spectra, batch.reference_ids);
    compute_modified_entropy_neighbors_against_references(batch, &reference_spectra)
}

/// Compute modified-entropy neighbors against already selected reference spectra.
fn compute_modified_entropy_neighbors_against_references(
    batch: &SearchBatch<'_>,
    reference_spectra: &[GenericSpectrum<f32>],
) -> Result<Vec<NeighborHit>> {
    let mut builder = FlashEntropyIndex::<f32>::builder()
        .mz_power(batch.config.mz_power)
        .intensity_power(batch.config.intensity_power)
        .mz_tolerance(batch.args.mz_tolerance)
        .weighted(batch.config.entropy_weighted)
        .parallel();
    if let Some(pepmass_tolerance) = batch.args.pepmass_tolerance {
        builder = builder.pepmass_tolerance(pepmass_tolerance)?;
    }
    let config_name = batch.config.name();
    let index_progress = batch.progress.spinner(format!(
        "building {config_name} reference index for top {} peaks",
        batch.peak_count
    ));
    let index = builder.build(reference_spectra)?;
    index_progress.finish();
    collect_external_neighbors(
        batch,
        || index.new_search_state(),
        |query, state, top_k_state, emit| {
            index.for_each_modified_top_k_with_state(
                query,
                batch.args.neighbors + 1,
                state,
                top_k_state,
                emit,
            )
        },
    )
}

/// Collect indexed top-k rows into internal neighbor hits.
fn collect_indexed_neighbors<F, G>(
    batch: &SearchBatch<'_>,
    new_search_state: G,
    search: F,
) -> Result<Vec<NeighborHit>>
where
    F: Fn(
            u32,
            &mut SearchState,
            &mut TopKSearchState,
            &mut dyn FnMut(FlashSearchResult),
        ) -> std::result::Result<(), SimilarityComputationError>
        + Sync,
    G: Fn() -> SearchState + Sync,
{
    let config_name = batch.config.name();
    let task = batch.progress.bar(
        u64::try_from(batch.query_ids.len()).unwrap_or(u64::MAX),
        format!("scoring {config_name} top {} peaks", batch.peak_count),
    );
    let chunks = batch
        .query_ids
        .par_iter()
        .map_init(
            || (new_search_state(), TopKSearchState::new()),
            |(state, top_k_state), &query_index| -> Result<Vec<NeighborHit>> {
                let query_id =
                    u32::try_from(query_index).context("query index does not fit u32")?;
                let mut raw_hits = Vec::with_capacity(batch.args.neighbors + 1);
                search(query_id, state, top_k_state, &mut |hit| raw_hits.push(hit))?;

                let hits = raw_hits
                    .into_iter()
                    .filter(|hit| usize::try_from(hit.spectrum_id).ok() != Some(query_index))
                    .filter(|hit| hit.score >= batch.args.score_threshold)
                    .take(batch.args.neighbors)
                    .map(|hit| {
                        usize::try_from(hit.spectrum_id)
                            .context("target index does not fit usize")?;
                        Ok(NeighborHit { score: hit.score })
                    })
                    .collect::<Result<Vec<_>>>()?;
                task.inc(1);
                Ok(hits)
            },
        )
        .collect::<Result<Vec<_>>>()?;
    task.finish();

    Ok(chunks.into_iter().flatten().collect())
}

/// Collect external-query top-k rows into internal neighbor hits.
fn collect_external_neighbors<F, G>(
    batch: &SearchBatch<'_>,
    new_search_state: G,
    search: F,
) -> Result<Vec<NeighborHit>>
where
    F: Fn(
            &GenericSpectrum<f32>,
            &mut SearchState,
            &mut TopKSearchState,
            &mut dyn FnMut(FlashSearchResult),
        ) -> std::result::Result<(), SimilarityComputationError>
        + Sync,
    G: Fn() -> SearchState + Sync,
{
    let config_name = batch.config.name();
    let task = batch.progress.bar(
        u64::try_from(batch.query_ids.len()).unwrap_or(u64::MAX),
        format!("scoring {config_name} top {} peaks", batch.peak_count),
    );
    let chunks = batch
        .query_ids
        .par_iter()
        .map_init(
            || (new_search_state(), TopKSearchState::new()),
            |(state, top_k_state), &query_index| -> Result<Vec<NeighborHit>> {
                let mut raw_hits = Vec::with_capacity(batch.args.neighbors + 1);
                search(
                    &batch.spectra[query_index],
                    state,
                    top_k_state,
                    &mut |hit| {
                        raw_hits.push(hit);
                    },
                )?;

                let hits = raw_hits
                    .into_iter()
                    .filter_map(|hit| reference_hit_target(batch.reference_ids, query_index, hit))
                    .filter(|(_target_index, hit)| hit.score >= batch.args.score_threshold)
                    .take(batch.args.neighbors)
                    .map(|(_target_index, hit)| NeighborHit { score: hit.score })
                    .collect::<Vec<_>>();
                task.inc(1);
                Ok(hits)
            },
        )
        .collect::<Result<Vec<_>>>()?;
    task.finish();

    Ok(chunks.into_iter().flatten().collect())
}

/// Resolve an external hit to an original target index, dropping self hits.
fn reference_hit_target(
    reference_ids: &[usize],
    query_index: usize,
    hit: FlashSearchResult,
) -> Option<(usize, FlashSearchResult)> {
    let reference_index = usize::try_from(hit.spectrum_id).ok()?;
    let target_index = *reference_ids.get(reference_index)?;
    (target_index != query_index).then_some((target_index, hit))
}

/// Return cloned spectra for the selected reference row ids.
fn reference_spectra(
    spectra: &[GenericSpectrum<f32>],
    reference_ids: &[usize],
) -> Vec<GenericSpectrum<f32>> {
    reference_ids
        .iter()
        .map(|&reference_id| spectra[reference_id].clone())
        .collect()
}

/// Return whether the reference panel is the full dataset in row order.
fn uses_full_reference_panel(reference_ids: &[usize], n_records: usize) -> bool {
    reference_ids.len() == n_records
        && reference_ids
            .iter()
            .enumerate()
            .all(|(expected, &actual)| expected == actual)
}

#[cfg(test)]
/// Unit tests for neighbor collection on synthetic spectra.
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use anyhow::Result;
    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumMut};

    use crate::{
        cli::ScanArgs,
        model::{DatasetName, LoadedRecord, SimilarityConfig},
        progress::ScanProgress,
        spectra::{prepare_spectra, select_query_ids},
    };

    use super::{SearchBatch, compute_neighbors};

    #[test]
    /// Cosine and entropy searches return non-self positive-scoring hits.
    fn computes_cosine_and_entropy_neighbors_on_synthetic_spectra() -> Result<()> {
        let args = ScanArgs {
            dataset: DatasetName::Harmonized,
            data_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("results"),
            similarity_configs: Vec::new(),
            mz_tolerance: 0.1,
            neighbors: 1,
            score_threshold: 0.0,
            histogram_bins: 100,
            pepmass_tolerance: None,
            pathway_representatives_per_class: 0,
            row_sample_size: None,
            reference_sample_size: None,
            max_spectra: None,
            gems_parts: None,
            seed: 13,
            no_merge_close_peaks: false,
        };
        let records = vec![
            synthetic_record("a", 500.0, &[(100.0, 10.0), (200.0, 20.0)])?,
            synthetic_record("b", 501.0, &[(100.02, 10.0), (200.02, 20.0)])?,
            synthetic_record("c", 502.0, &[(100.04, 9.0), (200.04, 18.0)])?,
        ];
        let progress = ScanProgress::new();
        let spectra = prepare_spectra(&progress, &records, 2, args.mz_tolerance, true)?;
        let query_ids = select_query_ids(records.len(), None, args.seed);
        let reference_ids = select_query_ids(records.len(), None, args.seed);

        for config in [
            SimilarityConfig::from_str("cosine:0.0:1.0")?,
            SimilarityConfig::from_str("modified-cosine:0.0:1.0")?,
            SimilarityConfig::from_str("entropy:0.0:1.0:true")?,
            SimilarityConfig::from_str("modified-entropy:0.0:1.0:true")?,
            SimilarityConfig::from_str("entropy:0.0:1.0:false")?,
            SimilarityConfig::from_str("modified-entropy:0.0:1.0:false")?,
        ] {
            let hits = compute_neighbors(&SearchBatch {
                args: &args,
                progress: &progress,
                config: &config,
                peak_count: 2,
                records: &records,
                spectra: &spectra,
                query_ids: &query_ids,
                reference_ids: &reference_ids,
            })?;
            assert_eq!(hits.len(), records.len());
            assert!(hits.iter().all(|hit| hit.score > 0.0));
        }

        Ok(())
    }

    /// Build one minimal labeled record for neighbor tests.
    fn synthetic_record(id: &str, precursor_mz: f64, peaks: &[(f32, f32)]) -> Result<LoadedRecord> {
        let mut spectrum = GenericSpectrum::<f32>::try_with_capacity(precursor_mz, peaks.len())?;
        for &(mz, intensity) in peaks {
            spectrum.add_peak(mz, intensity)?;
        }
        Ok(LoadedRecord {
            id: id.to_string(),
            npc_pathway: Some("Synthetic".to_string()),
            spectrum,
        })
    }
}
