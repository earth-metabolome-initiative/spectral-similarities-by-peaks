//! Spectrum row sampling and peak-count preprocessing.

use anyhow::Result;
use mass_spectrometry::prelude::{
    GenericSpectrum, SiriusMergeClosePeaks, SpectralProcessor, SpectrumAlloc,
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::{model::LoadedRecord, progress::progress_bar};

/// Select query row indices, optionally applying deterministic subsampling.
pub fn select_query_ids(n_records: usize, row_sample_size: Option<usize>, seed: u64) -> Vec<usize> {
    let mut ids = (0..n_records).collect::<Vec<_>>();
    if let Some(sample_size) = row_sample_size.filter(|&sample_size| sample_size < n_records) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        ids.shuffle(&mut rng);
        ids.truncate(sample_size);
        ids.sort_unstable();
    }
    ids
}

/// Select reference row indices, optionally applying deterministic subsampling.
pub fn select_reference_ids(
    n_records: usize,
    reference_sample_size: Option<usize>,
    seed: u64,
) -> Vec<usize> {
    select_query_ids(
        n_records,
        reference_sample_size,
        seed ^ 0xA5A5_5A5A_D3C3_B4B4,
    )
}

/// Build spectra truncated to the requested top-intensity peak count.
pub fn prepare_spectra(
    records: &[LoadedRecord],
    peak_count: usize,
    mz_tolerance: f64,
    merge_close_peaks: bool,
) -> Result<Vec<GenericSpectrum>> {
    let progress = progress_bar(
        u64::try_from(records.len()).unwrap_or(u64::MAX),
        format!("preparing top {peak_count} peaks"),
    );
    let spectra = records
        .par_iter()
        .map(|record| {
            top_peaks_spectrum(
                &record.spectrum,
                peak_count,
                mz_tolerance,
                merge_close_peaks,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    progress.finish_and_clear();
    Ok(spectra)
}

/// Apply top-peak selection and optional close-peak merging to one spectrum.
fn top_peaks_spectrum(
    spectrum: &GenericSpectrum,
    peak_count: usize,
    mz_tolerance: f64,
    merge_close_peaks: bool,
) -> Result<GenericSpectrum> {
    let truncated = spectrum.top_k_peaks(peak_count)?;

    if merge_close_peaks {
        let processor = SiriusMergeClosePeaks::<f64>::new_with_precision(mz_tolerance)?;
        return Ok(processor.process(&truncated));
    }
    Ok(truncated)
}
