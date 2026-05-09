//! Dataset retrieval and conversion into analysis records.

use anyhow::{Context, Result};
use mascot_rs::prelude::{Dataset, MGFVec, MascotGenericFormat};
use mass_spectrometry::prelude::{GenericSpectrum, SpectrumMut};

use crate::{
    cli::ScanArgs,
    model::{DatasetName, LoadedRecord},
};

/// Standardized MGF metadata key for `NPC` pathway labels.
const NPC_PATHWAYS_KEY: &str = "NPC_PATHWAYS";

/// Number of spectra in the deterministic smoke-test dataset.
const SYNTHETIC_SMOKE_RECORDS: usize = 24;
/// Number of peaks in each synthetic smoke-test spectrum.
const SYNTHETIC_SMOKE_PEAKS: usize = 24;

/// Load the selected dataset and convert it into records used by the pipeline.
pub fn load_records(args: &ScanArgs) -> Result<Vec<LoadedRecord>> {
    match args.dataset {
        DatasetName::Harmonized => {
            let target_directory = args.data_dir.join("harmonized-top-128");
            let load = tokio_runtime()?
                .block_on(Dataset::load(
                    MGFVec::<f64>::annotated_ms2_top_128_peaks()
                        .target_directory(&target_directory)
                        .verbose(),
                ))
                .with_context(|| {
                    format!("loading harmonized data in {}", target_directory.display())
                })?;
            Ok(records_from_mgf(load.spectra()))
        }
        DatasetName::Gems => {
            let target_directory = args.data_dir.join("gems-a10-top-128");
            let mut builder = MGFVec::<f64>::gems_a10_top_128_peaks()
                .target_directory(&target_directory)
                .verbose();
            if let Some(parts) = &args.gems_parts {
                builder = builder
                    .clone()
                    .parts(parts.iter().copied())
                    .context("selecting GeMS-A10 parts")?;
            }
            let load = tokio_runtime()?
                .block_on(Dataset::load(builder))
                .with_context(|| {
                    format!("loading GeMS-A10 data in {}", target_directory.display())
                })?;
            Ok(records_from_mgf(load.spectra()))
        }
        DatasetName::SyntheticSmoke => synthetic_smoke_records(),
    }
}

/// Build the `Tokio` runtime required by the async download stack.
fn tokio_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("creating Tokio runtime for dataset loading")
}

/// Convert a parsed `MGF` collection into experiment records.
fn records_from_mgf(records: &MGFVec<f64>) -> Vec<LoadedRecord> {
    records
        .iter()
        .enumerate()
        .map(|(index, record)| record_from_mgf(index, record))
        .collect()
}

/// Convert one `MGF` record into a loaded spectrum record.
fn record_from_mgf(index: usize, record: &MascotGenericFormat<f64>) -> LoadedRecord {
    let metadata = record.metadata();
    let id = record
        .feature_id()
        .or_else(|| metadata.splash())
        .map_or_else(|| index.to_string(), ToOwned::to_owned);
    let npc_pathway = metadata
        .arbitrary_metadata_value(NPC_PATHWAYS_KEY)
        .map(ToOwned::to_owned);
    let spectrum = record.as_ref().clone();

    LoadedRecord {
        id,
        npc_pathway,
        spectrum,
    }
}

/// Build a deterministic in-memory dataset used by full smoke tests.
fn synthetic_smoke_records() -> Result<Vec<LoadedRecord>> {
    (0..SYNTHETIC_SMOKE_RECORDS)
        .map(|index| {
            let cluster = index % 4;
            let replicate = index / 4;
            let cluster_offset = usize_to_f64(cluster) * 0.01;
            let replicate_scale = 1.0 + usize_to_f64(replicate) * 0.01;
            let precursor_mz = 500.0 + usize_to_f64(cluster);
            let mut spectrum =
                GenericSpectrum::try_with_capacity(precursor_mz, SYNTHETIC_SMOKE_PEAKS)?;

            for peak_index in 0..SYNTHETIC_SMOKE_PEAKS {
                let peak_rank = SYNTHETIC_SMOKE_PEAKS - peak_index;
                let mz = usize_to_f64(peak_index).mul_add(25.0, 100.0 + cluster_offset);
                let intensity = usize_to_f64(peak_rank) * replicate_scale;
                spectrum.add_peak(mz, intensity)?;
            }

            Ok(LoadedRecord {
                id: format!("synthetic-{index:02}"),
                npc_pathway: Some(format!("pathway-{cluster}")),
                spectrum,
            })
        })
        .collect()
}

/// Convert small synthetic-data indices into `f64`.
fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}
