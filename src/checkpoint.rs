//! Serde-backed checkpoints for resumable score-distribution scans.

use std::{
    cmp::Ordering,
    fs,
    io::{BufReader, BufWriter, ErrorKind, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    cli::ScanArgs,
    model::{LoadedRecord, ScoreDistribution, SimilarityConfig},
};

/// Current on-disk checkpoint format version.
const FORMAT_VERSION: u32 = 1;
/// Directory under `output_dir` that stores distribution checkpoints.
const CHECKPOINT_DIR: &str = "distributions";
/// Compressed checkpoint file suffix.
const CHECKPOINT_EXTENSION: &str = "bincode.zst";
/// Legacy uncompressed checkpoint file suffix.
const LEGACY_CHECKPOINT_EXTENSION: &str = "bincode";
/// Zstandard compression level for distribution checkpoints.
const CHECKPOINT_COMPRESSION_LEVEL: i32 = 6;
/// Initial state for the stable `FNV-1a` fingerprint hash.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// Multiplication prime for the stable `FNV-1a` fingerprint hash.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Scan-level fingerprint fields shared by every similarity configuration.
#[derive(Debug, Clone)]
pub struct CheckpointBase {
    /// Dataset label selected by the scan.
    dataset: String,
    /// Number of loaded records after any `--max-spectra` truncation.
    record_count: usize,
    /// Stable ordered hash of the loaded record identifiers.
    record_ids_hash: u64,
    /// Number of selected query rows.
    query_ids_len: usize,
    /// Stable ordered hash of selected query row indices.
    query_ids_hash: u64,
    /// Number of selected reference rows.
    reference_ids_len: usize,
    /// Stable ordered hash of selected reference row indices.
    reference_ids_hash: u64,
}

impl CheckpointBase {
    /// Build the scan-level checkpoint fingerprint base.
    #[must_use]
    pub fn new(
        args: &ScanArgs,
        records: &[LoadedRecord],
        query_ids: &[usize],
        reference_ids: &[usize],
    ) -> Self {
        Self {
            dataset: args.dataset.as_str().to_string(),
            record_count: records.len(),
            record_ids_hash: hash_record_ids(records),
            query_ids_len: query_ids.len(),
            query_ids_hash: hash_usize_slice(query_ids),
            reference_ids_len: reference_ids.len(),
            reference_ids_hash: hash_usize_slice(reference_ids),
        }
    }

    /// Build the full run fingerprint for one similarity configuration.
    #[must_use]
    pub fn fingerprint(
        &self,
        args: &ScanArgs,
        config: &SimilarityConfig,
        config_name: &str,
    ) -> RunFingerprint {
        RunFingerprint {
            dataset: self.dataset.clone(),
            record_count: self.record_count,
            record_ids_hash: self.record_ids_hash,
            query_ids_len: self.query_ids_len,
            query_ids_hash: self.query_ids_hash,
            reference_ids_len: self.reference_ids_len,
            reference_ids_hash: self.reference_ids_hash,
            config: config_name.to_string(),
            metric: config.metric_label().to_string(),
            mz_power_bits: config.mz_power.to_bits(),
            intensity_power_bits: config.intensity_power.to_bits(),
            entropy_weighted: config.entropy_weighted,
            neighbors: args.neighbors,
            score_threshold_bits: args.score_threshold.to_bits(),
            mz_tolerance_bits: args.mz_tolerance.to_bits(),
            pepmass_tolerance_bits: args.pepmass_tolerance.map(f64::to_bits),
            max_spectra: args.max_spectra,
            row_sample_size: args.row_sample_size,
            reference_sample_size: args.reference_sample_size,
            seed: args.seed,
            gems_parts: args.gems_parts.clone(),
            no_merge_close_peaks: args.no_merge_close_peaks,
        }
    }
}

/// Full fingerprint for deciding whether a checkpoint belongs to this run.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Serialize)]
pub struct RunFingerprint {
    /// Dataset label selected by the scan.
    dataset: String,
    /// Number of loaded records after any `--max-spectra` truncation.
    record_count: usize,
    /// Stable ordered hash of the loaded record identifiers.
    record_ids_hash: u64,
    /// Number of selected query rows.
    query_ids_len: usize,
    /// Stable ordered hash of selected query row indices.
    query_ids_hash: u64,
    /// Number of selected reference rows.
    reference_ids_len: usize,
    /// Stable ordered hash of selected reference row indices.
    reference_ids_hash: u64,
    /// Similarity configuration label.
    config: String,
    /// Similarity metric label.
    metric: String,
    /// Exact `f64` bit pattern for the m/z exponent.
    mz_power_bits: u64,
    /// Exact `f64` bit pattern for the intensity exponent.
    intensity_power_bits: u64,
    /// Entropy weighted flag.
    entropy_weighted: bool,
    /// Number of top non-self neighbors retained per query.
    neighbors: usize,
    /// Exact `f64` bit pattern for the score threshold.
    score_threshold_bits: u64,
    /// Exact `f64` bit pattern for the product m/z tolerance.
    mz_tolerance_bits: u64,
    /// Exact optional `f64` bit pattern for the precursor m/z tolerance.
    pepmass_tolerance_bits: Option<u64>,
    /// Optional loaded-spectrum limit.
    max_spectra: Option<usize>,
    /// Optional query-row sample size.
    row_sample_size: Option<usize>,
    /// Optional reference-column sample size.
    reference_sample_size: Option<usize>,
    /// Random seed for deterministic sampling.
    seed: u64,
    /// Optional selected `GeMS-A10` parts.
    gems_parts: Option<Vec<u8>>,
    /// Whether close peaks were left unmerged.
    no_merge_close_peaks: bool,
}

/// Serde representation of one score-distribution checkpoint.
#[derive(Debug, Deserialize, Serialize)]
struct DistributionCheckpoint {
    /// On-disk format version.
    format_version: u32,
    /// Dataset label selected by the scan.
    dataset: String,
    /// Similarity configuration label.
    config: String,
    /// Similarity metric label.
    metric: String,
    /// Retained peak count.
    peak_count: usize,
    /// Number of sorted scores stored in this checkpoint.
    n_scores: usize,
    /// Arithmetic mean of the stored scores.
    mean: f64,
    /// Sorted score distribution.
    scores: Vec<f64>,
    /// Full run fingerprint used to validate checkpoint reuse.
    fingerprint: RunFingerprint,
}

impl DistributionCheckpoint {
    /// Build a checkpoint from an in-memory score distribution.
    fn from_distribution(
        distribution: &ScoreDistribution,
        dataset: &str,
        config: &str,
        metric: &str,
        fingerprint: &RunFingerprint,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            dataset: dataset.to_string(),
            config: config.to_string(),
            metric: metric.to_string(),
            peak_count: distribution.peak_count,
            n_scores: distribution.scores.len(),
            mean: distribution.mean,
            scores: distribution.scores.clone(),
            fingerprint: fingerprint.clone(),
        }
    }

    /// Validate a checkpoint and return the cached distribution when it matches.
    fn into_distribution(
        self,
        dataset: &str,
        config: &str,
        metric: &str,
        peak_count: usize,
        fingerprint: &RunFingerprint,
    ) -> Option<ScoreDistribution> {
        if self.format_version != FORMAT_VERSION
            || self.dataset != dataset
            || self.config != config
            || self.metric != metric
            || self.peak_count != peak_count
            || self.n_scores != self.scores.len()
            || self.fingerprint != *fingerprint
            || !scores_are_sorted(&self.scores)
        {
            return None;
        }

        Some(ScoreDistribution {
            peak_count: self.peak_count,
            scores: self.scores,
            mean: self.mean,
        })
    }
}

/// Return the checkpoint file path for one config and retained peak count.
#[must_use]
pub fn checkpoint_path(output_dir: &Path, config_name: &str, peak_count: usize) -> PathBuf {
    checkpoint_path_with_extension(output_dir, config_name, peak_count, CHECKPOINT_EXTENSION)
}

/// Return the legacy uncompressed checkpoint path for one config and peak count.
fn legacy_checkpoint_path(output_dir: &Path, config_name: &str, peak_count: usize) -> PathBuf {
    checkpoint_path_with_extension(
        output_dir,
        config_name,
        peak_count,
        LEGACY_CHECKPOINT_EXTENSION,
    )
}

/// Return a checkpoint path for one config, peak count, and file suffix.
fn checkpoint_path_with_extension(
    output_dir: &Path,
    config_name: &str,
    peak_count: usize,
    extension: &str,
) -> PathBuf {
    output_dir
        .join(CHECKPOINT_DIR)
        .join(config_name)
        .join(format!("top_{peak_count:03}.{extension}"))
}

/// Load a valid checkpoint for one distribution, returning `None` on any mismatch.
///
/// Legacy uncompressed `.bincode` checkpoints are still accepted. When one is
/// reused successfully, it is migrated to the compressed `.bincode.zst` form
/// and the old file is removed after the compressed replacement is written.
pub fn load_distribution(
    output_dir: &Path,
    dataset: &str,
    config: &str,
    metric: &str,
    peak_count: usize,
    fingerprint: &RunFingerprint,
) -> Option<ScoreDistribution> {
    let path = checkpoint_path(output_dir, config, peak_count);
    if let Ok(file) = fs::File::open(path) {
        let reader = BufReader::new(file);
        if let Ok(decoder) = zstd::stream::Decoder::new(reader) {
            if let Ok(checkpoint) = bincode::deserialize_from::<_, DistributionCheckpoint>(decoder)
            {
                if let Some(distribution) =
                    checkpoint.into_distribution(dataset, config, metric, peak_count, fingerprint)
                {
                    return Some(distribution);
                }
            }
        }
    }

    let legacy_path = legacy_checkpoint_path(output_dir, config, peak_count);
    let legacy_file = fs::File::open(&legacy_path).ok()?;
    let legacy_reader = BufReader::new(legacy_file);
    let legacy_checkpoint =
        bincode::deserialize_from::<_, DistributionCheckpoint>(legacy_reader).ok()?;
    let distribution =
        legacy_checkpoint.into_distribution(dataset, config, metric, peak_count, fingerprint)?;
    if store_distribution(
        output_dir,
        dataset,
        config,
        metric,
        &distribution,
        fingerprint,
    )
    .is_ok()
    {
        match fs::remove_file(legacy_path) {
            Ok(()) | Err(_) => {}
        }
    }
    Some(distribution)
}

/// Store one distribution checkpoint atomically in the scan output directory.
///
/// # Errors
///
/// Returns an error when the parent directory cannot be created, serialization
/// fails, or the temporary file cannot be renamed into place.
pub fn store_distribution(
    output_dir: &Path,
    dataset: &str,
    config: &str,
    metric: &str,
    distribution: &ScoreDistribution,
    fingerprint: &RunFingerprint,
) -> Result<()> {
    let path = checkpoint_path(output_dir, config, distribution.peak_count);
    let parent = path
        .parent()
        .with_context(|| format!("checkpoint path {} has no parent", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let temporary_path = temporary_checkpoint_path(&path);
    let file = fs::File::create(&temporary_path)
        .with_context(|| format!("creating {}", temporary_path.display()))?;
    let mut writer = BufWriter::new(file);
    let checkpoint = DistributionCheckpoint::from_distribution(
        distribution,
        dataset,
        config,
        metric,
        fingerprint,
    );
    let mut encoder = zstd::stream::Encoder::new(&mut writer, CHECKPOINT_COMPRESSION_LEVEL)
        .with_context(|| format!("opening zstd encoder for {}", temporary_path.display()))?;
    bincode::serialize_into(&mut encoder, &checkpoint)
        .with_context(|| format!("serializing {}", temporary_path.display()))?;
    encoder
        .finish()
        .with_context(|| format!("finishing zstd encoder for {}", temporary_path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing {}", temporary_path.display()))?;
    rename_checkpoint(&temporary_path, &path)
        .with_context(|| format!("moving {} to {}", temporary_path.display(), path.display()))
}

/// Return a process-specific temporary checkpoint path next to the final file.
fn temporary_checkpoint_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().map_or_else(
        || "checkpoint".into(),
        |file_name| file_name.to_string_lossy(),
    );
    path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()))
}

/// Rename a temporary checkpoint into place, replacing the previous checkpoint.
fn rename_checkpoint(temporary_path: &Path, path: &Path) -> std::io::Result<()> {
    match fs::rename(temporary_path, path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            fs::remove_file(path)?;
            fs::rename(temporary_path, path)
        }
        Err(error) => Err(error),
    }
}

/// Return whether scores are sorted by the same ordering used after computation.
fn scores_are_sorted(scores: &[f64]) -> bool {
    scores
        .windows(2)
        .all(|window| window[0].total_cmp(&window[1]) != Ordering::Greater)
}

/// Compute a stable ordered hash over loaded record identifiers.
fn hash_record_ids(records: &[LoadedRecord]) -> u64 {
    let mut hash = FNV_OFFSET;
    update_usize(&mut hash, records.len());
    for record in records {
        update_bytes(&mut hash, record.id.as_bytes());
    }
    hash
}

/// Compute a stable ordered hash over a slice of `usize` values.
fn hash_usize_slice(values: &[usize]) -> u64 {
    let mut hash = FNV_OFFSET;
    update_usize(&mut hash, values.len());
    for &value in values {
        update_usize(&mut hash, value);
    }
    hash
}

/// Add one length-delimited byte slice to an `FNV-1a` hash.
fn update_bytes(hash: &mut u64, bytes: &[u8]) {
    update_usize(hash, bytes.len());
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Add one `usize` value to an `FNV-1a` hash.
fn update_usize(hash: &mut u64, value: usize) {
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

#[cfg(test)]
/// Unit tests for resumable distribution checkpoint storage.
mod tests {
    use std::{
        fs,
        io::{BufWriter, Read, Write},
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use anyhow::Result;

    use super::{
        DistributionCheckpoint, RunFingerprint, checkpoint_path, legacy_checkpoint_path,
        load_distribution, store_distribution,
    };
    use crate::model::ScoreDistribution;

    #[test]
    /// Stored distributions round-trip through the serde checkpoint format.
    fn stored_distribution_roundtrips() -> Result<()> {
        let root = temp_root("roundtrip")?;
        let fingerprint = test_fingerprint("cosine_mz0.000_int1.000");
        let distribution = ScoreDistribution {
            peak_count: 7,
            scores: vec![0.1, 0.2, 0.4],
            mean: 0.25,
        };

        store_distribution(
            &root,
            "synthetic-smoke",
            "cosine_mz0.000_int1.000",
            "cosine",
            &distribution,
            &fingerprint,
        )?;
        let loaded = load_distribution(
            &root,
            "synthetic-smoke",
            "cosine_mz0.000_int1.000",
            "cosine",
            7,
            &fingerprint,
        );

        assert_eq!(loaded, Some(distribution));
        assert_zstd_checkpoint(&checkpoint_path(&root, "cosine_mz0.000_int1.000", 7))?;
        assert!(!temporary_files_exist(&root)?);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Legacy uncompressed checkpoints are loaded and migrated to zstd.
    fn legacy_uncompressed_distribution_is_migrated() -> Result<()> {
        let root = temp_root("legacy")?;
        let fingerprint = test_fingerprint("cosine_mz0.000_int1.000");
        let distribution = ScoreDistribution {
            peak_count: 9,
            scores: vec![0.1, 0.2, 0.4],
            mean: 0.25,
        };
        let legacy_path = legacy_checkpoint_path(&root, "cosine_mz0.000_int1.000", 9);
        let parent = legacy_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("legacy checkpoint path has no parent"))?;
        fs::create_dir_all(parent)?;
        let file = fs::File::create(&legacy_path)?;
        let mut writer = BufWriter::new(file);
        let checkpoint = DistributionCheckpoint::from_distribution(
            &distribution,
            "synthetic-smoke",
            "cosine_mz0.000_int1.000",
            "cosine",
            &fingerprint,
        );
        bincode::serialize_into(&mut writer, &checkpoint)?;
        writer.flush()?;

        let loaded = load_distribution(
            &root,
            "synthetic-smoke",
            "cosine_mz0.000_int1.000",
            "cosine",
            9,
            &fingerprint,
        );

        assert_eq!(loaded, Some(distribution));
        assert!(!legacy_path.exists());
        assert_zstd_checkpoint(&checkpoint_path(&root, "cosine_mz0.000_int1.000", 9))?;
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Fingerprint mismatches are treated as cache misses.
    fn mismatched_fingerprint_is_ignored() -> Result<()> {
        let root = temp_root("mismatch")?;
        let fingerprint = test_fingerprint("cosine_mz0.000_int1.000");
        let distribution = ScoreDistribution {
            peak_count: 1,
            scores: vec![0.5],
            mean: 0.5,
        };
        store_distribution(
            &root,
            "synthetic-smoke",
            "cosine_mz0.000_int1.000",
            "cosine",
            &distribution,
            &fingerprint,
        )?;

        let other_fingerprint = test_fingerprint("cosine_mz1.000_int0.500");
        let loaded = load_distribution(
            &root,
            "synthetic-smoke",
            "cosine_mz0.000_int1.000",
            "cosine",
            1,
            &other_fingerprint,
        );

        assert_eq!(loaded, None);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Checkpoint paths are grouped by config and zero-padded peak count.
    fn checkpoint_paths_are_stable() {
        assert_eq!(
            checkpoint_path(
                &PathBuf::from("out"),
                "entropy_mz0.000_int1.000_weightedtrue",
                12
            ),
            PathBuf::from(
                "out/distributions/entropy_mz0.000_int1.000_weightedtrue/top_012.bincode.zst"
            )
        );
    }

    /// Assert that a checkpoint starts with the zstd frame magic number.
    fn assert_zstd_checkpoint(path: &Path) -> Result<()> {
        let mut magic = [0_u8; 4];
        fs::File::open(path)?.read_exact(&mut magic)?;
        assert_eq!(magic, [0x28, 0xb5, 0x2f, 0xfd]);
        Ok(())
    }

    /// Return a deterministic test fingerprint with one variable config label.
    fn test_fingerprint(config: &str) -> RunFingerprint {
        RunFingerprint {
            dataset: "synthetic-smoke".to_string(),
            record_count: 3,
            record_ids_hash: 11,
            query_ids_len: 2,
            query_ids_hash: 13,
            reference_ids_len: 3,
            reference_ids_hash: 17,
            config: config.to_string(),
            metric: "cosine".to_string(),
            mz_power_bits: 0.0_f64.to_bits(),
            intensity_power_bits: 1.0_f64.to_bits(),
            entropy_weighted: true,
            neighbors: 3,
            score_threshold_bits: 0.0_f64.to_bits(),
            mz_tolerance_bits: 0.05_f64.to_bits(),
            pepmass_tolerance_bits: None,
            max_spectra: None,
            row_sample_size: Some(2),
            reference_sample_size: Some(3),
            seed: 42,
            gems_parts: None,
            no_merge_close_peaks: false,
        }
    }

    /// Return a unique temporary directory for a checkpoint unit test.
    fn temp_root(label: &str) -> Result<PathBuf> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let root = std::env::temp_dir().join(format!(
            "spectral-similarities-checkpoint-{label}-{}-{}",
            std::process::id(),
            timestamp.as_nanos()
        ));
        fs::create_dir_all(&root)?;
        Ok(root)
    }

    /// Return whether any temporary checkpoint files remain below the root.
    fn temporary_files_exist(root: &Path) -> Result<bool> {
        for config_entry in fs::read_dir(root.join("distributions"))? {
            for checkpoint_entry in fs::read_dir(config_entry?.path())? {
                if checkpoint_entry?
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.to_string_lossy().contains("tmp"))
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}
