//! Command-line interface definitions for the experiment binary.

use std::{path::PathBuf, str::FromStr};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use crate::model::{DatasetName, Metric, SimilarityConfig};

/// Similarity configurations evaluated when no explicit configuration is given.
const DEFAULT_SIMILARITY_CONFIGS: &[&str] = &[
    "cosine:0.0:1.0",
    "cosine:0.0:0.5",
    "cosine:1.0:0.5",
    "entropy:0.0:1.0:true",
];

#[derive(Debug, Parser)]
#[command(author, version, about)]
/// Parsed top-level command-line arguments.
pub struct Cli {
    /// Selected subcommand.
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
/// Available command-line subcommands.
pub enum Commands {
    /// Compute top-neighbor similarity distributions across peak cutoffs.
    Scan(ScanArgs),
}

#[derive(Debug, Parser)]
/// Arguments for the `scan` subcommand.
pub struct ScanArgs {
    /// Dataset to load through mascot-rs.
    #[arg(long, value_enum)]
    pub dataset: DatasetName,
    /// Directory where mascot-rs caches downloaded datasets.
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,
    /// Output directory for binary artifacts.
    #[arg(long, default_value = "results")]
    pub output_dir: PathBuf,
    /// Similarity config as `metric:mz_power:intensity_power[:weighted]`.
    #[arg(long = "similarity-config", default_values = DEFAULT_SIMILARITY_CONFIGS)]
    pub similarity_configs: Vec<SimilarityConfig>,
    /// Product m/z tolerance in Da.
    #[arg(long, default_value_t = 0.05)]
    pub mz_tolerance: f64,
    /// Top non-self neighbors retained per query row.
    #[arg(long, default_value_t = 10)]
    pub neighbors: usize,
    /// Minimum score retained by top-k searches.
    #[arg(long, default_value_t = 0.0)]
    pub score_threshold: f64,
    /// Number of fixed-width histogram bins over the [0, 1] score range.
    #[arg(long, default_value_t = 100)]
    pub histogram_bins: usize,
    /// Optional precursor m/z tolerance in Da for candidate filtering.
    #[arg(long)]
    pub pepmass_tolerance: Option<f64>,
    /// Representatives sampled per NPC pathway for cosine-sum pathway scoring.
    #[arg(long, default_value_t = 0)]
    pub pathway_representatives_per_class: usize,
    /// Deterministic sample of query rows. The index still contains all loaded spectra.
    #[arg(long)]
    pub row_sample_size: Option<usize>,
    /// Deterministic sample of reference rows used as search columns.
    #[arg(long)]
    pub reference_sample_size: Option<usize>,
    /// Limit loaded spectra before building indexes, useful for smoke tests.
    #[arg(long)]
    pub max_spectra: Option<usize>,
    /// GeMS-A10 part numbers to load, comma-separated. Omit to load all parts.
    #[arg(long, value_delimiter = ',')]
    pub gems_parts: Option<Vec<u8>>,
    /// RNG seed for query-row sampling.
    #[arg(long, default_value_t = 13)]
    pub seed: u64,
    /// Keep close peaks instead of merging them before indexing.
    #[arg(long, default_value_t = false)]
    pub no_merge_close_peaks: bool,
}

impl ScanArgs {
    /// Validate cross-field constraints and normalize list-style arguments.
    ///
    /// # Errors
    ///
    /// Returns an error when numeric arguments are outside the supported range.
    pub fn validate(&mut self) -> Result<()> {
        if self.neighbors == 0 {
            bail!("--neighbors must be greater than zero");
        }
        if !self.mz_tolerance.is_finite() || self.mz_tolerance < 0.0 {
            bail!("--mz-tolerance must be finite and non-negative");
        }
        if !self.score_threshold.is_finite() {
            bail!("--score-threshold must be finite");
        }
        if self.histogram_bins == 0 {
            bail!("--histogram-bins must be greater than zero");
        }
        if let Some(pepmass_tolerance) = self.pepmass_tolerance {
            if !pepmass_tolerance.is_finite() || pepmass_tolerance < 0.0 {
                bail!("--pepmass-tolerance must be finite and non-negative");
            }
        }
        Ok(())
    }
}

impl ValueEnum for DatasetName {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Harmonized, Self::Gems, Self::SyntheticSmoke]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            Self::Harmonized => Some(clap::builder::PossibleValue::new("harmonized")),
            Self::Gems => Some(clap::builder::PossibleValue::new("gems")),
            Self::SyntheticSmoke => {
                Some(clap::builder::PossibleValue::new("synthetic-smoke").hide(true))
            }
        }
    }
}

impl FromStr for SimilarityConfig {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let parts = value.split(':').collect::<Vec<_>>();
        if parts.len() < 3 || parts.len() > 4 {
            bail!("expected metric:mz_power:intensity_power[:weighted], got {value}");
        }

        let metric = match parts[0] {
            "cosine" => Metric::Cosine,
            "entropy" => Metric::Entropy,
            other => bail!("unknown similarity metric {other}"),
        };
        let mz_power = parts[1]
            .parse::<f64>()
            .with_context(|| format!("invalid mz_power in {value}"))?;
        let intensity_power = parts[2]
            .parse::<f64>()
            .with_context(|| format!("invalid intensity_power in {value}"))?;
        let entropy_weighted = parts
            .get(3)
            .map(|raw| raw.parse::<bool>())
            .transpose()
            .with_context(|| format!("invalid entropy weighted flag in {value}"))?
            .unwrap_or(true);

        if metric == Metric::Cosine && parts.len() == 4 {
            bail!("cosine configs do not take a weighted flag: {value}");
        }

        Ok(Self {
            metric,
            mz_power,
            intensity_power,
            entropy_weighted,
        })
    }
}
