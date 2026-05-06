//! Shared model types for loaded spectra, similarity configurations, and CSV
//! output rows.

use std::fmt;

use mass_spectrometry::prelude::GenericSpectrum;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Dataset supported by the experiment runner.
pub enum DatasetName {
    /// Harmonized annotated `MS2` spectra retrieved through `mascot-rs`.
    Harmonized,
    /// `GeMS-A10` spectra retrieved through `mascot-rs`.
    Gems,
    /// Small deterministic in-memory dataset used only by smoke tests.
    SyntheticSmoke,
}

impl DatasetName {
    /// Return the stable lowercase label used in output rows.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Harmonized => "harmonized",
            Self::Gems => "gems-a10",
            Self::SyntheticSmoke => "synthetic-smoke",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Similarity family used to score spectrum pairs.
pub enum Metric {
    /// Linear cosine similarity.
    Cosine,
    /// Entropy similarity.
    Entropy,
}

#[derive(Debug, Clone)]
/// Parameterization of a similarity index run.
pub struct SimilarityConfig {
    /// Similarity family to compute.
    pub metric: Metric,
    /// Exponent applied to peak m/z values.
    pub mz_power: f64,
    /// Exponent applied to peak intensities.
    pub intensity_power: f64,
    /// Whether entropy scoring uses weighted entropy.
    pub entropy_weighted: bool,
}

impl SimilarityConfig {
    /// Return a filesystem- and CSV-friendly configuration label.
    #[must_use]
    pub fn name(&self) -> String {
        match self.metric {
            Metric::Cosine => format!(
                "cosine_mz{:.3}_int{:.3}",
                self.mz_power, self.intensity_power
            ),
            Metric::Entropy => format!(
                "entropy_mz{:.3}_int{:.3}_weighted{}",
                self.mz_power, self.intensity_power, self.entropy_weighted
            ),
        }
    }

    /// Return the short metric label used in output rows.
    #[must_use]
    pub const fn metric_label(&self) -> &'static str {
        match self.metric {
            Metric::Cosine => "cosine",
            Metric::Entropy => "entropy",
        }
    }
}

impl fmt::Display for SimilarityConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.metric {
            Metric::Cosine => write!(
                formatter,
                "cosine:{}:{}",
                self.mz_power, self.intensity_power
            ),
            Metric::Entropy => write!(
                formatter,
                "entropy:{}:{}:{}",
                self.mz_power, self.intensity_power, self.entropy_weighted
            ),
        }
    }
}

#[derive(Debug, Clone)]
/// Loaded spectrum plus the metadata carried into neighbor output rows.
pub struct LoadedRecord {
    /// Stable record identifier, falling back to row index when necessary.
    pub id: String,
    /// Optional human-readable spectrum name.
    pub name: Option<String>,
    /// Optional `NPC` pathway label.
    pub npc_pathway: Option<String>,
    /// Optional `NPC` superclass label.
    pub npc_superclass: Option<String>,
    /// Optional `NPC` class label.
    pub npc_class: Option<String>,
    /// Spectrum used for peak selection and similarity indexing.
    pub spectrum: GenericSpectrum,
}

#[derive(Debug, Serialize)]
/// One retained neighbor hit written to `similarities.csv`.
pub struct NeighborHit {
    /// Dataset label.
    pub dataset: String,
    /// Similarity configuration label.
    pub config: String,
    /// Similarity family label.
    pub metric: &'static str,
    /// Exponent applied to peak m/z values.
    pub mz_power: f64,
    /// Exponent applied to peak intensities.
    pub intensity_power: f64,
    /// Whether entropy scoring used weighted entropy.
    pub entropy_weighted: bool,
    /// Product m/z tolerance in Da.
    pub mz_tolerance: f64,
    /// Optional precursor m/z tolerance in Da.
    pub pepmass_tolerance: Option<f64>,
    /// Retained peak count for this run.
    pub peak_count: usize,
    /// Zero-based query row index in the loaded dataset.
    pub query_index: usize,
    /// Zero-based target row index in the loaded dataset.
    pub target_index: usize,
    /// One-based neighbor rank for the query.
    pub rank: usize,
    /// Similarity score.
    pub score: f64,
    /// Number of matched peaks reported by the index.
    pub n_matches: usize,
    /// Query record identifier.
    pub query_id: String,
    /// Target record identifier.
    pub target_id: String,
    /// Query spectrum name, if available.
    pub query_name: Option<String>,
    /// Target spectrum name, if available.
    pub target_name: Option<String>,
    /// Query `NPC` pathway label, if available.
    pub query_npc_pathway: Option<String>,
    /// Target `NPC` pathway label, if available.
    pub target_npc_pathway: Option<String>,
    /// Query `NPC` superclass label, if available.
    pub query_npc_superclass: Option<String>,
    /// Target `NPC` superclass label, if available.
    pub target_npc_superclass: Option<String>,
    /// Query `NPC` class label, if available.
    pub query_npc_class: Option<String>,
    /// Target `NPC` class label, if available.
    pub target_npc_class: Option<String>,
}

#[derive(Debug, Serialize)]
/// Summary statistics for one score distribution.
pub struct DistributionSummary {
    /// Dataset label.
    pub dataset: String,
    /// Similarity configuration label.
    pub config: String,
    /// Similarity family label.
    pub metric: &'static str,
    /// Retained peak count for this distribution.
    pub peak_count: usize,
    /// Number of scores summarized.
    pub n_scores: usize,
    /// Arithmetic mean of the scores.
    pub mean: f64,
    /// Population standard deviation of the scores.
    pub stddev: f64,
    /// Minimum score.
    pub min: f64,
    /// First percentile.
    pub q01: f64,
    /// Fifth percentile.
    pub q05: f64,
    /// Tenth percentile.
    pub q10: f64,
    /// Twenty-fifth percentile.
    pub q25: f64,
    /// Median score.
    pub median: f64,
    /// Seventy-fifth percentile.
    pub q75: f64,
    /// Ninetieth percentile.
    pub q90: f64,
    /// Ninety-fifth percentile.
    pub q95: f64,
    /// Ninety-ninth percentile.
    pub q99: f64,
    /// Maximum score.
    pub max: f64,
}

#[derive(Debug, Serialize)]
/// Nonparametric comparison between two peak-count score distributions.
pub struct DistributionComparison {
    /// Dataset label.
    pub dataset: String,
    /// Similarity configuration label.
    pub config: String,
    /// Similarity family label.
    pub metric: &'static str,
    /// First retained peak count in the comparison.
    pub peak_count_a: usize,
    /// Second retained peak count in the comparison.
    pub peak_count_b: usize,
    /// Number of scores in the lower peak-count distribution.
    pub n_scores_a: usize,
    /// Number of scores in the higher peak-count distribution.
    pub n_scores_b: usize,
    /// Mean score for the lower peak-count distribution.
    pub mean_a: f64,
    /// Mean score for the higher peak-count distribution.
    pub mean_b: f64,
    /// Difference `mean_b - mean_a`.
    pub mean_delta: f64,
    /// Two-sample Kolmogorov-Smirnov statistic.
    pub ks_statistic: f64,
    /// Asymptotic two-sample Kolmogorov-Smirnov p-value approximation.
    pub ks_pvalue_asymptotic: f64,
    /// One-dimensional empirical Wasserstein distance.
    pub wasserstein_1d: f64,
}

#[derive(Debug, Serialize)]
/// One fixed-width histogram bin for a score distribution.
pub struct DistributionHistogramBin {
    /// Dataset label.
    pub dataset: String,
    /// Similarity configuration label.
    pub config: String,
    /// Similarity family label.
    pub metric: &'static str,
    /// Retained peak count for this distribution.
    pub peak_count: usize,
    /// Zero-based histogram bin index.
    pub bin_index: usize,
    /// Inclusive lower score bound for this bin.
    pub bin_lower: f64,
    /// Exclusive upper score bound, except for the final bin.
    pub bin_upper: f64,
    /// Number of scores in the bin.
    pub count: usize,
    /// Fraction of the distribution in the bin.
    pub fraction: f64,
}

#[derive(Debug, Serialize)]
/// Cosine-sum score for one query against one candidate pathway.
pub struct PathwayScore {
    /// Dataset label.
    pub dataset: String,
    /// Similarity configuration label.
    pub config: String,
    /// Retained peak count for this scoring run.
    pub peak_count: usize,
    /// Zero-based query row index in the loaded dataset.
    pub query_index: usize,
    /// Query record identifier.
    pub query_id: String,
    /// Query `NPC` pathway label, if available.
    pub query_npc_pathway: Option<String>,
    /// Candidate pathway represented by the reference spectra.
    pub candidate_npc_pathway: String,
    /// Number of representative spectra for the candidate pathway.
    pub representatives: usize,
    /// Sum of cosine similarities to the candidate pathway representatives.
    pub score: f64,
}

#[derive(Debug, Serialize)]
/// Best pathway prediction produced by cosine-sum representative scoring.
pub struct PathwayPrediction {
    /// Dataset label.
    pub dataset: String,
    /// Similarity configuration label.
    pub config: String,
    /// Retained peak count for this scoring run.
    pub peak_count: usize,
    /// Zero-based query row index in the loaded dataset.
    pub query_index: usize,
    /// Query record identifier.
    pub query_id: String,
    /// Query `NPC` pathway label, if available.
    pub query_npc_pathway: Option<String>,
    /// Predicted pathway label, if any candidate pathway was scored.
    pub predicted_npc_pathway: Option<String>,
    /// Best candidate score.
    pub predicted_score: f64,
    /// Whether the prediction matches the query pathway when both are known.
    pub is_correct: Option<bool>,
    /// Number of candidate pathways scored.
    pub candidate_pathways: usize,
}

#[derive(Debug, Clone)]
/// Score distribution retained for comparing one peak count to the next.
pub struct ScoreDistribution {
    /// Retained peak count.
    pub peak_count: usize,
    /// Raw similarity scores.
    pub scores: Vec<f64>,
    /// Arithmetic mean of the scores.
    pub mean: f64,
}
