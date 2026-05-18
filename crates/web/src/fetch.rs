//! HTTP fetch + npz/json parsing helpers for the browser viewer.
//!
//! All functions return [`Result<T, String>`] so error messages can be
//! displayed directly in the UI without dragging in `anyhow`.

use std::io::Cursor;

use gloo_net::http::Request;
use ndarray::Array3;
use ndarray_npy::NpzReader;
use serde::Deserialize;

/// Top-level dataset listing fetched once on app load.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    /// One entry per dataset under `data/<slug>/`.
    pub datasets: Vec<DatasetEntry>,
}

/// One dataset's locator inside `data/manifest.json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct DatasetEntry {
    /// URL-safe slug, also the subdirectory under `data/`.
    pub slug: String,
    /// Human-readable label.
    pub label: String,
    /// Relative URL of `distribution_grid_configs.json`.
    pub configs_url: String,
    /// Relative URL of `distribution_grid.npz`.
    pub grid_url: String,
    /// Optional relative URL of `pathway_discriminability_lines.json`.
    /// `None` means the dataset has no NPC pathway annotations and the
    /// pathway-classification tab is disabled for it.
    #[serde(default)]
    pub pathways_url: Option<String>,
}

/// Pre-parsed similarity-config descriptor inside
/// `pathway_discriminability_lines.json`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PathwayConfigEntry {
    /// Raw config slug, used as the legend label.
    pub slug: String,
    /// Similarity-metric family (`cosine`, `modified-cosine`, `entropy`,
    /// `modified-entropy`).
    pub family: String,
    /// m/z exponent.
    pub mz_exp: f64,
    /// Intensity exponent.
    pub intensity_exp: f64,
    /// Optional entropy-weighting flag.
    pub weighted: Option<bool>,
}

/// One pathway's AUROC / AUPRC / accuracy / MCC matrices inside
/// `pathway_discriminability_lines.json`. Any metric that the pathway
/// does not define is serialised as JSON `null`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PathwayEntry {
    /// Display label.
    pub label: String,
    /// One of `"aggregate"` (pooled micro-averaged classifier),
    /// `"per_class"` (one-vs-rest for a fixed pathway), or
    /// `"aggregate_weighted"` (support-weighted average across pathways
    /// of the one-vs-rest accuracy and MCC).
    pub kind: String,
    /// `configs.len()` rows by `peak_counts.len()` columns of AUROC.
    /// Missing cells are JSON `null`, the whole matrix is `null` when
    /// AUROC is not defined for this pathway.
    #[serde(default)]
    pub auroc: Option<Vec<Vec<Option<f64>>>>,
    /// `configs.len()` rows by `peak_counts.len()` columns of AUPRC.
    #[serde(default)]
    pub auprc: Option<Vec<Vec<Option<f64>>>>,
    /// `configs.len()` rows by `peak_counts.len()` columns of accuracy.
    #[serde(default)]
    pub accuracy: Option<Vec<Vec<Option<f64>>>>,
    /// `configs.len()` rows by `peak_counts.len()` columns of MCC.
    #[serde(default)]
    pub mcc: Option<Vec<Vec<Option<f64>>>>,
}

/// Whole document at `pathway_discriminability_lines.json`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PathwayLinesData {
    /// Sorted x-axis labels (retained peak counts).
    pub peak_counts: Vec<u64>,
    /// Sorted similarity configs, one per row of every pathway's matrix.
    pub configs: Vec<PathwayConfigEntry>,
    /// One entry per pathway (aggregate plus the seven base NPC pathways).
    pub pathways: Vec<PathwayEntry>,
}

/// One config row inside `distribution_grid_configs.json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ConfigEntry {
    /// Position on the config axis of `distribution_grid.npz`.
    pub config_index: usize,
    /// Stable slug name (`SimilarityConfig::name()`).
    pub config: String,
}

/// Parsed contents of one dataset's `distribution_grid.npz`.
pub struct DistributionGrid {
    /// Δμ grid of shape `(configs, 128, 128)`.
    pub mean_delta: Array3<f64>,
    /// KS statistic grid.
    pub ks_statistic: Array3<f64>,
    /// Asymptotic KS p-value grid.
    pub ks_pvalue_asymptotic: Array3<f64>,
    /// 1D Wasserstein distance grid.
    pub wasserstein_1d: Array3<f64>,
}

/// Fetch and parse `data/manifest.json`.
///
/// # Errors
///
/// Returns a string error if the HTTP request or JSON deserialization fails.
pub async fn load_manifest(base_url: &str) -> Result<Manifest, String> {
    let url = format!("{base_url}manifest.json");
    let response = expect_json_response(&url).await?;
    let manifest: Manifest = response
        .json()
        .await
        .map_err(|error| format!("parsing {url}: {error}"))?;
    Ok(manifest)
}

/// Fetch and parse a dataset's `distribution_grid_configs.json`.
///
/// # Errors
///
/// Returns a string error if the HTTP request or JSON deserialization fails.
pub async fn load_configs(url: &str) -> Result<Vec<ConfigEntry>, String> {
    let response = expect_json_response(url).await?;
    let mut configs: Vec<ConfigEntry> = response
        .json()
        .await
        .map_err(|error| format!("parsing {url}: {error}"))?;
    configs.sort_by_key(|entry| entry.config_index);
    Ok(configs)
}

/// Fetch and parse a dataset's `distribution_grid.npz`.
///
/// # Errors
///
/// Returns a string error if the HTTP request, npz framing, or any
/// individual array deserialization fails.
pub async fn load_grid(url: &str) -> Result<DistributionGrid, String> {
    let response = Request::get(url)
        .send()
        .await
        .map_err(|error| format!("GET {url}: {error}"))?;
    if !response.ok() {
        return Err(format!("GET {url}: HTTP {}", response.status()));
    }
    let bytes = response
        .binary()
        .await
        .map_err(|error| format!("reading {url}: {error}"))?;
    let mut reader =
        NpzReader::new(Cursor::new(bytes)).map_err(|error| format!("npz framing: {error}"))?;
    let read_grid =
        |name: &str, reader: &mut NpzReader<Cursor<Vec<u8>>>| -> Result<Array3<f64>, String> {
            reader
                .by_name(name)
                .map_err(|error| format!("reading {name}: {error}"))
        };
    let mean_delta = read_grid("mean_delta.npy", &mut reader)?;
    let ks_statistic = read_grid("ks_statistic.npy", &mut reader)?;
    let ks_pvalue_asymptotic = read_grid("ks_pvalue_asymptotic.npy", &mut reader)?;
    let wasserstein_1d = read_grid("wasserstein_1d.npy", &mut reader)?;
    Ok(DistributionGrid {
        mean_delta,
        ks_statistic,
        ks_pvalue_asymptotic,
        wasserstein_1d,
    })
}

/// Fetch and parse a dataset's `pathway_discriminability_lines.json`.
///
/// # Errors
///
/// Returns a string error if the HTTP request or JSON deserialization fails.
pub async fn load_pathway_lines(url: &str) -> Result<PathwayLinesData, String> {
    let response = expect_json_response(url).await?;
    let data: PathwayLinesData = response
        .json()
        .await
        .map_err(|error| format!("parsing {url}: {error}"))?;
    Ok(data)
}

/// GET `url` and confirm the server actually returned JSON. dx serve and
/// many static-site hosts answer unknown paths with `200 OK` plus an
/// `index.html` body (an SPA-style fallback), which would otherwise
/// surface as an opaque "expected value at line 1 column 1" further down
/// the JSON parsing path. Returning early with a content-type-aware error
/// here keeps the failure mode legible when the page is loaded from a
/// stale URL that no longer matches the deployment root.
async fn expect_json_response(url: &str) -> Result<gloo_net::http::Response, String> {
    let response = Request::get(url)
        .send()
        .await
        .map_err(|error| format!("GET {url}: {error}"))?;
    if !response.ok() {
        return Err(format!("GET {url}: HTTP {}", response.status()));
    }
    if let Some(content_type) = response.headers().get("content-type") {
        let lowered = content_type.to_ascii_lowercase();
        if !lowered.contains("json") {
            return Err(format!(
                "GET {url}: expected JSON response, server returned content-type \"{content_type}\" \
                 (likely an SPA index.html fallback; reload from the site root)"
            ));
        }
    }
    Ok(response)
}
