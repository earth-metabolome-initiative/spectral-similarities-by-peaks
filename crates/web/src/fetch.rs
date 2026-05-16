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
    let response = Request::get(&url)
        .send()
        .await
        .map_err(|error| format!("GET {url}: {error}"))?;
    if !response.ok() {
        return Err(format!("GET {url}: HTTP {}", response.status()));
    }
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
    let response = Request::get(url)
        .send()
        .await
        .map_err(|error| format!("GET {url}: {error}"))?;
    if !response.ok() {
        return Err(format!("GET {url}: HTTP {}", response.status()));
    }
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
