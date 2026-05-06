//! Library entry points for measuring how `MS2` similarity distributions
//! change as spectra are truncated to different peak counts.

/// Command-line argument parsing and value conversion.
pub mod cli;
/// Dataset loading and conversion into local records.
mod data;
/// Empirical distribution summaries and adjacent cutoff comparisons.
mod distribution;
/// Shared data structures used across the experiment pipeline.
mod model;
/// Top-neighbor similarity search over prepared spectra.
mod neighbors;
/// Binary and columnar artifact writers.
mod output;
/// Fixed-representative `NPC` pathway scoring.
mod pathway;
/// Progress-bar helpers for long-running local scans.
mod progress;
/// Top-level command orchestration.
pub mod run;
/// Spectrum sampling and preprocessing utilities.
mod spectra;
