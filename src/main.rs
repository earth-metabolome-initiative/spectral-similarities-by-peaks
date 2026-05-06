//! Command-line experiment for measuring how `MS2` similarity distributions
//! change as spectra are truncated to different peak counts.

use anyhow::Result;
use clap::Parser;
use spectral_similarities_by_peaks::{cli::Cli, run};

/// Parse the command line and dispatch the selected subcommand.
fn main() -> Result<()> {
    let cli = Cli::parse();
    run::run(cli)
}
