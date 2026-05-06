//! Progress-bar construction for local command-line runs.

use indicatif::{ProgressBar, ProgressStyle};

#[allow(clippy::literal_string_with_formatting_args)]
/// Create a progress bar with the standard scan template.
pub fn progress_bar(len: u64, message: String) -> ProgressBar {
    let progress = ProgressBar::new(len);
    if let Ok(style) =
        ProgressStyle::with_template("{msg} [{elapsed_precise}] {wide_bar} {pos}/{len}")
    {
        progress.set_style(style);
    }
    progress.set_message(message);
    progress
}
