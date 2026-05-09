//! Progress reporting for local command-line runs.

#![allow(clippy::literal_string_with_formatting_args)]

use std::io::{self, IsTerminal};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Progress reporter shared by all phases of one scan.
pub struct ScanProgress {
    /// Interactive multibar renderer, disabled for non-terminal output.
    multi: Option<MultiProgress>,
}

impl ScanProgress {
    /// Create a reporter that uses live multibars only on an interactive terminal.
    pub fn new() -> Self {
        Self {
            multi: io::stderr().is_terminal().then(MultiProgress::new),
        }
    }

    /// Create a bounded progress bar.
    pub fn bar(&self, len: u64, message: impl Into<String>) -> ProgressTask {
        let message = message.into();
        if let Some(multi) = &self.multi {
            let progress = multi.add(ProgressBar::new(len));
            if let Ok(style) = ProgressStyle::with_template(
                "{msg} [{elapsed_precise} elapsed, {eta_precise} eta] \
                 {wide_bar} {pos}/{len} ({per_sec})",
            ) {
                progress.set_style(style);
            }
            progress.set_message(message);
            return ProgressTask {
                progress: Some(progress),
                plain_message: None,
            };
        }

        eprintln!("starting {message}");
        ProgressTask {
            progress: None,
            plain_message: Some(message),
        }
    }

    /// Create an indeterminate progress spinner for phases without row callbacks.
    pub fn spinner(&self, message: impl Into<String>) -> ProgressTask {
        let message = message.into();
        if let Some(multi) = &self.multi {
            let progress = multi.add(ProgressBar::new_spinner());
            if let Ok(style) = ProgressStyle::with_template("{msg} [{elapsed_precise}] {spinner}") {
                progress.set_style(style);
            }
            progress.set_message(message);
            progress.enable_steady_tick(std::time::Duration::from_millis(100));
            return ProgressTask {
                progress: Some(progress),
                plain_message: None,
            };
        }

        eprintln!("starting {message}");
        ProgressTask {
            progress: None,
            plain_message: Some(message),
        }
    }
}

impl Default for ScanProgress {
    /// Create the default terminal-aware progress reporter.
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for one active progress line.
pub struct ProgressTask {
    /// Live `indicatif` progress bar when running on a terminal.
    progress: Option<ProgressBar>,
    /// Message printed at start in non-terminal mode.
    plain_message: Option<String>,
}

impl ProgressTask {
    /// Increment a bounded progress bar.
    pub fn inc(&self, delta: u64) {
        if let Some(progress) = &self.progress {
            progress.inc(delta);
        }
    }

    /// Update the visible progress message.
    pub fn set_message(&self, message: impl Into<String>) {
        if let Some(progress) = &self.progress {
            progress.set_message(message.into());
        }
    }

    /// Mark the progress line as complete and clear it from interactive output.
    pub fn finish(&self) {
        if let Some(progress) = &self.progress {
            progress.finish_and_clear();
        } else if let Some(message) = &self.plain_message {
            eprintln!("finished {message}");
        }
    }
}

#[cfg(test)]
/// Unit tests for progress handle behavior.
mod tests {
    use super::ScanProgress;

    #[test]
    /// Progress tasks support common lifecycle calls.
    fn progress_task_lifecycle_is_noop_safe() {
        let progress = ScanProgress::new();
        let task = progress.bar(2, "testing bounded progress");
        task.set_message("testing bounded progress update");
        task.inc(1);
        task.inc(1);
        task.finish();

        let spinner = progress.spinner("testing spinner progress");
        spinner.finish();
    }
}
