//! Empirical summaries and adjacent-distribution tests for similarity scores.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::cmp::Ordering;

use anyhow::{Result, bail};

use crate::{
    cli::ScanArgs,
    model::{
        DistributionComparison, DistributionHistogramBin, DistributionSummary, ScoreDistribution,
        SimilarityConfig,
    },
};

/// Summarize one sorted empirical similarity-score distribution.
///
/// # Errors
///
/// Returns an error when `sorted` is empty.
pub fn summarize_sorted_distribution(
    args: &ScanArgs,
    config: &SimilarityConfig,
    peak_count: usize,
    sorted: &[f64],
) -> Result<DistributionSummary> {
    if sorted.is_empty() {
        bail!("no scores found for top {peak_count} peaks");
    }
    debug_assert!(is_sorted(sorted));
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / sorted.len() as f64;

    Ok(DistributionSummary {
        dataset: args.dataset.as_str().to_string(),
        config: config.name(),
        metric: config.metric_label(),
        peak_count,
        n_scores: sorted.len(),
        mean,
        stddev: variance.sqrt(),
        min: sorted[0],
        q01: quantile_sorted(sorted, 0.01),
        q05: quantile_sorted(sorted, 0.05),
        q10: quantile_sorted(sorted, 0.10),
        q25: quantile_sorted(sorted, 0.25),
        median: quantile_sorted(sorted, 0.50),
        q75: quantile_sorted(sorted, 0.75),
        q90: quantile_sorted(sorted, 0.90),
        q95: quantile_sorted(sorted, 0.95),
        q99: quantile_sorted(sorted, 0.99),
        max: sorted[sorted.len() - 1],
    })
}

/// Return the canonical self-comparison row for one distribution.
///
/// A distribution compared against itself produces a zero KS statistic, a
/// Wasserstein distance of zero, a mean delta of zero, and a p-value of one.
/// Diagonal cells in the full peak-count comparison grid use this to guarantee
/// the renderer sees the identity values instead of accidentally drifting on
/// floating-point edge cases.
#[must_use]
pub fn self_comparison(
    args: &ScanArgs,
    config: &SimilarityConfig,
    distribution: &ScoreDistribution,
) -> DistributionComparison {
    DistributionComparison {
        dataset: args.dataset.as_str().to_string(),
        config: config.name(),
        metric: config.metric_label(),
        peak_count_a: distribution.peak_count,
        peak_count_b: distribution.peak_count,
        n_scores_a: distribution.scores.len(),
        n_scores_b: distribution.scores.len(),
        mean_a: distribution.mean,
        mean_b: distribution.mean,
        mean_delta: 0.0,
        ks_statistic: 0.0,
        ks_pvalue_asymptotic: 1.0,
        wasserstein_1d: 0.0,
    }
}

/// Compare two adjacent empirical similarity-score distributions.
///
/// # Errors
///
/// Returns an error when either distribution's score vector is empty.
pub fn compare_distributions(
    args: &ScanArgs,
    config: &SimilarityConfig,
    previous: &ScoreDistribution,
    current: &ScoreDistribution,
) -> Result<DistributionComparison> {
    if previous.scores.is_empty() || current.scores.is_empty() {
        bail!("cannot compare empty score distributions");
    }
    debug_assert!(is_sorted(&previous.scores));
    debug_assert!(is_sorted(&current.scores));
    let ks_statistic = ks_two_sample_statistic_sorted(&previous.scores, &current.scores);
    Ok(DistributionComparison {
        dataset: args.dataset.as_str().to_string(),
        config: config.name(),
        metric: config.metric_label(),
        peak_count_a: previous.peak_count,
        peak_count_b: current.peak_count,
        n_scores_a: previous.scores.len(),
        n_scores_b: current.scores.len(),
        mean_a: previous.mean,
        mean_b: current.mean,
        mean_delta: current.mean - previous.mean,
        ks_statistic,
        ks_pvalue_asymptotic: ks_asymptotic_pvalue(
            ks_statistic,
            previous.scores.len(),
            current.scores.len(),
        ),
        wasserstein_1d: wasserstein_1d_sorted(&previous.scores, &current.scores),
    })
}

/// Build fixed-width histogram bins over the `[0, 1]` similarity range from sorted scores.
///
/// # Errors
///
/// Returns an error when `sorted` is empty.
pub fn histogram_sorted_distribution(
    args: &ScanArgs,
    config: &SimilarityConfig,
    peak_count: usize,
    sorted: &[f64],
) -> Result<Vec<DistributionHistogramBin>> {
    if sorted.is_empty() {
        bail!("cannot histogram an empty score distribution");
    }
    debug_assert!(is_sorted(sorted));
    let bin_width = 1.0 / args.histogram_bins as f64;
    let counts = histogram_counts_sorted(sorted, args.histogram_bins, bin_width);

    let n_scores = sorted.len() as f64;
    Ok(counts
        .into_iter()
        .enumerate()
        .map(|(bin_index, count)| {
            let bin_lower = bin_index as f64 * bin_width;
            DistributionHistogramBin {
                dataset: args.dataset.as_str().to_string(),
                config: config.name(),
                metric: config.metric_label(),
                peak_count,
                bin_index,
                bin_lower,
                bin_upper: bin_lower + bin_width,
                count,
                fraction: count as f64 / n_scores,
            }
        })
        .collect())
}

/// Count fixed-width histogram bins from sorted finite scores.
fn histogram_counts_sorted(sorted: &[f64], bins: usize, bin_width: f64) -> Vec<usize> {
    debug_assert!(!sorted.is_empty());
    debug_assert!(bins > 0);
    debug_assert!(is_sorted(sorted));

    let mut counts = vec![0_usize; bins];
    let mut cursor = 0;
    for (bin_index, count) in counts.iter_mut().enumerate().take(bins - 1) {
        let upper = (bin_index + 1) as f64 * bin_width;
        let start = cursor;
        while cursor < sorted.len() && sorted[cursor] < upper {
            cursor += 1;
        }
        *count = cursor - start;
    }
    counts[bins - 1] = sorted.len() - cursor;
    counts
}

/// Return a linearly interpolated quantile from an ascending sorted sample.
fn quantile_sorted(sorted: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = position - lower as f64;
    sorted[lower].mul_add(1.0 - weight, sorted[upper] * weight)
}

/// Return whether a sample is sorted by the same ordering used for scores.
fn is_sorted(values: &[f64]) -> bool {
    values
        .windows(2)
        .all(|window| window[0].total_cmp(&window[1]) != Ordering::Greater)
}

/// Compute the two-sample Kolmogorov-Smirnov statistic from sorted samples.
fn ks_two_sample_statistic_sorted(left: &[f64], right: &[f64]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return f64::NAN;
    }
    debug_assert!(is_sorted(left));
    debug_assert!(is_sorted(right));

    let mut i = 0;
    let mut j = 0;
    let mut max_delta = 0.0_f64;
    while i < left.len() || j < right.len() {
        let next = match (left.get(i), right.get(j)) {
            (Some(&a), Some(&b)) => match a.total_cmp(&b) {
                Ordering::Less | Ordering::Equal => a,
                Ordering::Greater => b,
            },
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => break,
        };

        while i < left.len() && left[i] <= next {
            i += 1;
        }
        while j < right.len() && right[j] <= next {
            j += 1;
        }

        let cdf_left = i as f64 / left.len() as f64;
        let cdf_right = j as f64 / right.len() as f64;
        max_delta = max_delta.max((cdf_left - cdf_right).abs());
    }
    max_delta
}

/// Approximate the asymptotic p-value for a two-sample `KS` statistic.
fn ks_asymptotic_pvalue(statistic: f64, n_left: usize, n_right: usize) -> f64 {
    if !statistic.is_finite() || n_left == 0 || n_right == 0 {
        return f64::NAN;
    }
    // Identical empirical distributions: statistic is exactly 0. The
    // alternating-series approximation oscillates between +1 and -1 at
    // lambda=0 and lands at sum=0 after the 100-term cap, which would
    // misreport p=0 instead of the analytic limit p=1.
    if statistic == 0.0 {
        return 1.0;
    }
    let effective_n = (n_left as f64 * n_right as f64) / (n_left + n_right) as f64;
    let sqrt_n = effective_n.sqrt();
    let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * statistic;
    let mut sum = 0.0_f64;
    for term in 1..=100 {
        let sign = if term % 2 == 1 { 1.0 } else { -1.0 };
        let value = (-2.0 * f64::from(term).powi(2) * lambda.powi(2)).exp();
        sum += sign * value;
        if value < 1e-12 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Compute the one-dimensional empirical Wasserstein distance from sorted samples.
fn wasserstein_1d_sorted(left: &[f64], right: &[f64]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return f64::NAN;
    }
    debug_assert!(is_sorted(left));
    debug_assert!(is_sorted(right));

    let mut i = 0;
    let mut j = 0;
    let mut cdf_left = 0.0_f64;
    let mut cdf_right = 0.0_f64;
    let mut previous = left[0].min(right[0]);
    let mut area = 0.0_f64;

    while i < left.len() || j < right.len() {
        let next = match (left.get(i), right.get(j)) {
            (Some(&a), Some(&b)) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => break,
        };
        area = (next - previous)
            .abs()
            .mul_add((cdf_left - cdf_right).abs(), area);

        while i < left.len() && left[i] <= next {
            i += 1;
        }
        while j < right.len() && right[j] <= next {
            j += 1;
        }
        cdf_left = i as f64 / left.len() as f64;
        cdf_right = j as f64 / right.len() as f64;
        previous = next;
    }

    area
}

#[cfg(test)]
/// Unit tests for distribution statistics.
mod tests {
    use super::{histogram_counts_sorted, ks_two_sample_statistic_sorted, wasserstein_1d_sorted};

    #[test]
    /// Sorted histogram counts preserve fixed-width bin and clamping semantics.
    fn sorted_histogram_counts_match_fixed_width_bins() {
        let scores = [-0.5, 0.0, 0.249, 0.25, 0.5, 0.75, 1.0, 1.5];
        assert_eq!(histogram_counts_sorted(&scores, 4, 0.25), vec![3, 1, 1, 3]);
    }

    #[test]
    /// Identical samples have a zero `KS` statistic.
    fn ks_statistic_is_zero_for_identical_samples() {
        let sample = [0.1, 0.2, 0.3, 0.4];
        assert!(ks_two_sample_statistic_sorted(&sample, &sample).abs() < f64::EPSILON);
    }

    #[test]
    /// Fully separated samples have a maximal `KS` statistic.
    fn ks_statistic_detects_separated_samples() {
        let left = [0.0, 0.0, 0.0];
        let right = [1.0, 1.0, 1.0];
        assert!((ks_two_sample_statistic_sorted(&left, &right) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    /// Equal-size samples shifted by one unit have distance one.
    fn wasserstein_matches_shift_for_equal_samples() {
        let left = [0.0, 1.0, 2.0];
        let right = [1.0, 2.0, 3.0];
        assert!((wasserstein_1d_sorted(&left, &right) - 1.0).abs() < 1e-12);
    }

    #[test]
    /// Identical empirical distributions have an asymptotic p-value of 1.
    fn ks_asymptotic_pvalue_is_one_when_statistic_is_zero() {
        let pvalue = super::ks_asymptotic_pvalue(0.0, 1024, 1024);
        assert!(
            (pvalue - 1.0).abs() < f64::EPSILON,
            "expected p=1 for identical distributions, got {pvalue}"
        );
    }
}
