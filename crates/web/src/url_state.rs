//! URL query-string serialisation for shareable viewer state.
//!
//! Every meaningful UI choice (active tab, dataset, heatmap metric / scale
//! / sliders / config, pathway tab metric / pathway / filter sets) is
//! mirrored to the browser's query string via `history.replaceState`, so a
//! copy-pasted URL drops a second visitor on the same view.
//!
//! On boot the [`Viewer`](crate::Viewer) calls [`read`] once to obtain a
//! [`UrlState`] used as the default for every `use_signal` initialiser. A
//! `use_effect` then calls [`write`] on every signal change so the URL
//! always reflects the current state.

use std::collections::HashSet;

use wasm_bindgen::JsValue;
use web_sys::UrlSearchParams;

use crate::config_key::{ExpKey, Family};
use crate::pathway_panel::WeightedChoice;
use spectral_render::PathwayMetric;

/// Parsed `?key=value&…` parameters from the page URL.
#[derive(Clone, Debug, Default)]
pub struct UrlState {
    /// `"heatmaps"` | `"pathways"`.
    pub tab: Option<String>,
    /// Dataset slug (`harmonized-full`, `gems-sampled`, …).
    pub dataset: Option<String>,
    /// Heatmap metric kind, one of the [`crate::MetricKind`] tags.
    pub metric: Option<String>,
    /// `"log"` | `"linear"`.
    pub scale: Option<String>,
    /// Alpha slider raw position (0..=300).
    pub alpha: Option<u32>,
    /// D slider raw position (0..=300).
    pub d: Option<u32>,
    /// Heatmap-tab config slug (e.g. `cosine_mz0.000_int1.000`).
    pub config: Option<String>,
    /// Pathway-tab metric: `"auroc"` | `"auprc"`.
    pub p_metric: Option<String>,
    /// Pathway-tab selected pathway label.
    pub pathway: Option<String>,
    /// Pathway-tab family filter set (lowercase family slugs).
    pub families: Option<Vec<String>>,
    /// Pathway-tab m/z exponent filter set.
    pub mz: Option<Vec<f64>>,
    /// Pathway-tab intensity exponent filter set.
    pub int: Option<Vec<f64>>,
    /// Pathway-tab weighted-flag filter set: `"na"`, `"true"`, `"false"`.
    pub weighted: Option<Vec<String>>,
}

impl UrlState {
    /// Parse a `family` value back into the existing web-side [`Family`] enum.
    #[must_use]
    pub fn parse_family(value: &str) -> Family {
        match value {
            "modified-cosine" => Family::ModifiedCosine,
            "entropy" => Family::Entropy,
            "modified-entropy" => Family::ModifiedEntropy,
            _ => Family::Cosine,
        }
    }

    /// Slug used in the URL for a given family.
    #[must_use]
    pub const fn family_slug(family: Family) -> &'static str {
        match family {
            Family::Cosine => "cosine",
            Family::ModifiedCosine => "modified-cosine",
            Family::Entropy => "entropy",
            Family::ModifiedEntropy => "modified-entropy",
        }
    }

    /// Slug used in the URL for a given pathway weighted choice.
    #[must_use]
    pub const fn weighted_slug(choice: WeightedChoice) -> &'static str {
        match choice {
            WeightedChoice::NotApplicable => "na",
            WeightedChoice::True => "true",
            WeightedChoice::False => "false",
        }
    }

    /// Parse a weighted-flag slug back into a [`WeightedChoice`].
    #[must_use]
    pub fn parse_weighted(value: &str) -> WeightedChoice {
        match value {
            "true" => WeightedChoice::True,
            "false" => WeightedChoice::False,
            _ => WeightedChoice::NotApplicable,
        }
    }

    /// Slug for the metric kind.
    #[must_use]
    pub const fn pathway_metric_slug(metric: PathwayMetric) -> &'static str {
        match metric {
            PathwayMetric::Auroc => "auroc",
            PathwayMetric::Auprc => "auprc",
            PathwayMetric::Accuracy => "accuracy",
            PathwayMetric::Mcc => "mcc",
        }
    }

    /// Parse a pathway-metric slug back into [`PathwayMetric`].
    #[must_use]
    pub fn parse_pathway_metric(value: &str) -> PathwayMetric {
        match value {
            "auprc" => PathwayMetric::Auprc,
            "accuracy" => PathwayMetric::Accuracy,
            "mcc" => PathwayMetric::Mcc,
            _ => PathwayMetric::Auroc,
        }
    }
}

/// Read the current page URL's query string into a [`UrlState`].
/// Returns an empty [`UrlState`] when the browser API is unavailable or
/// the query string is malformed.
#[must_use]
pub fn read() -> UrlState {
    let Some(window) = web_sys::window() else {
        return UrlState::default();
    };
    let Ok(search) = window.location().search() else {
        return UrlState::default();
    };
    let trimmed = search.trim_start_matches('?');
    let Ok(params) = UrlSearchParams::new_with_str(trimmed) else {
        return UrlState::default();
    };
    UrlState {
        tab: params.get("tab"),
        dataset: params.get("dataset"),
        metric: params.get("metric"),
        scale: params.get("scale"),
        alpha: params.get("alpha").and_then(|v| v.parse().ok()),
        d: params.get("d").and_then(|v| v.parse().ok()),
        config: params.get("config"),
        p_metric: params.get("pmetric"),
        pathway: params.get("pathway"),
        families: params.get("families").as_deref().map(parse_set),
        mz: params.get("mz").as_deref().map(parse_float_set),
        int: params.get("int").as_deref().map(parse_float_set),
        weighted: params.get("weighted").as_deref().map(parse_set),
    }
}

/// Serialise the current [`UrlState`] into the page's query string via
/// `history.replaceState`. Silently returns when the browser API is
/// unavailable.
pub fn write(state: &UrlState) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(history) = window.history() else {
        return;
    };
    let Ok(params) = UrlSearchParams::new() else {
        return;
    };
    if let Some(value) = &state.tab {
        params.set("tab", value);
    }
    if let Some(value) = &state.dataset {
        params.set("dataset", value);
    }
    if let Some(value) = &state.metric {
        params.set("metric", value);
    }
    if let Some(value) = &state.scale {
        params.set("scale", value);
    }
    if let Some(value) = state.alpha {
        params.set("alpha", &value.to_string());
    }
    if let Some(value) = state.d {
        params.set("d", &value.to_string());
    }
    if let Some(value) = &state.config {
        params.set("config", value);
    }
    if let Some(value) = &state.p_metric {
        params.set("pmetric", value);
    }
    if let Some(value) = &state.pathway {
        params.set("pathway", value);
    }
    if let Some(values) = &state.families {
        if !values.is_empty() {
            params.set("families", &join_sorted(values.iter().cloned()));
        }
    }
    if let Some(values) = &state.mz {
        if !values.is_empty() {
            params.set("mz", &join_sorted(values.iter().map(|v| format!("{v:.3}"))));
        }
    }
    if let Some(values) = &state.int {
        if !values.is_empty() {
            params.set(
                "int",
                &join_sorted(values.iter().map(|v| format!("{v:.3}"))),
            );
        }
    }
    if let Some(values) = &state.weighted {
        if !values.is_empty() {
            params.set("weighted", &join_sorted(values.iter().cloned()));
        }
    }

    let query = params.to_string().as_string().unwrap_or_default();
    let target = if query.is_empty() {
        window
            .location()
            .pathname()
            .unwrap_or_else(|_| "/".to_string())
    } else {
        format!(
            "{}?{query}",
            window
                .location()
                .pathname()
                .unwrap_or_else(|_| "/".to_string())
        )
    };
    let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(&target));
}

/// Parse a comma-separated string into a list of distinct strings,
/// preserving the order in which each value first appeared.
fn parse_set(value: &str) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut output = Vec::new();
    for token in value.split(',').filter(|s| !s.is_empty()) {
        let owned = token.to_string();
        if seen.insert(owned.clone()) {
            output.push(owned);
        }
    }
    output
}

/// Parse a comma-separated string into a list of distinct `f64`s.
/// Invalid entries are dropped, exact bit-equal duplicates collapse.
#[allow(clippy::cast_possible_truncation)]
fn parse_float_set(value: &str) -> Vec<f64> {
    let mut seen: HashSet<i64> = HashSet::new();
    let mut output = Vec::new();
    for token in value.split(',').filter(|s| !s.is_empty()) {
        let Ok(parsed) = token.parse::<f64>() else {
            continue;
        };
        let key = (parsed * 1000.0).round() as i64;
        if seen.insert(key) {
            output.push(parsed);
        }
    }
    output
}

/// Join an iterator of strings into a comma-separated, lexicographically
/// sorted list so the URL is stable across renders.
fn join_sorted<I>(values: I) -> String
where
    I: IntoIterator<Item = String>,
{
    let mut collected: Vec<String> = values.into_iter().collect();
    collected.sort();
    collected.join(",")
}

/// Translate the existing web-side [`ExpKey`] set into the float list
/// shape used in the URL.
#[must_use]
pub fn exp_keys_as_floats(set: &HashSet<ExpKey>) -> Vec<f64> {
    set.iter().map(|key| key.as_f64()).collect()
}

/// Translate a URL-derived float list back into the canonical
/// [`ExpKey`] set.
#[must_use]
pub fn floats_as_exp_keys(values: &[f64]) -> HashSet<ExpKey> {
    values.iter().copied().map(ExpKey::from_f64).collect()
}

/// Translate the [`Family`] set into URL-friendly slugs.
#[must_use]
pub fn families_as_slugs(set: &HashSet<Family>) -> Vec<String> {
    set.iter()
        .copied()
        .map(|f| UrlState::family_slug(f).to_string())
        .collect()
}

/// Translate URL-derived family slugs back into the [`Family`] set.
#[must_use]
pub fn slugs_as_families(values: &[String]) -> HashSet<Family> {
    values.iter().map(|s| UrlState::parse_family(s)).collect()
}

/// Translate the [`WeightedChoice`] set into URL-friendly slugs.
#[must_use]
pub fn weighted_as_slugs(set: &HashSet<WeightedChoice>) -> Vec<String> {
    set.iter()
        .copied()
        .map(|c| UrlState::weighted_slug(c).to_string())
        .collect()
}

/// Translate URL-derived weighted slugs back into the
/// [`WeightedChoice`] set.
#[must_use]
pub fn slugs_as_weighted(values: &[String]) -> HashSet<WeightedChoice> {
    values.iter().map(|s| UrlState::parse_weighted(s)).collect()
}
