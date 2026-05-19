//! Browser viewer entry point.
//!
//! Boots a Dioxus 0.7 single-page app that fetches per-dataset
//! `distribution_grid.npz` files from `data/` and renders heatmaps on
//! demand via the `spectral-render` crate compiled to WebAssembly.

mod config_key;
mod fetch;
mod pathway_panel;
mod responsive_svg;
mod svg_export;
mod url_state;

use std::collections::HashSet;
use std::rc::Rc;

use config_key::{ConfigCatalog, ConfigKey, ExpKey, Family};
use dioxus::prelude::*;
use dioxus_free_icons::Icon;
use dioxus_free_icons::icons::fa_brands_icons::FaGithub;
use dioxus_free_icons::icons::fa_solid_icons::{
    FaChartArea, FaCircleInfo, FaDatabase, FaDownload, FaHandshake, FaSliders, FaWaveSquare,
};
use spectral_render::{GridViews, Metric, PathwayMetric, Scales, render_cell_svg};

use crate::fetch::{ConfigEntry, DatasetEntry, DistributionGrid, Manifest, PathwayLinesData};
use crate::pathway_panel::{PathwayPanel, WeightedChoice, default_filter_state};

const DATA_BASE_URL: &str = "data/";

/// `dataset_resource`'s payload type. The leading `usize` is the
/// `dataset_index` the fetch was kicked off for, so consumers can ignore
/// the value while the user is mid-switch (i.e. when the resource still
/// holds the previous dataset's response).
type DatasetResource = Resource<Result<(usize, Vec<ConfigEntry>, DistributionGrid), String>>;

/// `pathway_resource`'s payload type. Same `usize`-tagged shape so the
/// pathway-classification panel can also gate its render on freshness.
type PathwayResource = Resource<Result<(usize, PathwayLinesData), String>>;

/// Axis of the heatmap value. Combined with [`ColorScale`] to produce one of
/// the eight rendered metrics in [`spectral_render::Metric`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum MetricKind {
    KsPvalueAsymptotic,
    KsStatistic,
    MeanDelta,
    Wasserstein1d,
}

impl MetricKind {
    const ALL: [(Self, &'static str); 4] = [
        (Self::KsPvalueAsymptotic, "KS p-value"),
        (Self::KsStatistic, "KS statistic"),
        (Self::MeanDelta, "Δ mean"),
        (Self::Wasserstein1d, "Wasserstein"),
    ];

    /// URL-friendly slug used by [`url_state`] to round-trip the choice.
    const fn slug(self) -> &'static str {
        match self {
            Self::KsPvalueAsymptotic => "ks_pvalue",
            Self::KsStatistic => "ks_stat",
            Self::MeanDelta => "mean_delta",
            Self::Wasserstein1d => "wasserstein",
        }
    }

    /// Inverse of [`Self::slug`]. Unknown slugs fall back to KS p-value.
    fn from_slug(value: &str) -> Self {
        match value {
            "ks_stat" => Self::KsStatistic,
            "mean_delta" => Self::MeanDelta,
            "wasserstein" => Self::Wasserstein1d,
            _ => Self::KsPvalueAsymptotic,
        }
    }
}

/// Linear vs logarithmic color mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum ColorScale {
    Log,
    Linear,
}

impl ColorScale {
    const ALL: [(Self, &'static str); 2] = [(Self::Log, "Log"), (Self::Linear, "Linear")];

    /// URL-friendly slug used by [`url_state`].
    const fn slug(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Linear => "linear",
        }
    }

    /// Inverse of [`Self::slug`]. Unknown slugs fall back to log scale.
    fn from_slug(value: &str) -> Self {
        match value {
            "linear" => Self::Linear,
            _ => Self::Log,
        }
    }
}

/// Top-level view inside the [`Viewer`]. Heatmaps tab shows the existing
/// p-value / D-statistic heatmap explorer. Pathways tab shows AUROC /
/// AUPRC line plots from `pathway_discriminability_lines.json`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum ViewerTab {
    Heatmaps,
    Pathways,
}

impl ViewerTab {
    const ALL: [(Self, &'static str); 2] =
        [(Self::Heatmaps, "Heatmaps"), (Self::Pathways, "Pathways")];

    /// URL-friendly slug.
    const fn slug(self) -> &'static str {
        match self {
            Self::Heatmaps => "heatmaps",
            Self::Pathways => "pathways",
        }
    }

    /// Inverse of [`Self::slug`].
    fn from_slug(value: &str) -> Self {
        match value {
            "pathways" => Self::Pathways,
            _ => Self::Heatmaps,
        }
    }
}

/// One-sentence plain-language tooltip for a dataset pill.
fn dataset_description(slug: &str) -> String {
    match slug {
        "harmonized-full" => {
            "Harmonized MS2 reference dataset: 443,905 annotated spectra, full top-128 peaks per spectrum.".to_string()
        }
        "gems-sampled" => {
            "GeMS-A10 sample: 100,000 query spectra searched against 1,000,000 reference spectra, full top-128 peaks per spectrum.".to_string()
        }
        other => format!("Dataset {other}."),
    }
}

/// Plain-language tooltip for a metric-kind pill.
const fn metric_kind_description(kind: MetricKind) -> &'static str {
    match kind {
        MetricKind::KsPvalueAsymptotic => {
            "Kolmogorov-Smirnov p-value: probability that the two score distributions are drawn from the same population. Small p-values mean the distributions are reliably different."
        }
        MetricKind::KsStatistic => {
            "Kolmogorov-Smirnov D statistic: the largest gap (0 to 1) between the cumulative distribution functions of the two score samples. Bigger D means more difference."
        }
        MetricKind::MeanDelta => {
            "Difference of the mean similarity score between the two peak-count cells. Positive means the row's average score is higher than the column's."
        }
        MetricKind::Wasserstein1d => {
            "Earth-mover (1-D Wasserstein) distance between the two score distributions: the average amount you'd have to shift score mass to morph one into the other."
        }
    }
}

/// Plain-language tooltip for the color-scale pill.
const fn scale_description(scale: ColorScale) -> &'static str {
    match scale {
        ColorScale::Log => {
            "Logarithmic color mapping. Emphasises differences over many orders of magnitude (e.g., p-values ranging from 1 down to 10^-200)."
        }
        ColorScale::Linear => {
            "Linear color mapping. Equal value differences map to equal color differences."
        }
    }
}

/// Plain-language tooltip for a similarity-family pill.
const fn family_description(family: Family) -> &'static str {
    match family {
        Family::Cosine => {
            "Standard cosine similarity: the angle between the two spectra treated as intensity vectors."
        }
        Family::ModifiedCosine => {
            "Modified cosine: like cosine, but also matches peaks that are offset by the precursor mass difference between the two spectra."
        }
        Family::Entropy => {
            "Spectral entropy similarity: an information-theory metric that penalises noisy, low-entropy spectra."
        }
        Family::ModifiedEntropy => {
            "Modified spectral entropy: same as entropy but with the shift-aware peak matching used by modified cosine."
        }
    }
}

/// Plain-language tooltip for an m/z exponent pill.
fn mz_exp_description(value: f64) -> String {
    if value <= 0.0 {
        "Do not weight peaks by their m/z value (only intensity contributes to the score)."
            .to_string()
    } else if (value - 1.0).abs() < 1.0e-9 {
        "Weight each peak's contribution linearly by its m/z value (heavier fragments count more)."
            .to_string()
    } else if (value - 3.0).abs() < 1.0e-9 {
        "NIST-style weighting: each peak's contribution is multiplied by its m/z cubed.".to_string()
    } else {
        format!("Multiply each peak's contribution by its m/z raised to the {value} power.")
    }
}

/// Plain-language tooltip for an intensity-exponent pill.
fn int_exp_description(value: f64) -> String {
    if (value - 1.0).abs() < 1.0e-9 {
        "Use the raw peak intensities (no rescaling).".to_string()
    } else if value < 1.0 {
        format!(
            "Raise each peak intensity to the {value} power. Values below 1 compress the dynamic range so faint peaks count more relative to the brightest ones."
        )
    } else {
        format!(
            "Raise each peak intensity to the {value} power. Values above 1 amplify the brightest peaks."
        )
    }
}

/// Plain-language tooltip for the weighting pill (entropy variants).
const fn weighted_description(weighted: bool) -> &'static str {
    if weighted {
        "Apply Stein-style entropy weighting that emphasises peaks with high information content."
    } else {
        "Plain spectral entropy without information-content weighting; all peaks contribute equally to the entropy term."
    }
}

/// Long-form noun phrase for the chosen metric, suitable for embedding in
/// figure captions.
const fn metric_caption_phrase(kind: MetricKind) -> &'static str {
    match kind {
        MetricKind::KsPvalueAsymptotic => "asymptotic Kolmogorov–Smirnov p-value",
        MetricKind::KsStatistic => "Kolmogorov–Smirnov D statistic",
        MetricKind::MeanDelta => {
            "signed difference of mean similarity scores (column mean minus row mean)"
        }
        MetricKind::Wasserstein1d => "1-D Wasserstein (earth-mover) distance",
    }
}

/// Sentence describing how the color ramp maps to metric values for the
/// chosen kind. Encodes the palette convention (viridis vs red-blue).
const fn color_meaning(kind: MetricKind) -> &'static str {
    match kind {
        MetricKind::KsPvalueAsymptotic => {
            "yellow cells correspond to p-values near 1 (the two score distributions are statistically indistinguishable), \
             while purple cells correspond to p-values approaching 0 (the distributions are reliably distinct)"
        }
        MetricKind::KsStatistic => {
            "yellow cells correspond to large D values (the two empirical CDFs are far apart), \
             while purple cells correspond to D ≈ 0 (the CDFs nearly coincide)"
        }
        MetricKind::MeanDelta => {
            "blue cells indicate that the column's mean score is higher than the row's, \
             red cells indicate the opposite, and near-white cells indicate equal means"
        }
        MetricKind::Wasserstein1d => {
            "yellow cells correspond to a large transport cost (the distributions are far apart in score-mass), \
             while purple cells correspond to a near-zero cost (the distributions are very similar)"
        }
    }
}

/// Sentence fragment naming the similarity family + weighting for embedding
/// inside the caption.
fn config_caption_phrase(key: ConfigKey) -> String {
    let weighted_tag = match key.weighted {
        Some(true) => " (weighted)",
        Some(false) => " (unweighted)",
        None => "",
    };
    let mut weight_parts: Vec<String> = Vec::new();
    let mz = key.mz_exp.as_f64();
    if mz > 0.0 {
        weight_parts.push(if (mz - 1.0).abs() < 1.0e-9 {
            "m/z".to_string()
        } else {
            format!("m/z^{}", key.mz_exp.label())
        });
    }
    let int_v = key.int_exp.as_f64();
    if int_v > 0.0 {
        weight_parts.push(if (int_v - 1.0).abs() < 1.0e-9 {
            "intensity".to_string()
        } else {
            format!("intensity^{}", key.int_exp.label())
        });
    }
    let family = key.family.label().to_lowercase();
    if weight_parts.is_empty() {
        format!("the {family}{weighted_tag} similarity")
    } else {
        format!(
            "the {family}{weighted_tag} similarity with per-peak weighting w ∝ {}",
            weight_parts.join(" · ")
        )
    }
}

/// Convert the plain caption into styled HTML:
///
/// 1. The first sentence (everything up to the first period) is wrapped in
///    `<strong>` so it acts as a bold title-like opening.
/// 2. Every color word that appears later in the prose is wrapped in a bold
///    span using a matching hex color.
///
/// Returns an HTML fragment safe to inject via Dioxus' `dangerous_inner_html`
/// because the caption template only ever contains values produced by this
/// crate (no user-supplied text).
#[must_use]
pub fn caption_to_html(caption: &str) -> String {
    let (title, rest) =
        first_sentence_end(caption).map_or((caption, ""), |split_at| caption.split_at(split_at));

    // Order matters: longer / compound tokens must be replaced before the
    // shorter ones they contain (e.g. "near-white" before "white"). The
    // replacement honours word boundaries so substrings like "red" inside
    // "predicted" do not get coloured.
    let color_words: &[(&str, &str)] = &[
        ("near-white", "#6b6b6b"),
        ("yellow", "#b08400"),
        ("purple", "#5d2786"),
        ("blue", "#205e8c"),
        ("red", "#9d4133"),
        ("coral", "#c84766"),
        ("cyan", "#2b85ac"),
    ];
    let mut rest_html = rest.to_string();
    for (word, color) in color_words {
        let replacement = format!("<strong style=\"color: {color};\">{word}</strong>");
        rest_html = replace_word_boundary(&rest_html, word, &replacement);
    }
    format!("<strong>{title}</strong>{rest_html}")
}

/// Replace every word-boundary-bounded occurrence of `needle` in `haystack`
/// with `replacement`. An occurrence is considered "bounded" when neither
/// the character immediately before nor the character immediately after is
/// alphanumeric, `_`, or `-`. This is what stops a literal color word like
/// "red" from being recoloured inside "p**red**icted".
fn replace_word_boundary(haystack: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return haystack.to_string();
    }
    let is_word_char = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    let mut out = String::with_capacity(haystack.len());
    let mut last_end = 0;
    let mut search_start = 0;
    while let Some(rel_idx) = haystack[search_start..].find(needle) {
        let idx = search_start + rel_idx;
        let prev_is_word = haystack[..idx]
            .chars()
            .next_back()
            .is_some_and(is_word_char);
        let next_is_word = haystack[idx + needle.len()..]
            .chars()
            .next()
            .is_some_and(is_word_char);
        out.push_str(&haystack[last_end..idx]);
        if prev_is_word || next_is_word {
            out.push_str(needle);
        } else {
            out.push_str(replacement);
        }
        last_end = idx + needle.len();
        search_start = last_end;
    }
    out.push_str(&haystack[last_end..]);
    out
}

/// Find the byte offset immediately after the first sentence-ending
/// period in `text`, or `None` if there is no such period. A period
/// counts as sentence-ending only when it is followed by whitespace or
/// end-of-string, so decimals like `intensity^0.6` are not treated as a
/// sentence boundary and the caption's title segment stays bold all
/// the way to the real first `. `.
fn first_sentence_end(text: &str) -> Option<usize> {
    let mut search_start = 0;
    while let Some(rel) = text[search_start..].find('.') {
        let idx = search_start + rel;
        let after_dot = idx + '.'.len_utf8();
        let bounded = text[after_dot..]
            .chars()
            .next()
            .is_none_or(char::is_whitespace);
        if bounded {
            return Some(after_dot);
        }
        search_start = after_dot;
    }
    None
}

/// Format an α / D threshold value compactly for inline mention.
fn format_threshold(value: f64) -> String {
    if value <= 0.0 {
        "0".to_string()
    } else if value < 0.01 {
        format!("{value:.2e}")
    } else if (value - 1.0).abs() < 1.0e-9 {
        "1".to_string()
    } else {
        format!("{value:.3}")
    }
}

/// Compose the academic-style figure caption shown under each heatmap.
fn figure_caption(
    dataset_label: &str,
    config_key: ConfigKey,
    metric_kind: MetricKind,
    scale: ColorScale,
    alpha: f64,
    d_threshold: f64,
) -> String {
    let scale_word = match scale {
        ColorScale::Log => "logarithmic",
        ColorScale::Linear => "linear",
    };
    format!(
        "Heatmap of the {metric} computed between the MS2 spectral-similarity \
         score distributions of the {dataset} dataset under {pretty_config}. \
         Each cell (r, c) reports the metric value between the score \
         distribution obtained when retaining the top r intensity peaks per \
         spectrum on the row side of the comparison and the top c peaks on \
         the column side, with r and c spanning 1–128. Color encodes the \
         metric on a {scale} scale with the actual range shown on the \
         right-hand colorbar: {color}. The dashed coral curve traces the \
         iso-contour at α = {alpha} (the asymptotic KS p-value level set); \
         the dashed cyan curve traces the iso-contour at D = {d} (the KS \
         statistic level set, i.e. the maximum gap between the two empirical \
         cumulative distribution functions). The main diagonal is masked \
         because every (r, r) cell would compare a distribution against \
         itself.",
        metric = metric_caption_phrase(metric_kind),
        dataset = dataset_label,
        pretty_config = config_caption_phrase(config_key),
        scale = scale_word,
        color = color_meaning(metric_kind),
        alpha = format_threshold(alpha),
        d = format_threshold(d_threshold),
    )
}

const fn combine_metric(kind: MetricKind, scale: ColorScale) -> Metric {
    match (kind, scale) {
        (MetricKind::KsPvalueAsymptotic, ColorScale::Log) => Metric::KsPvalueAsymptoticLog,
        (MetricKind::KsPvalueAsymptotic, ColorScale::Linear) => Metric::KsPvalueAsymptoticLinear,
        (MetricKind::KsStatistic, ColorScale::Log) => Metric::KsStatisticLog,
        (MetricKind::KsStatistic, ColorScale::Linear) => Metric::KsStatisticLinear,
        (MetricKind::MeanDelta, ColorScale::Log) => Metric::MeanDeltaLog,
        (MetricKind::MeanDelta, ColorScale::Linear) => Metric::MeanDeltaLinear,
        (MetricKind::Wasserstein1d, ColorScale::Log) => Metric::Wasserstein1dLog,
        (MetricKind::Wasserstein1d, ColorScale::Linear) => Metric::Wasserstein1dLinear,
    }
}

/// Main stylesheet, baked into the index.html `<head>` at build time so the
/// browser fetches it in parallel with the WASM blob and the first paint is
/// already styled. The `with_static_head(true)` option asks the dx CLI to
/// emit a `<link rel="stylesheet">` directly into the page template instead
/// of injecting it from the runtime DOM effect that fires after WASM boots
/// (the runtime path is what caused the unstyled flash). The Google Fonts
/// `<link>` lives in `crates/web/index.html` for the same reason.
#[allow(clippy::volatile_composites)]
static MAIN_CSS: Asset = asset!(
    "/assets/style.css",
    AssetOptions::css().with_static_head(true)
);

fn main() {
    console_error_panic_hook::set_once();
    dioxus::launch(App);
}

#[component]
#[allow(non_snake_case)]
fn App() -> Element {
    let manifest = use_resource(|| async move { fetch::load_manifest(DATA_BASE_URL).await });

    rsx! {
        // Touching MAIN_CSS keeps the asset reachable from the rsx tree so
        // the dx CLI walks it during the asset-collection pass. The static
        // head injection is what actually puts the <link> on the page,
        // before WASM ever loads, so this evaluates to no observable DOM.
        document::Stylesheet { href: MAIN_CSS }

        main { class: "page",
            Hero {}
            match &*manifest.read_unchecked() {
                Some(Ok(Manifest { datasets })) if !datasets.is_empty() => rsx! {
                    Viewer { datasets: datasets.clone() }
                },
                Some(Ok(_)) => rsx! {
                    p { class: "error",
                        "data/manifest.json contains no datasets."
                    }
                },
                Some(Err(error)) => rsx! {
                    p { class: "error",
                        "Failed to load data/manifest.json: {error}"
                    }
                },
                None => rsx! { p { class: "loading", "Loading manifest…" } },
            }
        }
    }
}

#[component]
#[allow(non_snake_case)]
fn Hero() -> Element {
    rsx! {
        header { class: "hero",
            p { class: "eyebrow", "Earth Metabolome Initiative" }
            div { class: "hero-titlebar",
                h1 { class: "hero-title",
                    "Spectral similarities by "
                    span { class: "hero-accent", "peaks" }
                }
                div { class: "pill-row hero-actions",
                    a {
                        class: "pill",
                        href: "https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        title: "Open the spectral-similarities-by-peaks repository on GitHub.",
                        aria_label: "Open the spectral-similarities-by-peaks repository on GitHub.",
                        span { aria_hidden: "true",
                            Icon { width: 14, height: 14, icon: FaGithub }
                        }
                        "Source code"
                    }
                    a {
                        class: "pill pill-collab",
                        href: "https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks/issues/new?template=collab.yml",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        title: "Open a pre-filled GitHub issue describing what you want to collaborate on.",
                        aria_label: "Open a pre-filled GitHub issue to describe a collaboration interest.",
                        span { aria_hidden: "true",
                            Icon { width: 14, height: 14, icon: FaHandshake }
                        }
                        "I want to collab!"
                    }
                }
            }
            p { class: "abstract",
                "This experiment quantifies, across ~80 billion (query, candidate) similarity scores drawn from the "
                a {
                    href: "https://zenodo.org/records/20042904",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "harmonized annotated dataset"
                }
                " and a sample of the "
                a {
                    href: "https://zenodo.org/records/20040772",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "GeMS-A10 corpus"
                }
                ", how the empirical score distribution of a spectral similarity changes when each spectrum is truncated to its top-"
                em { "k" }
                " intensity peaks for "
                em { "k" }
                " from 1 to 128. Eighteen configurations across "
                a {
                    href: "https://doi.org/10.1016/1044-0305(94)87009-8",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Cosine"
                }
                ", "
                a {
                    href: "https://doi.org/10.1038/nbt.3597",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Modified cosine"
                }
                ", "
                a {
                    href: "https://doi.org/10.1038/s41592-021-01331-z",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Entropy"
                }
                ", and Modified entropy at several m/z and intensity exponents are compared cell-by-cell on the 128 by 128 grid by KS D statistic, asymptotic KS p-value, mean difference, and 1-D Wasserstein distance. The Rust pipeline uses "
                a {
                    href: "https://doi.org/10.1038/s41592-023-02012-9",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Flash entropy search"
                }
                "-style indices and still ran ~70k compute hours on Lawrencium. Distributions stabilise quickly: D ≤ 0.05 at 4 to 47 retained peaks, D ≤ 0.01 at 7 to 103, with intensity exponent dominating the per-config diversity ranking. With 6 to 28 million pairs per cell the asymptotic p-values fall inside "
                a {
                    href: "https://en.wikipedia.org/wiki/Lindley%27s_paradox",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Lindley's-paradox"
                }
                " territory, so the figures rely on the sample-size-invariant D statistic. A second line of inquiry builds a reference panel of up to 35 spectra (5 from each of the 7 base "
                a {
                    href: "https://doi.org/10.1186/s13321-022-00624-5",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "NPClassifier"
                }
                " pathways), scores every query against it as a per-pathway similarity-sum, and reports AUROC, AUPRC, accuracy, and MCC. The similarity-sum carries no usable predictive signal: per-pathway one-vs-rest MCC stays within [-0.17, 0.13] and the support-weighted aggregate caps at 0.024, a clean illustration of why "
                a {
                    href: "https://doi.org/10.1186/s13040-023-00322-4",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "MCC tracks performance more honestly than accuracy or AUROC under class imbalance"
                }
                ". With a single deterministic representative draw none of these numbers carry confidence intervals. Future work in the README covers exponent ranges beyond the current grid, repeated sampling of random class representatives, and a mechanistic write-up of why certain similarity-and-peak-count combinations seem to align with certain pathways. The Pathways tab below lets the reader pick a pathway and metric and toggle similarity-family, m/z, intensity, and entropy-weighting filters."
            }
        }
    }
}

#[component]
#[allow(non_snake_case)]
fn Viewer(datasets: Vec<DatasetEntry>) -> Element {
    // Restore URL-encoded state once on mount. Every `use_signal` below
    // initialises from this snapshot, and a `use_effect` at the end of
    // this function mirrors every signal change back into the URL.
    let initial = url_state::read();

    let initial_dataset_index = initial
        .dataset
        .as_deref()
        .and_then(|slug| datasets.iter().position(|d| d.slug == slug))
        .unwrap_or(0);
    let dataset_index = use_signal(|| initial_dataset_index);
    let active_tab = use_signal(|| {
        initial
            .tab
            .as_deref()
            .map_or(ViewerTab::Heatmaps, ViewerTab::from_slug)
    });
    let metric_kind = use_signal(|| {
        initial
            .metric
            .as_deref()
            .map_or(MetricKind::KsPvalueAsymptotic, MetricKind::from_slug)
    });
    let color_scale = use_signal(|| {
        initial
            .scale
            .as_deref()
            .map_or(ColorScale::Log, ColorScale::from_slug)
    });
    // Slider positions on a 0..=300 axis mapped through `log_slider_value`
    // to a value in `[10^-3, 10^0]`. 170 ≈ 0.05, the literature default.
    let alpha_milli = use_signal(|| initial.alpha.unwrap_or(170));
    let d_centi = use_signal(|| initial.d.unwrap_or(170));
    // Holds the user-selected ConfigKey. Re-derived when the dataset changes
    // (or restored from the `config=` URL parameter via the catalog lookup).
    let initial_config_slug = initial.config.clone();
    let active_key: Signal<Option<ConfigKey>> =
        use_signal(|| initial_config_slug.as_deref().and_then(ConfigKey::parse));

    // Pathway-tab signals. Filter sets stay empty until the JSON loads,
    // then `default_filter_state` replaces them with the "all on" defaults
    // (unless the URL already provided a subset).
    let pathway_metric = use_signal(|| {
        initial.p_metric.as_deref().map_or(
            PathwayMetric::Auroc,
            url_state::UrlState::parse_pathway_metric,
        )
    });
    let pathway_index: Signal<usize> = use_signal(|| 0);
    let pathway_families: Signal<HashSet<Family>> = use_signal(|| {
        initial
            .families
            .as_ref()
            .map(|v| url_state::slugs_as_families(v))
            .unwrap_or_default()
    });
    let pathway_mz_keys: Signal<HashSet<ExpKey>> = use_signal(|| {
        initial
            .mz
            .as_ref()
            .map(|v| url_state::floats_as_exp_keys(v))
            .unwrap_or_default()
    });
    let pathway_int_keys: Signal<HashSet<ExpKey>> = use_signal(|| {
        initial
            .int
            .as_ref()
            .map(|v| url_state::floats_as_exp_keys(v))
            .unwrap_or_default()
    });
    let pathway_weighted: Signal<HashSet<WeightedChoice>> = use_signal(|| {
        initial
            .weighted
            .as_ref()
            .map(|v| url_state::slugs_as_weighted(v))
            .unwrap_or_default()
    });

    // Pending pathway label from the URL. Resolved to a `pathway_index`
    // once the JSON arrives in the configuration panel.
    let initial_pathway_label = initial.pathway.clone();
    let pending_pathway_label: Signal<Option<String>> =
        use_signal(|| initial_pathway_label.clone());

    let datasets_for_resource = datasets.clone();
    let dataset_resource = use_resource(move || {
        let datasets = datasets_for_resource.clone();
        let chosen = dataset_index();
        async move {
            let entry = datasets
                .get(chosen)
                .ok_or_else(|| "dataset index out of range".to_string())?;
            let configs = fetch::load_configs(&entry.configs_url).await?;
            let grid = fetch::load_grid(&entry.grid_url).await?;
            // Carrying `chosen` lets downstream consumers ignore the
            // payload while the user is mid-switch (the resource value
            // outlives the dataset_index change until the new fetch
            // resolves).
            Ok::<(usize, Vec<ConfigEntry>, DistributionGrid), String>((chosen, configs, grid))
        }
    });

    let datasets_for_pathway = datasets.clone();
    let pathway_resource = use_resource(move || {
        let datasets = datasets_for_pathway.clone();
        let chosen = dataset_index();
        async move {
            let entry = datasets
                .get(chosen)
                .ok_or_else(|| "dataset index out of range".to_string())?;
            let url = entry
                .pathways_url
                .as_ref()
                .ok_or_else(|| "no pathways URL for this dataset".to_string())?;
            let data = fetch::load_pathway_lines(url).await?;
            Ok::<(usize, PathwayLinesData), String>((chosen, data))
        }
    });

    let pathways_url_for_panel = datasets
        .get(dataset_index())
        .and_then(|entry| entry.pathways_url.clone());

    // Mirror every UI choice into the URL so it stays shareable.
    let datasets_for_url = datasets.clone();
    use_effect(move || {
        let mut state = url_state::UrlState::default();
        let tab = active_tab();
        state.tab = Some(tab.slug().to_string());
        if let Some(entry) = datasets_for_url.get(dataset_index()) {
            state.dataset = Some(entry.slug.clone());
        }
        match tab {
            ViewerTab::Heatmaps => {
                state.metric = Some(metric_kind().slug().to_string());
                state.scale = Some(color_scale().slug().to_string());
                state.alpha = Some(alpha_milli());
                state.d = Some(d_centi());
                if let Some(key) = *active_key.read() {
                    state.config = Some(key.slug());
                }
            }
            ViewerTab::Pathways => {
                state.p_metric =
                    Some(url_state::UrlState::pathway_metric_slug(pathway_metric()).to_string());
                if let Some(label) = pending_pathway_label.read().clone() {
                    state.pathway = Some(label);
                }
                let families_snapshot = pathway_families.read().clone();
                if !families_snapshot.is_empty() {
                    state.families = Some(url_state::families_as_slugs(&families_snapshot));
                }
                let mz_snapshot = pathway_mz_keys.read().clone();
                if !mz_snapshot.is_empty() {
                    state.mz = Some(url_state::exp_keys_as_floats(&mz_snapshot));
                }
                let int_snapshot = pathway_int_keys.read().clone();
                if !int_snapshot.is_empty() {
                    state.int = Some(url_state::exp_keys_as_floats(&int_snapshot));
                }
                let weighted_snapshot = pathway_weighted.read().clone();
                if !weighted_snapshot.is_empty() {
                    state.weighted = Some(url_state::weighted_as_slugs(&weighted_snapshot));
                }
            }
        }
        url_state::write(&state);
    });

    rsx! {
        ConfigurationPanel {
            datasets: datasets.clone(),
            dataset_index,
            active_tab,
            metric_kind,
            color_scale,
            alpha_milli,
            d_centi,
            active_key,
            dataset_resource,
            pathway_resource,
            pathway_metric,
            pathway_index,
            pathway_families,
            pathway_mz_keys,
            pathway_int_keys,
            pathway_weighted,
            pending_pathway_label,
        }
        match active_tab() {
            ViewerTab::Heatmaps => rsx! {
                HeatmapPanel {
                    datasets: datasets.clone(),
                    dataset_index,
                    metric_kind,
                    color_scale,
                    alpha_milli,
                    d_centi,
                    active_key,
                    dataset_resource,
                }
            },
            ViewerTab::Pathways => rsx! {
                PathwayPanel {
                    pathways_url: pathways_url_for_panel.clone(),
                    pathway_resource,
                    dataset_index,
                    pathway_index,
                    metric: pathway_metric,
                    families: pathway_families,
                    mz_keys: pathway_mz_keys,
                    int_keys: pathway_int_keys,
                    weighted: pathway_weighted,
                }
            },
        }
    }
}

#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn ConfigurationPanel(
    datasets: Vec<DatasetEntry>,
    dataset_index: Signal<usize>,
    active_tab: Signal<ViewerTab>,
    metric_kind: Signal<MetricKind>,
    color_scale: Signal<ColorScale>,
    alpha_milli: Signal<u32>,
    d_centi: Signal<u32>,
    active_key: Signal<Option<ConfigKey>>,
    dataset_resource: DatasetResource,
    pathway_resource: PathwayResource,
    pathway_metric: Signal<PathwayMetric>,
    pathway_index: Signal<usize>,
    pathway_families: Signal<HashSet<Family>>,
    pathway_mz_keys: Signal<HashSet<ExpKey>>,
    pathway_int_keys: Signal<HashSet<ExpKey>>,
    pathway_weighted: Signal<HashSet<WeightedChoice>>,
    pending_pathway_label: Signal<Option<String>>,
) -> Element {
    let current_idx = dataset_index();
    let configs_state = dataset_resource.read_unchecked();
    // Only build the catalog when the resource payload corresponds to the
    // currently selected dataset. While the user is mid-switch the resource
    // still holds the previous dataset's configs, and seeding `active_key`
    // from them would point the heatmap at the old config under a new
    // dataset label (the "needs two clicks to switch" bug).
    let catalog: Option<Rc<ConfigCatalog>> = match &*configs_state {
        Some(Ok((fetched_idx, configs, _))) if *fetched_idx == current_idx => {
            Some(Rc::new(ConfigCatalog::new(
                &configs
                    .iter()
                    .map(|c| (c.config_index, c.config.clone()))
                    .collect::<Vec<_>>(),
            )))
        }
        _ => None,
    };

    // Initialise active_key on first dataset load.
    if let Some(catalog_rc) = &catalog {
        if active_key.read().is_none() {
            if let Some(first) = catalog_rc.first() {
                let mut sig = active_key;
                sig.set(Some(first));
            }
        }
    }

    let dataset_has_pathways = datasets
        .get(current_idx)
        .map(|entry| entry.pathways_url.is_some())
        .unwrap_or(false);

    // Once the pathway JSON arrives, seed every filter set with every
    // value seen in the data so the default is "all on", and resolve any
    // pending URL-derived pathway label into a concrete `pathway_index`.
    // Same guard as above: only act when the payload matches the active
    // dataset.
    let pathway_state = pathway_resource.read_unchecked();
    if let Some(Ok((fetched_idx, data))) = &*pathway_state {
        if *fetched_idx == current_idx {
            if pathway_families.read().is_empty() {
                let (fams, mz, ints, weighted) = default_filter_state(data);
                pathway_families.clone().set(fams);
                pathway_mz_keys.clone().set(mz);
                pathway_int_keys.clone().set(ints);
                pathway_weighted.clone().set(weighted);
            }
            if let Some(label) = pending_pathway_label.read().clone() {
                if let Some(idx) = data.pathways.iter().position(|p| p.label == label) {
                    if pathway_index() != idx {
                        pathway_index.clone().set(idx);
                    }
                }
            } else if let Some(entry) = data.pathways.get(pathway_index()) {
                pending_pathway_label.clone().set(Some(entry.label.clone()));
            }
        }
    }

    rsx! {
        section { class: "panel",
            div { class: "panel-head",
                span { aria_hidden: "true",
                    Icon { width: 18, height: 18, icon: FaSliders, class: "panel-icon" }
                }
                h2 { class: "panel-title", "Configuration" }
            }
            p { class: "panel-subtitle",
                "Each row toggles one axis of the experiment. Combinations that aren't in the dataset are dimmed."
            }

            // Dataset row
            div { class: "field-row",
                span {
                    class: "field-label",
                    title: "Which precomputed similarity scan to inspect.",
                    "Dataset"
                }
                div { class: "pill-row",
                    for (i, entry) in datasets.iter().enumerate() {
                        Pill {
                            key: "{entry.slug}",
                            label: entry.label.clone(),
                            active: dataset_index() == i,
                            disabled: false,
                            tone: PillTone::Blue,
                            description: Some(dataset_description(&entry.slug)),
                            on_click: {
                                let mut sig = dataset_index;
                                let mut key_sig = active_key;
                                let mut fam_sig = pathway_families;
                                let mut mz_sig = pathway_mz_keys;
                                let mut int_sig = pathway_int_keys;
                                let mut weighted_sig = pathway_weighted;
                                move |_| {
                                    sig.set(i);
                                    key_sig.set(None);
                                    fam_sig.set(HashSet::new());
                                    mz_sig.set(HashSet::new());
                                    int_sig.set(HashSet::new());
                                    weighted_sig.set(HashSet::new());
                                }
                            },
                        }
                    }
                }
            }

            // View (tab) row
            div { class: "field-row",
                span {
                    class: "field-label",
                    title: "Heatmaps shows the p-value and KS-statistic grids. Pathways shows AUROC / AUPRC line plots from the pathway-classification task.",
                    "View"
                }
                div { class: "pill-row",
                    for (tab, label) in ViewerTab::ALL {
                        Pill {
                            key: "{tab.slug()}",
                            label: label.to_string(),
                            active: active_tab() == tab,
                            disabled: matches!(tab, ViewerTab::Pathways) && !dataset_has_pathways,
                            tone: PillTone::Rust,
                            description: Some(match tab {
                                ViewerTab::Heatmaps => "Existing per-config distribution heatmaps".to_string(),
                                ViewerTab::Pathways => "Per-pathway AUROC / AUPRC line plots from pathway_discriminability_per_class.parquet".to_string(),
                            }),
                            on_click: {
                                let mut sig = active_tab;
                                move |_| sig.set(tab)
                            },
                        }
                    }
                }
            }

            // Tab-specific configuration
            if active_tab() == ViewerTab::Pathways {
                PathwayConfigSection {
                    pathway_resource,
                    dataset_index,
                    pathway_metric,
                    pathway_index,
                    pathway_families,
                    pathway_mz_keys,
                    pathway_int_keys,
                    pathway_weighted,
                    pending_pathway_label,
                }
            } else { HeatmapConfigSection {
                catalog,
                metric_kind,
                color_scale,
                alpha_milli,
                d_centi,
                active_key,
            } }
        }
    }
}

/// Pathway-classification configuration rows. Rendered when the user
/// switches to the Pathways tab. Reads the same pathway-lines resource as
/// the right-panel `PathwayPanel` so toggling a filter does not trigger a
/// network request.
#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn PathwayConfigSection(
    pathway_resource: PathwayResource,
    dataset_index: Signal<usize>,
    pathway_metric: Signal<PathwayMetric>,
    pathway_index: Signal<usize>,
    pathway_families: Signal<HashSet<Family>>,
    pathway_mz_keys: Signal<HashSet<ExpKey>>,
    pathway_int_keys: Signal<HashSet<ExpKey>>,
    pathway_weighted: Signal<HashSet<WeightedChoice>>,
    pending_pathway_label: Signal<Option<String>>,
) -> Element {
    let current_idx = dataset_index();
    let state = pathway_resource.read_unchecked();
    let data = match &*state {
        // Hold the previous dataset's payload back while a new fetch is
        // mid-flight so the filter pills don't briefly re-render against
        // stale configs.
        Some(Ok((fetched_idx, data))) if *fetched_idx == current_idx => data.clone(),
        _ => {
            return rsx! {
                p { class: "loading", "Loading pathway data…" }
            };
        }
    };

    let mut available_families: Vec<Family> = Vec::new();
    let mut available_mz: Vec<f64> = Vec::new();
    let mut available_int: Vec<f64> = Vec::new();
    let mut available_weighted: Vec<Option<bool>> = Vec::new();
    for entry in &data.configs {
        let fam = pathway_panel::parse_family(&entry.family);
        if !available_families.contains(&fam) {
            available_families.push(fam);
        }
        if !available_mz.iter().any(|v| (v - entry.mz_exp).abs() < 1e-9) {
            available_mz.push(entry.mz_exp);
        }
        if !available_int
            .iter()
            .any(|v| (v - entry.intensity_exp).abs() < 1e-9)
        {
            available_int.push(entry.intensity_exp);
        }
        if !available_weighted.contains(&entry.weighted) {
            available_weighted.push(entry.weighted);
        }
    }
    available_mz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    available_int.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let metric_options: [(PathwayMetric, &str, &str); 4] = [
        (
            PathwayMetric::Auroc,
            "AUROC",
            "Area under the ROC curve, computed from the similarity score's ranking of (query, candidate) pairs. Insensitive to class prior.",
        ),
        (
            PathwayMetric::Auprc,
            "AUPRC",
            "Area under the precision-recall curve. Sensitive to the share of positives in the class, useful for rare pathways.",
        ),
        (
            PathwayMetric::Accuracy,
            "Accuracy",
            "Share of queries whose argmax-similarity pathway prediction matches the truth.",
        ),
        (
            PathwayMetric::Mcc,
            "MCC",
            "Matthews correlation coefficient of the one-vs-rest classifier. Robust to class imbalance.",
        ),
    ];
    let active_pathway = data.pathways.get(pathway_index());
    let metric_supported = |metric: PathwayMetric| {
        active_pathway.is_some_and(|p| {
            matches!(
                metric,
                PathwayMetric::Auroc if p.auroc.is_some()
            ) || matches!(metric, PathwayMetric::Auprc if p.auprc.is_some())
                || matches!(metric, PathwayMetric::Accuracy if p.accuracy.is_some())
                || matches!(metric, PathwayMetric::Mcc if p.mcc.is_some())
        })
    };

    rsx! {
        // Metric row: AUROC | AUPRC | Accuracy | MCC
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Pathway-pair classifier metric drawn on the y-axis. Pills greyed out when the active pathway has no values for that metric.",
                "Metric"
            }
            div { class: "pill-row",
                for (metric, label, description) in metric_options {
                    Pill {
                        key: "{label}",
                        label: label.to_string(),
                        active: pathway_metric() == metric,
                        disabled: !metric_supported(metric),
                        tone: PillTone::Rust,
                        description: Some(description.to_string()),
                        on_click: {
                            let mut sig = pathway_metric;
                            move |_| sig.set(metric)
                        },
                    }
                }
            }
        }

        // Pathway row
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Aggregate (micro) pools every pair into one classifier; the other entries are per-pathway one-vs-rest classifiers.",
                "Pathway"
            }
            div { class: "pill-row",
                for (idx, entry) in data.pathways.iter().enumerate() {
                    Pill {
                        key: "{entry.label}",
                        label: entry.label.clone(),
                        active: pathway_index() == idx,
                        disabled: false,
                        tone: PillTone::Blue,
                        description: Some(match entry.kind.as_str() {
                            "aggregate" => "Pooled micro-averaged classifier across every (query, candidate) pair.".to_string(),
                            _ => "One-vs-rest classifier with the named pathway as the positive class.".to_string(),
                        }),
                        on_click: {
                            let mut sig = pathway_index;
                            let mut fams = pathway_families;
                            let mut mz = pathway_mz_keys;
                            let mut ints = pathway_int_keys;
                            let mut weighted = pathway_weighted;
                            let mut label_sig = pending_pathway_label;
                            let label_value = entry.label.clone();
                            let data_for_reset = data.clone();
                            move |_| {
                                sig.set(idx);
                                label_sig.set(Some(label_value.clone()));
                                let (fs, mzs, is, ws) = default_filter_state(&data_for_reset);
                                fams.set(fs);
                                mz.set(mzs);
                                ints.set(is);
                                weighted.set(ws);
                            }
                        },
                    }
                }
            }
        }

        // Family row
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Toggle similarity-metric families. Lines for deselected families disappear from the plot.",
                "Family"
            }
            div { class: "pill-row",
                for family in available_families.iter().copied() {
                    Pill {
                        key: "{family.label()}",
                        label: family.label().to_string(),
                        active: pathway_families.read().contains(&family),
                        disabled: false,
                        tone: PillTone::Green,
                        description: Some(family_description(family).to_string()),
                        on_click: {
                            let mut sig = pathway_families;
                            move |_| {
                                let mut set = sig.read().clone();
                                if set.contains(&family) {
                                    set.remove(&family);
                                } else {
                                    set.insert(family);
                                }
                                sig.set(set);
                            }
                        },
                    }
                }
            }
        }

        // m/z exponent row
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Toggle m/z exponents. Encoded as line dash pattern in the plot.",
                "m/z exp"
            }
            div { class: "pill-row",
                for mz in available_mz.iter().copied() {
                    Pill {
                        key: "mz-{mz}",
                        label: format!("{mz:.1}"),
                        active: pathway_mz_keys.read().contains(&ExpKey::from_f64(mz)),
                        disabled: false,
                        tone: PillTone::Rust,
                        description: Some(mz_exp_description(mz)),
                        on_click: {
                            let mut sig = pathway_mz_keys;
                            move |_| {
                                let key = ExpKey::from_f64(mz);
                                let mut set = sig.read().clone();
                                if set.contains(&key) {
                                    set.remove(&key);
                                } else {
                                    set.insert(key);
                                }
                                sig.set(set);
                            }
                        },
                    }
                }
            }
        }

        // Intensity exponent row
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Toggle intensity exponents. Encoded as colour mix factor in the plot.",
                "int exp"
            }
            div { class: "pill-row",
                for intensity in available_int.iter().copied() {
                    Pill {
                        key: "int-{intensity}",
                        label: format!("{intensity:.2}"),
                        active: pathway_int_keys.read().contains(&ExpKey::from_f64(intensity)),
                        disabled: false,
                        tone: PillTone::Rust,
                        description: Some(int_exp_description(intensity)),
                        on_click: {
                            let mut sig = pathway_int_keys;
                            move |_| {
                                let key = ExpKey::from_f64(intensity);
                                let mut set = sig.read().clone();
                                if set.contains(&key) {
                                    set.remove(&key);
                                } else {
                                    set.insert(key);
                                }
                                sig.set(set);
                            }
                        },
                    }
                }
            }
        }

        // Weighted row
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Toggle entropy weighting. The flag is only meaningful for entropy / modified-entropy configs.",
                "weighted"
            }
            div { class: "pill-row",
                for value in available_weighted.iter().copied() {
                    {
                        let choice = WeightedChoice::from_optional(value);
                        rsx! {
                            Pill {
                                key: "{choice.label()}",
                                label: choice.label().to_string(),
                                active: pathway_weighted.read().contains(&choice),
                                disabled: false,
                                tone: PillTone::Green,
                                description: Some(value.map_or_else(
                                    || "Configs without an entropy-weighting flag (cosine and modified-cosine families).".to_string(),
                                    |flag| weighted_description(flag).to_string(),
                                )),
                                on_click: {
                                    let mut sig = pathway_weighted;
                                    move |_| {
                                        let mut set = sig.read().clone();
                                        if set.contains(&choice) {
                                            set.remove(&choice);
                                        } else {
                                            set.insert(choice);
                                        }
                                        sig.set(set);
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Heatmap-specific configuration rows. Extracted so the outer
/// `ConfigurationPanel` can switch between this and `PathwayConfigSection`
/// based on the active tab.
#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn HeatmapConfigSection(
    catalog: Option<Rc<ConfigCatalog>>,
    metric_kind: Signal<MetricKind>,
    color_scale: Signal<ColorScale>,
    alpha_milli: Signal<u32>,
    d_centi: Signal<u32>,
    active_key: Signal<Option<ConfigKey>>,
) -> Element {
    rsx! {
        // Metric (kind) row
        div { class: "field-row",
                span {
                    class: "field-label",
                    title: "Which statistic of the score distribution comparison is mapped to color.",
                    "Metric"
                }
                div { class: "pill-row",
                    for (kind, label) in MetricKind::ALL {
                        Pill {
                            label: label.to_string(),
                            active: metric_kind() == kind,
                            disabled: false,
                            tone: PillTone::Rust,
                            description: Some(metric_kind_description(kind).to_string()),
                            on_click: {
                                let mut sig = metric_kind;
                                move |_| sig.set(kind)
                            },
                        }
                    }
                }
            }

            // Color-scale row
            div { class: "field-row",
                span {
                    class: "field-label",
                    title: "How the metric value is mapped to color: linear treats all values the same; log emphasises differences across orders of magnitude.",
                    "Scale"
                }
                div { class: "pill-row",
                    for (scale, label) in ColorScale::ALL {
                        Pill {
                            label: label.to_string(),
                            active: color_scale() == scale,
                            disabled: false,
                            tone: PillTone::Green,
                            description: Some(scale_description(scale).to_string()),
                            on_click: {
                                let mut sig = color_scale;
                                move |_| sig.set(scale)
                            },
                        }
                    }
                }
            }

            // Configuration-decomposed rows (require catalog loaded)
            if let Some(catalog) = catalog.clone() {
                ConfigSelector { catalog: catalog, active_key: active_key }
            } else {
                p { class: "loading", "Loading dataset configs…" }
            }

            // Alpha slider (log scale 10^-3 .. 10^0)
            div { class: "field-row",
                span {
                    class: "field-label",
                    title: "Significance threshold (probability cutoff) for the dashed p-value contour drawn on the heatmap. Smaller α means a stricter cutoff.",
                    "α (p-value)"
                }
                div { class: "pill-row",
                    input {
                        class: "alpha-slider",
                        r#type: "range",
                        min: "0",
                        max: "300",
                        step: "1",
                        value: "{alpha_milli}",
                        list: "log-decade-ticks",
                        aria_label: "Significance threshold alpha for the p-value contour, logarithmic scale from 0.001 to 1.",
                        aria_valuetext: "{format_log_slider(alpha_milli)}",
                        oninput: {
                            let mut sig = alpha_milli;
                            move |event: Event<FormData>| {
                                if let Ok(value) = event.value().parse::<u32>() {
                                    sig.set(value);
                                }
                            }
                        },
                    }
                    span { class: "alpha-value", "α = {format_log_slider(alpha_milli)}" }
                }
            }

            // D-statistic slider (log scale 10^-3 .. 10^0)
            div { class: "field-row",
                span {
                    class: "field-label",
                    title: "Maximum CDF-gap threshold for the dashed D contour. Smaller D means the two distributions have to be very close before the contour fires.",
                    "D (KS statistic)"
                }
                div { class: "pill-row",
                    input {
                        class: "alpha-slider",
                        r#type: "range",
                        min: "0",
                        max: "300",
                        step: "1",
                        value: "{d_centi}",
                        list: "log-decade-ticks",
                        aria_label: "Kolmogorov-Smirnov D threshold for the effect-size contour, logarithmic scale from 0.001 to 1.",
                        aria_valuetext: "{format_log_slider(d_centi)}",
                        oninput: {
                            let mut sig = d_centi;
                            move |event: Event<FormData>| {
                                if let Ok(value) = event.value().parse::<u32>() {
                                    sig.set(value);
                                }
                            }
                        },
                    }
                    span { class: "alpha-value", "D = {format_log_slider(d_centi)}" }
                }
            }

            // Shared datalist for the decade tick marks on both sliders.
            datalist { id: "log-decade-ticks",
                option { value: "0", label: "0.001" }
                option { value: "100", label: "0.01" }
                option { value: "200", label: "0.1" }
                option { value: "300", label: "1" }
            }
    }
}

/// Convert a log-scale slider position (0..=300) to its represented value
/// in `[10^-3, 10^0]`.
fn log_slider_value(slider: u32) -> f64 {
    let log_value = f64::from(slider).mul_add(0.01, -3.0);
    10.0_f64.powf(log_value)
}

/// Format the current log-slider value for the inline UI label, switching
/// to scientific notation for tiny values.
fn format_log_slider(slider: Signal<u32>) -> String {
    let value = log_slider_value(slider());
    if value < 0.01 {
        format!("{value:.2e}")
    } else {
        format!("{value:.3}")
    }
}

#[component]
#[allow(non_snake_case)]
fn ConfigSelector(catalog: Rc<ConfigCatalog>, active_key: Signal<Option<ConfigKey>>) -> Element {
    let Some(current) = *active_key.read() else {
        return rsx! { p { class: "loading", "Choosing default config…" } };
    };

    let available_families = catalog.families();
    let mz_options = catalog.mz_for(current.family);
    let int_options = catalog.int_for(current.family, current.mz_exp);
    let supports_weighted = catalog.supports_weighted(current.family);

    rsx! {
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Which spectral similarity definition was used to score each query against the references.",
                "Similarity"
            }
            div { class: "pill-row",
                for family in Family::ALL {
                    Pill {
                        key: "{family.label()}",
                        label: family.label().to_string(),
                        active: current.family == family,
                        disabled: !available_families.contains(&family),
                        tone: PillTone::Red,
                        description: Some(family_description(family).to_string()),
                        on_click: {
                            let catalog = catalog.clone();
                            let mut key_sig = active_key;
                            move |_| {
                                if let Some(next) = catalog.closest(family) {
                                    key_sig.set(Some(next));
                                }
                            }
                        },
                    }
                }
            }
        }
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Exponent applied to each peak's m/z (mass-to-charge) value before it contributes to the similarity score.",
                "m/z exponent"
            }
            div { class: "pill-row",
                for mz in mz_options.iter().copied() {
                    Pill {
                        key: "{mz.label()}",
                        label: mz.label(),
                        active: current.mz_exp == mz,
                        disabled: false,
                        tone: PillTone::Blue,
                        description: Some(mz_exp_description(mz.as_f64())),
                        on_click: {
                            let catalog = catalog.clone();
                            let mut key_sig = active_key;
                            move |_| {
                                if let Some(next) = catalog.closest_for_mz(current.family, mz) {
                                    key_sig.set(Some(next));
                                }
                            }
                        },
                    }
                }
            }
        }
        div { class: "field-row",
            span {
                class: "field-label",
                title: "Exponent applied to each peak's intensity. Less than 1 compresses the dynamic range so weaker peaks count more.",
                "intensity exp"
            }
            div { class: "pill-row",
                for int_exp in int_options.iter().copied() {
                    Pill {
                        key: "{int_exp.label()}",
                        label: int_exp.label(),
                        active: current.int_exp == int_exp,
                        disabled: false,
                        tone: PillTone::Rust,
                        description: Some(int_exp_description(int_exp.as_f64())),
                        on_click: {
                            let mut key_sig = active_key;
                            move |_| {
                                let mut updated = current;
                                updated.int_exp = int_exp;
                                key_sig.set(Some(updated));
                            }
                        },
                    }
                }
            }
        }
        if supports_weighted {
            div { class: "field-row",
                span {
                    class: "field-label",
                    title: "Whether the entropy similarity applies the information-content weighting that emphasises informative peaks.",
                    "weighting"
                }
                div { class: "pill-row",
                    Pill {
                        label: "Weighted".to_string(),
                        active: current.weighted == Some(true),
                        disabled: false,
                        tone: PillTone::Green,
                        description: Some(weighted_description(true).to_string()),
                        on_click: {
                            let mut key_sig = active_key;
                            move |_| {
                                let mut updated = current;
                                updated.weighted = Some(true);
                                key_sig.set(Some(updated));
                            }
                        },
                    }
                    Pill {
                        label: "Unweighted".to_string(),
                        active: current.weighted == Some(false),
                        disabled: false,
                        tone: PillTone::Green,
                        description: Some(weighted_description(false).to_string()),
                        on_click: {
                            let mut key_sig = active_key;
                            move |_| {
                                let mut updated = current;
                                updated.weighted = Some(false);
                                key_sig.set(Some(updated));
                            }
                        },
                    }
                }
            }
        }
    }
}

#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn Pill(
    label: String,
    active: bool,
    disabled: bool,
    tone: PillTone,
    /// Plain-language explanation shown as the button's tooltip and exposed
    /// to assistive tech as the accessible name. When `None` the visible
    /// `label` text is used.
    #[props(default)]
    description: Option<String>,
    on_click: EventHandler<MouseEvent>,
) -> Element {
    let mut class = String::from("pill ");
    class.push_str(tone.class());
    if disabled {
        class.push_str(" is-disabled");
    } else if active {
        class.push_str(" is-active");
    }
    let description = description.unwrap_or_else(|| label.clone());
    rsx! {
        button {
            class: "{class}",
            r#type: "button",
            disabled: disabled,
            title: "{description}",
            aria_label: "{description}",
            aria_pressed: if active { "true" } else { "false" },
            onclick: move |event| on_click.call(event),
            "{label}"
        }
    }
}

/// Color family used for a pill row. Each axis of the configuration panel
/// gets its own tone so the eye can group choices by category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PillTone {
    /// Primary blue (dataset, m/z exponent).
    Blue,
    /// Warm rust (metric kind, intensity exponent).
    Rust,
    /// Sage green (scale, weighting).
    Green,
    /// Muted red (similarity family).
    Red,
}

impl PillTone {
    /// Return the CSS class suffix for this tone (e.g. `tone-blue`).
    #[must_use]
    pub const fn class(self) -> &'static str {
        match self {
            Self::Blue => "tone-blue",
            Self::Rust => "tone-rust",
            Self::Green => "tone-green",
            Self::Red => "tone-red",
        }
    }
}

#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn HeatmapPanel(
    datasets: Vec<DatasetEntry>,
    dataset_index: Signal<usize>,
    metric_kind: Signal<MetricKind>,
    color_scale: Signal<ColorScale>,
    alpha_milli: Signal<u32>,
    d_centi: Signal<u32>,
    active_key: Signal<Option<ConfigKey>>,
    dataset_resource: DatasetResource,
) -> Element {
    let current_idx = dataset_index();
    let dataset_label = datasets
        .get(current_idx)
        .map(|d| d.label.clone())
        .unwrap_or_default();
    let state = dataset_resource.read_unchecked();
    rsx! {
        section { class: "panel",
            div { class: "panel-head",
                span { aria_hidden: "true",
                    Icon { width: 18, height: 18, icon: FaChartArea, class: "panel-icon" }
                }
                h2 { class: "panel-title", "Heatmap" }
            }
            match &*state {
                // Hold the previous dataset's heatmap back while the new
                // fetch is still in flight so the figure never appears
                // under a mismatched dataset label.
                Some(Ok((fetched_idx, configs, grid))) if *fetched_idx == current_idx => rsx! {
                    HeatmapBody {
                        configs: configs.clone(),
                        grid: Rc::new(GridBundle::from(grid)),
                        dataset_label: dataset_label.clone(),
                        metric_kind,
                        color_scale,
                        alpha_milli,
                        d_centi,
                        active_key,
                    }
                },
                Some(Err(error)) => rsx! {
                    p { class: "error", "Failed to load dataset: {error}" }
                },
                _ => rsx! { p { class: "loading", "Loading dataset…" } },
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct GridBundle {
    mean_delta: ndarray::Array3<f64>,
    ks_statistic: ndarray::Array3<f64>,
    ks_pvalue_asymptotic: ndarray::Array3<f64>,
    wasserstein_1d: ndarray::Array3<f64>,
}

impl From<&DistributionGrid> for GridBundle {
    fn from(grid: &DistributionGrid) -> Self {
        Self {
            mean_delta: grid.mean_delta.clone(),
            ks_statistic: grid.ks_statistic.clone(),
            ks_pvalue_asymptotic: grid.ks_pvalue_asymptotic.clone(),
            wasserstein_1d: grid.wasserstein_1d.clone(),
        }
    }
}

#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn HeatmapBody(
    configs: Vec<ConfigEntry>,
    grid: Rc<GridBundle>,
    dataset_label: String,
    metric_kind: Signal<MetricKind>,
    color_scale: Signal<ColorScale>,
    alpha_milli: Signal<u32>,
    d_centi: Signal<u32>,
    active_key: Signal<Option<ConfigKey>>,
) -> Element {
    let catalog = Rc::new(ConfigCatalog::new(
        &configs
            .iter()
            .map(|c| (c.config_index, c.config.clone()))
            .collect::<Vec<_>>(),
    ));

    let grid_for_scales = grid.clone();
    let scales = use_memo(move || {
        let views = GridViews {
            mean_delta: grid_for_scales.mean_delta.view(),
            ks_statistic: grid_for_scales.ks_statistic.view(),
            ks_pvalue_asymptotic: grid_for_scales.ks_pvalue_asymptotic.view(),
            wasserstein_1d: grid_for_scales.wasserstein_1d.view(),
        };
        Scales::from_grids(&views)
    });

    let grid_for_render = grid.clone();
    let catalog_for_render = catalog.clone();
    let dataset_label_for_render = dataset_label.clone();
    let svg = use_memo(move || -> Result<(String, String, String), String> {
        let key = active_key
            .read()
            .ok_or_else(|| "no config selected".to_string())?;
        let config_index = catalog_for_render
            .index_for(&key)
            .ok_or_else(|| "selected config not in dataset".to_string())?;
        let slug = catalog_for_render
            .slug(config_index)
            .unwrap_or("(unknown)")
            .to_string();
        let metric_kind_value = metric_kind();
        let scale_value = color_scale();
        let metric = combine_metric(metric_kind_value, scale_value);
        let alpha = log_slider_value(alpha_milli());
        let d_threshold = log_slider_value(d_centi());
        let views = GridViews {
            mean_delta: grid_for_render.mean_delta.view(),
            ks_statistic: grid_for_render.ks_statistic.view(),
            ks_pvalue_asymptotic: grid_for_render.ks_pvalue_asymptotic.view(),
            wasserstein_1d: grid_for_render.wasserstein_1d.view(),
        };
        let s = *scales.read();
        let svg_string = render_cell_svg(
            &slug,
            views,
            &s,
            config_index,
            metric,
            &[alpha],
            &[d_threshold],
            Some(dataset_label_for_render.as_str()),
        )
        .map_err(|error| error.to_string())?;
        let caption = figure_caption(
            &dataset_label_for_render,
            key,
            metric_kind_value,
            scale_value,
            alpha,
            d_threshold,
        );
        let responsive = responsive_svg::make_responsive(svg_string);
        let data_uri = svg_export::to_data_uri(&responsive);
        let metric_slug = format!("{}_{}", metric_kind_value.slug(), scale_value.slug());
        let filename_stem = svg_export::sanitize_filename(&format!(
            "heatmap_{dataset}_{slug}_{metric_slug}",
            dataset = dataset_label_for_render.as_str()
        ));
        Ok((caption, data_uri, filename_stem))
    });

    rsx! {
        match &*svg.read_unchecked() {
            Ok((caption, data_uri, filename_stem)) => {
                let caption_html = caption_to_html(caption);
                rsx! {
                    figure { class: "heatmap-figure",
                        div { class: "heatmap-frame",
                            img {
                                class: "heatmap-image",
                                src: "{data_uri}",
                                alt: "{caption}",
                            }
                        }
                        div { class: "figure-actions",
                            a {
                                class: "pill download-pill",
                                href: "{data_uri}",
                                download: "{filename_stem}.svg",
                                title: "Download this figure as SVG. Right-click the image itself for \"Copy image\" or \"Save image as ...\".",
                                aria_label: "Download this figure as SVG",
                                span { aria_hidden: "true",
                                    Icon { width: 14, height: 14, icon: FaDownload, class: "panel-icon" }
                                }
                                "SVG"
                            }
                        }
                        figcaption { class: "figure-caption",
                            span { aria_hidden: "true",
                                Icon { width: 14, height: 14, icon: FaWaveSquare, class: "panel-icon" }
                            }
                            span { dangerous_inner_html: "{caption_html}" }
                        }
                    }
                }
            },
            Err(error) => rsx! {
                p { class: "error",
                    span { aria_hidden: "true",
                        Icon { width: 14, height: 14, icon: FaCircleInfo, class: "panel-icon" }
                    }
                    "Render error: {error}"
                }
            },
        }
    }
}

#[allow(dead_code, clippy::missing_const_for_fn)]
fn _silence_database_icon_unused() {
    // The `FaDatabase` import is kept for future dataset rows but isn't on
    // a render path yet; this no-op makes clippy's `unused_imports` happy
    // without burning a `#[allow]` on the top-level import.
    let _ = FaDatabase;
}

#[cfg(test)]
mod caption_tests {
    use super::caption_to_html;

    #[test]
    fn coloring_respects_word_boundaries() {
        let caption =
            "Caption title here. It is the share of queries whose predicted-pathway matches.";
        let html = caption_to_html(caption);
        // The literal color word "red" sits inside "predicted-pathway" and
        // must not be wrapped in a coloured span.
        assert!(
            !html.contains("p<strong style=\"color: #9d4133;\">red</strong>icted"),
            "should not colour the `red` substring inside `predicted`: {html}"
        );
    }

    #[test]
    fn coloring_still_applies_to_standalone_color_words() {
        let caption = "Title here. The dashed coral curve traces α and the cyan curve traces D.";
        let html = caption_to_html(caption);
        assert!(
            html.contains("<strong style=\"color: #c84766;\">coral</strong>"),
            "expected coral to be coloured: {html}"
        );
        assert!(
            html.contains("<strong style=\"color: #2b85ac;\">cyan</strong>"),
            "expected cyan to be coloured: {html}"
        );
    }

    #[test]
    fn title_bold_extends_past_decimal_periods() {
        // The previous heuristic stopped the title at the first '.' it
        // saw, which broke on captions whose first sentence mentions a
        // decimal exponent (e.g. m/z^3 · intensity^0.6). The fix only
        // treats a '.' as a sentence boundary when followed by whitespace
        // or end-of-string.
        let caption = "Heatmap under the cosine similarity with per-peak weighting w ∝ m/z^3 · intensity^0.6. Each cell reports the metric value.";
        let html = caption_to_html(caption);
        assert!(
            html.contains("intensity^0.6.</strong>"),
            "the bold title should include the whole exponent through `intensity^0.6.`: {html}"
        );
        assert!(
            !html.contains("intensity^0.</strong>"),
            "the bold title must not stop at the decimal `.`: {html}"
        );
    }

    #[test]
    fn title_bold_falls_back_to_full_caption_when_no_period() {
        let html = caption_to_html("Caption without a period");
        assert!(
            html.contains("<strong>Caption without a period</strong>"),
            "expected the whole caption to be bold when it has no sentence terminator: {html}"
        );
    }
}
