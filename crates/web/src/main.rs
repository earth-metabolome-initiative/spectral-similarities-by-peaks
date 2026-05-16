//! Browser viewer entry point.
//!
//! Boots a Dioxus 0.7 single-page app that fetches per-dataset
//! `distribution_grid.npz` files from `data/` and renders heatmaps on
//! demand via the `spectral-render` crate compiled to WebAssembly.

mod config_key;
mod fetch;
mod responsive_svg;

use std::rc::Rc;

use config_key::{ConfigCatalog, ConfigKey, Family};
use dioxus::prelude::*;
use dioxus_free_icons::Icon;
use dioxus_free_icons::icons::fa_brands_icons::FaGithub;
use dioxus_free_icons::icons::fa_solid_icons::{
    FaChartArea, FaCircleInfo, FaDatabase, FaSliders, FaWaveSquare,
};
use spectral_render::{GridViews, Metric, Scales, render_cell_svg};

use crate::fetch::{ConfigEntry, DatasetEntry, DistributionGrid, Manifest};

const DATA_BASE_URL: &str = "data/";

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
}

/// Linear vs logarithmic color mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum ColorScale {
    Log,
    Linear,
}

impl ColorScale {
    const ALL: [(Self, &'static str); 2] = [(Self::Log, "Log"), (Self::Linear, "Linear")];
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
/// inside the caption. Reads better than [`ConfigKey::pretty`] inside flowing
/// prose because the comma there clashes with the surrounding sentence.
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
fn caption_to_html(caption: &str) -> String {
    let (title, rest) = caption
        .find('.')
        .map_or((caption, ""), |idx| caption.split_at(idx + 1));

    // Order matters: longer / compound tokens must be replaced before the
    // shorter ones they contain (e.g. "near-white" before "white").
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
        rest_html = rest_html.replace(word, &replacement);
    }
    format!("<strong>{title}</strong>{rest_html}")
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

const FONTS_HREF: &str = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;700&family=Syne:wght@700&display=swap";

fn main() {
    console_error_panic_hook::set_once();
    dioxus::launch(App);
}

#[component]
#[allow(non_snake_case)]
fn App() -> Element {
    let manifest = use_resource(|| async move { fetch::load_manifest(DATA_BASE_URL).await });

    rsx! {
        document::Stylesheet { href: FONTS_HREF }
        document::Stylesheet { href: "style.css" }

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
            h1 { class: "hero-title",
                "Spectral similarities by "
                span { class: "hero-accent", "peaks" }
            }
            p { class: "abstract",
                "This experiment quantifies, on two reference MS2 corpora (the 443 905-spectrum "
                a {
                    href: "https://zenodo.org/records/20042904",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "harmonized annotated dataset"
                }
                " and a 100 000-query sample of the "
                a {
                    href: "https://zenodo.org/records/20040772",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "GeMS-A10 corpus"
                }
                "), how the empirical score distribution of a spectral similarity changes when each spectrum is truncated to its top-"
                em { "k" }
                " intensity peaks for "
                em { "k" }
                " spanning 1 to 128. Eighteen similarity configurations covering "
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
                " and Modified entropy (this work, by analogy with Modified cosine) at multiple m/z and intensity exponents are compared cell-by-cell on the resulting 128 by 128 grid using four metrics: asymptotic Kolmogorov-Smirnov p-value, KS D statistic, signed difference of means, and 1-D Wasserstein distance. The distributions stabilise quickly across every configuration. The negligible-drift threshold D ≤ 0.05 is reached at 4 to 47 retained peaks and the strict-equivalence threshold D ≤ 0.01 at 7 to 103, with the intensity exponent dominating the per-configuration diversity ranking. Because each cell aggregates 6 to 28 million pairwise scores, the asymptotic p-values fall deep inside "
                a {
                    href: "https://en.wikipedia.org/wiki/Lindley%27s_paradox",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Lindley's-paradox"
                }
                " territory and collapse to numerically zero across most of the grid. The figures therefore lean on the sample-size-invariant D statistic as the primary effect-size cue. Pathway-classification metrics, namely AUROC and AUPRC of a k-nearest-pathway classifier aggregating these similarities against fixed "
                a {
                    href: "https://doi.org/10.1186/s13321-022-00624-5",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "NPClassifier-pathway"
                }
                " representatives, are still being computed on the cluster and will be folded in once the underlying parquet finishes transferring."
            }
            div { class: "pill-row",
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
                    class: "pill",
                    href: "https://github.com/LucaCappelletti94/mascot-rs",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    title: "Open the mascot-rs companion library on GitHub.",
                    aria_label: "Open the mascot-rs companion library on GitHub.",
                    span { aria_hidden: "true",
                        Icon { width: 14, height: 14, icon: FaGithub }
                    }
                    "mascot-rs"
                }
            }
        }
    }
}

#[component]
#[allow(non_snake_case)]
fn Viewer(datasets: Vec<DatasetEntry>) -> Element {
    let dataset_index = use_signal(|| 0_usize);
    let metric_kind = use_signal(|| MetricKind::KsPvalueAsymptotic);
    let color_scale = use_signal(|| ColorScale::Log);
    // Slider positions on a 0..=300 axis mapped through `log_slider_value`
    // to a value in `[10^-3, 10^0]`. 170 ≈ 0.05, the literature default.
    let alpha_milli = use_signal(|| 170_u32);
    let d_centi = use_signal(|| 170_u32);
    // Holds the user-selected ConfigKey. Re-derived when the dataset changes.
    let active_key: Signal<Option<ConfigKey>> = use_signal(|| None);

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
            Ok::<(Vec<ConfigEntry>, DistributionGrid), String>((configs, grid))
        }
    });

    rsx! {
        ConfigurationPanel {
            datasets: datasets.clone(),
            dataset_index,
            metric_kind,
            color_scale,
            alpha_milli,
            d_centi,
            active_key,
            dataset_resource,
        }
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
    }
}

#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn ConfigurationPanel(
    datasets: Vec<DatasetEntry>,
    dataset_index: Signal<usize>,
    metric_kind: Signal<MetricKind>,
    color_scale: Signal<ColorScale>,
    alpha_milli: Signal<u32>,
    d_centi: Signal<u32>,
    active_key: Signal<Option<ConfigKey>>,
    dataset_resource: Resource<Result<(Vec<ConfigEntry>, DistributionGrid), String>>,
) -> Element {
    let configs_state = dataset_resource.read_unchecked();
    let catalog: Option<Rc<ConfigCatalog>> = match &*configs_state {
        Some(Ok((configs, _))) => Some(Rc::new(ConfigCatalog::new(
            &configs
                .iter()
                .map(|c| (c.config_index, c.config.clone()))
                .collect::<Vec<_>>(),
        ))),
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
                            label: entry.label.clone(),
                            active: dataset_index() == i,
                            disabled: false,
                            tone: PillTone::Blue,
                            description: Some(dataset_description(&entry.slug)),
                            on_click: {
                                let mut sig = dataset_index;
                                let mut key_sig = active_key;
                                move |_| {
                                    sig.set(i);
                                    key_sig.set(None);
                                }
                            },
                        }
                    }
                }
            }

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
    dataset_resource: Resource<Result<(Vec<ConfigEntry>, DistributionGrid), String>>,
) -> Element {
    let dataset_label = datasets
        .get(dataset_index())
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
                Some(Ok((configs, grid))) => rsx! {
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
                None => rsx! { p { class: "loading", "Loading dataset…" } },
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
    let svg = use_memo(move || -> Result<(String, String), String> {
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
        Ok((caption, responsive_svg::make_responsive(svg_string)))
    });

    rsx! {
        match &*svg.read_unchecked() {
            Ok((caption, markup)) => {
                let caption_html = caption_to_html(caption);
                rsx! {
                    figure { class: "heatmap-figure",
                        div {
                            class: "heatmap-frame",
                            role: "img",
                            aria_label: "{caption}",
                            dangerous_inner_html: "{markup}",
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
