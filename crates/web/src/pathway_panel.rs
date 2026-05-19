//! Right-panel pathway-classification view for the WASM viewer.
//!
//! Reads `pathway_discriminability_lines.json` once per dataset, then
//! renders an AUROC or AUPRC line plot via
//! [`spectral_render::render_pathway_lines_svg`] every time the user
//! changes pathway, metric, or one of the family / m/z / intensity /
//! weighted filter sets.

use std::collections::HashSet;

use dioxus::prelude::*;
use dioxus_free_icons::Icon;
use dioxus_free_icons::icons::fa_solid_icons::{FaCircleInfo, FaDownload};
use spectral_render::{
    PathwayFamily as RenderFamily, PathwayLineSeries, PathwayMetric, pretty_series_label,
    render_pathway_lines_svg,
};

use crate::config_key::{ExpKey, Family};
use crate::fetch::{PathwayConfigEntry, PathwayLinesData};
use crate::responsive_svg;

/// Shape of the WASM-side pathway-discriminability resource. The leading
/// `usize` is the `dataset_index` the fetch ran for, mirrored from
/// `crate::PathwayResource`.
type TaggedPathwayResource = Resource<Result<(usize, PathwayLinesData), String>>;

/// Choice for the `weighted` filter pill row: every per-class series has a
/// known `Option<bool>` flag and a config that lacks the flag entirely.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WeightedChoice {
    /// Configs whose slug does not carry a `_weighted` suffix.
    NotApplicable,
    /// Entropy variants with `weighted=true` in the slug.
    True,
    /// Entropy variants with `weighted=false` in the slug.
    False,
}

impl WeightedChoice {
    /// Display label for the pill row.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NotApplicable => "no flag",
            Self::True => "weighted",
            Self::False => "unweighted",
        }
    }

    /// Translate the JSON `Option<bool>` into the corresponding choice.
    pub const fn from_optional(value: Option<bool>) -> Self {
        match value {
            None => Self::NotApplicable,
            Some(true) => Self::True,
            Some(false) => Self::False,
        }
    }
}

/// Default state of every filter set for a freshly-selected pathway.
pub fn default_filter_state(
    data: &PathwayLinesData,
) -> (
    HashSet<Family>,
    HashSet<ExpKey>,
    HashSet<ExpKey>,
    HashSet<WeightedChoice>,
) {
    let mut families = HashSet::new();
    let mut mz_keys = HashSet::new();
    let mut int_keys = HashSet::new();
    let mut weighted = HashSet::new();
    for entry in &data.configs {
        families.insert(parse_family(&entry.family));
        mz_keys.insert(ExpKey::from_f64(entry.mz_exp));
        int_keys.insert(ExpKey::from_f64(entry.intensity_exp));
        weighted.insert(WeightedChoice::from_optional(entry.weighted));
    }
    (families, mz_keys, int_keys, weighted)
}

/// Parse the JSON `family` string back into the existing web-side `Family`.
pub fn parse_family(value: &str) -> Family {
    match value {
        "modified-cosine" => Family::ModifiedCosine,
        "entropy" => Family::Entropy,
        "modified-entropy" => Family::ModifiedEntropy,
        _ => Family::Cosine,
    }
}

/// Map the web's `Family` enum to the render crate's enum.
const fn render_family(family: Family) -> RenderFamily {
    match family {
        Family::Cosine => RenderFamily::Cosine,
        Family::ModifiedCosine => RenderFamily::ModifiedCosine,
        Family::Entropy => RenderFamily::Entropy,
        Family::ModifiedEntropy => RenderFamily::ModifiedEntropy,
    }
}

/// Build the `Vec<PathwayLineSeries>` for the active pathway after applying
/// every filter, then render it to SVG and wrap it for responsive display.
pub fn render_svg(
    data: &PathwayLinesData,
    pathway_index: usize,
    metric: PathwayMetric,
    families: &HashSet<Family>,
    mz_keys: &HashSet<ExpKey>,
    int_keys: &HashSet<ExpKey>,
    weighted: &HashSet<WeightedChoice>,
) -> Result<RenderedFigure, String> {
    let pathway = data
        .pathways
        .get(pathway_index)
        .ok_or_else(|| "pathway index out of range".to_string())?;
    let matrix_owner = match metric {
        PathwayMetric::Auroc => &pathway.auroc,
        PathwayMetric::Auprc => &pathway.auprc,
        PathwayMetric::Accuracy => &pathway.accuracy,
        PathwayMetric::Mcc => &pathway.mcc,
    };
    let matrix = matrix_owner.as_ref().ok_or_else(|| {
        format!(
            "{} is not defined for pathway entry \"{}\".",
            metric.y_label(),
            pathway.label
        )
    })?;
    let mut series: Vec<PathwayLineSeries> = Vec::new();
    for (config_index, config) in data.configs.iter().enumerate() {
        if !filter_keeps(config, families, mz_keys, int_keys, weighted) {
            continue;
        }
        let Some(row) = matrix.get(config_index) else {
            continue;
        };
        let points: Vec<(i32, f64)> = row
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| {
                let value = (*cell)?;
                let peak = *data.peak_counts.get(i)?;
                let peak_i32 = i32::try_from(peak).ok()?;
                Some((peak_i32, value))
            })
            .collect();
        if points.is_empty() {
            continue;
        }
        let render_fam = render_family(parse_family(&config.family));
        series.push(PathwayLineSeries {
            label: pretty_series_label(
                render_fam,
                config.mz_exp,
                config.intensity_exp,
                config.weighted,
            ),
            family: render_fam,
            mz_exp: config.mz_exp,
            intensity_exp: config.intensity_exp,
            weighted: config.weighted,
            points,
        });
    }
    let title = format!("{}, {}", metric.title(), pathway.label);
    let svg = render_pathway_lines_svg(&title, metric, &series, 1080, 700)
        .map_err(|err| err.to_string())?;
    let caption = compose_caption(pathway, metric, series.len(), data.configs.len());
    let responsive = responsive_svg::make_responsive(svg);
    let data_uri = crate::svg_export::to_data_uri(&responsive);
    let filename_stem = crate::svg_export::sanitize_filename(&format!(
        "pathway_{}_{}",
        metric.y_label(),
        pathway.label
    ));
    Ok(RenderedFigure {
        caption,
        data_uri,
        filename_stem,
    })
}

/// Bundle returned by [`render_svg`] so the caller (the `PathwayPanel`
/// component) can render the figure as an `<img>` tag, attach a sensible
/// `download` filename to the export anchor, and surface the same plain
/// caption as both the screen-reader label and the `<figcaption>`.
pub struct RenderedFigure {
    /// Plain caption used as the `<img alt>` text and as the
    /// `<figcaption>` content (after bold-title HTML markup).
    pub caption: String,
    /// `data:image/svg+xml;base64,…` URI used by the `<img src>` and by
    /// the download anchor.
    pub data_uri: String,
    /// Slug used as the download filename stem (the renderer adds
    /// `.svg`).
    pub filename_stem: String,
}

/// Build the figure caption shown under the plot. The first sentence is
/// the figure title (later wrapped in `<strong>` by
/// [`crate::caption_to_html`]). The remaining sentences describe the
/// classifier, the metric, the axes, the visual encoding, and the current
/// filter state.
fn compose_caption(
    pathway: &crate::fetch::PathwayEntry,
    metric: PathwayMetric,
    visible_configs: usize,
    total_configs: usize,
) -> String {
    let metric_short = metric.y_label();
    let metric_full = match metric {
        PathwayMetric::Auroc => "Area Under the Receiver Operating Characteristic curve",
        PathwayMetric::Auprc => "Area Under the Precision-Recall curve",
        PathwayMetric::Accuracy => "Accuracy at the argmax-similarity decision rule",
        PathwayMetric::Mcc => "Matthews Correlation Coefficient",
    };
    let baseline = match metric {
        PathwayMetric::Auroc => {
            "It reads 0.5 when the score ranks pairs no better than chance, and 1.0 when every positive pair scores above every negative pair."
        }
        PathwayMetric::Auprc => {
            "Its baseline equals the share of positives in the comparison (the class prior), so AUPRC is more informative than AUROC when positives are rare."
        }
        PathwayMetric::Accuracy => {
            "It is the share of queries whose predicted-pathway label (argmax over candidate similarities) matches the truth. With seven base pathways and skewed support a near-1.0 score can still hide a poor minority-class classifier."
        }
        PathwayMetric::Mcc => {
            "It is the Pearson correlation between the binary truth and the binary prediction, ranging from -1 (perfect disagreement) through 0 (independent of the truth) to +1 (perfect agreement). Robust to class imbalance, unlike accuracy."
        }
    };
    let (title, classifier_sentence) = match pathway.kind.as_str() {
        "aggregate" => (
            format!(
                "{metric_short} ({metric_full}) of the pooled pathway-pair classifier on the harmonized dataset, micro-averaged across every (query, candidate) similarity-score pair.",
            ),
            "The classifier labels a pair as positive when the candidate spectrum's NPC pathway matches the query's NPC pathway. The similarity score ranks pairs, and the metric measures how well that ranking separates same-pathway pairs from different-pathway pairs.".to_string(),
        ),
        _ => (
            format!(
                "{metric_short} ({metric_full}) of the one-vs-rest pathway-pair classifier on the harmonized dataset, with {pathway_label} fixed as the positive class.",
                pathway_label = pathway.label,
            ),
            format!(
                "The classifier restricts to (query, candidate) pairs whose query is annotated with the {pathway} NPC pathway. The similarity score ranks those pairs, and the metric measures how well that ranking promotes candidates also annotated with {pathway} above candidates annotated with any other pathway.",
                pathway = pathway.label,
            ),
        ),
    };
    let encoding = "Line colour encodes the similarity family (blue for cosine, orange for modified-cosine, green for entropy, red for modified-entropy). Dash pattern encodes the m/z exponent (solid for 0.0, dashed for 1.0, dotted for 3.0). A colour mix factor distinguishes the intensity exponent and the entropy-weighting flag within each family.".to_string();
    let axes = "The x-axis is the retained top-intensity peak count, swept from 1 to 128. The y-axis is the metric value.".to_string();
    let filter_note = format!(
        "Showing {visible_configs} of {total_configs} similarity configs. Toggle the family, m/z, intensity, and weighted pills on the left to filter further."
    );
    format!("{title} {classifier_sentence} {baseline} {axes} {encoding} {filter_note}")
}

/// Return true when every filter set includes the config's axis value.
fn filter_keeps(
    config: &PathwayConfigEntry,
    families: &HashSet<Family>,
    mz_keys: &HashSet<ExpKey>,
    int_keys: &HashSet<ExpKey>,
    weighted: &HashSet<WeightedChoice>,
) -> bool {
    families.contains(&parse_family(&config.family))
        && mz_keys.contains(&ExpKey::from_f64(config.mz_exp))
        && int_keys.contains(&ExpKey::from_f64(config.intensity_exp))
        && weighted.contains(&WeightedChoice::from_optional(config.weighted))
}

/// Right-panel renderer. Reads the pathway-lines resource and dispatches
/// to either the line plot, the loading state, the disabled state (when
/// the dataset has no pathways URL), or an error.
#[component]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn PathwayPanel(
    pathways_url: Option<String>,
    pathway_resource: TaggedPathwayResource,
    dataset_index: Signal<usize>,
    pathway_index: Signal<usize>,
    metric: Signal<PathwayMetric>,
    families: Signal<HashSet<Family>>,
    mz_keys: Signal<HashSet<ExpKey>>,
    int_keys: Signal<HashSet<ExpKey>>,
    weighted: Signal<HashSet<WeightedChoice>>,
) -> Element {
    if pathways_url.is_none() {
        return rsx! {
            section { class: "panel panel-figure",
                p { class: "info",
                    span { aria_hidden: "true",
                        Icon { width: 14, height: 14, icon: FaCircleInfo, class: "panel-icon" }
                    }
                    "Per-pathway classification is not defined for this dataset, since GeMS-A10 carries no NPC pathway annotations."
                }
            }
        };
    }
    let current_idx = dataset_index();
    let state = pathway_resource.read_unchecked();
    let data = match &*state {
        // Resource value can outlive a dataset_index change while the
        // new fetch is in flight. Treat that intermediate window as
        // "still loading" instead of rendering the previous dataset's
        // pathway plot under the new dataset label.
        Some(Ok((fetched_idx, data))) if *fetched_idx == current_idx => data.clone(),
        Some(Ok(_)) | None => {
            return rsx! {
                section { class: "panel panel-figure",
                    p { class: "loading", "Loading pathway data…" }
                }
            };
        }
        Some(Err(err)) => {
            let message = err.clone();
            return rsx! {
                section { class: "panel panel-figure",
                    p { class: "error",
                        span { aria_hidden: "true",
                            Icon { width: 14, height: 14, icon: FaCircleInfo, class: "panel-icon" }
                        }
                        "Could not load pathway data: {message}"
                    }
                }
            };
        }
    };
    let pathway_idx = pathway_index();
    let metric_value = metric();
    let families_set = families.read().clone();
    let mz_set = mz_keys.read().clone();
    let int_set = int_keys.read().clone();
    let weighted_set = weighted.read().clone();
    let rendered = render_svg(
        &data,
        pathway_idx,
        metric_value,
        &families_set,
        &mz_set,
        &int_set,
        &weighted_set,
    );
    match rendered {
        Ok(RenderedFigure {
            caption,
            data_uri,
            filename_stem,
        }) => {
            let caption_html = crate::caption_to_html(&caption);
            rsx! {
                section { class: "panel panel-figure",
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
                                Icon { width: 14, height: 14, icon: FaCircleInfo, class: "panel-icon" }
                            }
                            span { dangerous_inner_html: "{caption_html}" }
                        }
                    }
                }
            }
        }
        Err(error) => rsx! {
            section { class: "panel panel-figure",
                p { class: "error",
                    span { aria_hidden: "true",
                        Icon { width: 14, height: 14, icon: FaCircleInfo, class: "panel-icon" }
                    }
                    "Render error: {error}"
                }
            }
        },
    }
}
