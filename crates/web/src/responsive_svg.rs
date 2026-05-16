//! Rewrite plotters' fixed-pixel SVG output so the browser can scale it
//! to fit its container.
//!
//! Plotters emits something like
//! `<svg width="2000" height="1800" xmlns="...">`. Without a `viewBox`,
//! CSS `width: 100%; height: auto` is ignored and the SVG always lays out
//! at its intrinsic pixel size, forcing a horizontal scrollbar. Adding a
//! `viewBox` lets the browser keep the aspect ratio and scale the drawing
//! to whatever width the container offers.

/// Rewrite the root `<svg>` opening tag of `svg`:
/// - replace `width="W"` / `height="H"` with `width="100%" height="100%"`
/// - inject a `viewBox="0 0 W H"` based on the original W / H attributes
/// - inject `preserveAspectRatio="xMidYMid meet"` so it scales uniformly
///
/// Also rounds the legend's background and border `<rect>` corners by
/// adding `rx="8"`, since plotters emits sharp corners and the only rects
/// with the legend's fingerprint are the legend box itself.
///
/// If the tag already has a `viewBox`, leaves it alone (only normalises
/// width/height to `100%`). If the tag can't be parsed (no `<svg ...>` head),
/// returns the input unchanged.
#[must_use]
pub fn make_responsive(mut svg: String) -> String {
    svg = round_legend_corners(svg);
    let Some(open_end) = svg.find('>') else {
        return svg;
    };
    let open = &svg[..open_end];
    if !open.starts_with("<svg") {
        return svg;
    }
    let width = extract_attr(open, "width").and_then(parse_pixel_dim);
    let height = extract_attr(open, "height").and_then(parse_pixel_dim);
    let has_viewbox = open.contains(" viewBox=") || open.contains("\tviewBox=");
    let viewbox_attr = if has_viewbox {
        String::new()
    } else if let (Some(w), Some(h)) = (width, height) {
        format!(" viewBox=\"0 0 {w} {h}\"")
    } else {
        String::new()
    };
    let preserve_attr = if open.contains("preserveAspectRatio=") {
        String::new()
    } else {
        " preserveAspectRatio=\"xMidYMid meet\"".to_string()
    };
    let replaced_open = replace_attr(open, "width", "100%");
    let replaced_open = replace_attr(&replaced_open, "height", "100%");
    let mut new_open = replaced_open;
    if !viewbox_attr.is_empty() {
        // Insert just after the `<svg`.
        if let Some(after_tag) = new_open.find("<svg").map(|i| i + "<svg".len()) {
            new_open.insert_str(after_tag, &viewbox_attr);
        }
    }
    if !preserve_attr.is_empty() {
        if let Some(after_tag) = new_open.find("<svg").map(|i| i + "<svg".len()) {
            new_open.insert_str(after_tag, &preserve_attr);
        }
    }
    svg.replace_range(..open_end, &new_open);
    svg
}

/// Find the legend's background rect (`opacity="0.82" fill="#FFFFFF"`) and
/// its border rect (`fill="none" stroke="#000000"`) and inject `rx="8"` on
/// each so the legend box renders with softly rounded corners.
fn round_legend_corners(svg: String) -> String {
    let signatures: &[(&str, &str)] = &[
        (
            " opacity=\"0.82\" fill=\"#FFFFFF\" stroke=\"none\"",
            " rx=\"8\" opacity=\"0.82\" fill=\"#FFFFFF\" stroke=\"none\"",
        ),
        (
            " fill=\"none\" stroke=\"#000000\"",
            " rx=\"8\" fill=\"none\" stroke=\"#000000\"",
        ),
    ];
    let mut out = svg;
    for (from, to) in signatures {
        out = out.replace(from, to);
    }
    out
}

fn extract_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

fn parse_pixel_dim(raw: &str) -> Option<u32> {
    let trimmed = raw.trim_end_matches("px");
    trimmed.parse().ok()
}

fn replace_attr(tag: &str, name: &str, value: &str) -> String {
    let needle = format!("{name}=\"");
    if let Some(start) = tag.find(&needle) {
        let after = start + needle.len();
        if let Some(rel_end) = tag[after..].find('"') {
            let mut out = String::with_capacity(tag.len());
            out.push_str(&tag[..after]);
            out.push_str(value);
            out.push_str(&tag[after + rel_end..]);
            return out;
        }
    }
    // No existing attribute: insert immediately after `<svg`.
    tag.replacen("<svg", &format!("<svg {name}=\"{value}\""), 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_viewbox_and_normalises_dimensions() {
        let input =
            "<svg width=\"2000\" height=\"1800\" xmlns=\"http://www.w3.org/2000/svg\"></svg>"
                .to_string();
        let out = make_responsive(input);
        assert!(out.contains("viewBox=\"0 0 2000 1800\""), "got {out}");
        assert!(out.contains("width=\"100%\""), "got {out}");
        assert!(out.contains("height=\"100%\""), "got {out}");
        assert!(out.contains("preserveAspectRatio=\"xMidYMid meet\""));
    }

    #[test]
    fn leaves_existing_viewbox_alone() {
        let input = "<svg viewBox=\"0 0 100 50\" width=\"100\" height=\"50\"></svg>".to_string();
        let out = make_responsive(input);
        // viewBox preserved as-is
        assert!(out.contains("viewBox=\"0 0 100 50\""));
    }
}
