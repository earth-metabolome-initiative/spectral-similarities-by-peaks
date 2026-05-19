//! Convert plotters-emitted SVG strings into `data:` URIs the browser
//! can use as `<img src=...>` (which makes right-click → "Copy image"
//! and "Save image as ..." behave like a normal raster image) and as
//! `<a href=... download=...>` targets (one-click SVG download).
//!
//! Uses base64 rather than percent-encoding so arbitrary UTF-8 content
//! in the SVG (axis labels with `≤` etc.) round-trips without escaping
//! edge cases.

/// Build a `data:image/svg+xml;base64,…` URI from an SVG string.
#[must_use]
pub fn to_data_uri(svg: &str) -> String {
    let encoded = base64_encode(svg.as_bytes());
    format!("data:image/svg+xml;base64,{encoded}")
}

/// Collapse a free-form caption / config label into something safe to
/// use as a download filename stem. Non-alphanumeric characters become
/// underscores, leading and trailing underscores are trimmed, the result
/// is lower-cased and capped to a sensible length.
#[must_use]
pub fn sanitize_filename(stem: &str) -> String {
    let mut out = String::with_capacity(stem.len());
    let mut prev_underscore = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '.') {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_');
    let limited: String = trimmed.chars().take(96).collect();
    if limited.is_empty() {
        "plot".to_string()
    } else {
        limited
    }
}

/// Standard base64 alphabet.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode a byte slice as base64 with `=` padding. Self-contained so we
/// don't need an extra crate just for the data-URI path.
fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut chunks = input.chunks_exact(3);
    for chunk in &mut chunks {
        let triple = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(triple & 0x3F) as usize] as char);
    }
    let remainder = chunks.remainder();
    match remainder.len() {
        1 => {
            let triple = u32::from(remainder[0]) << 16;
            out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let triple = (u32::from(remainder[0]) << 16) | (u32::from(remainder[1]) << 8);
            out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrips_known_values() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn sanitize_collapses_runs_of_punctuation() {
        assert_eq!(
            sanitize_filename("Modified cosine, mz=1.0, int=0.5"),
            "modified_cosine_mz_1.0_int_0.5"
        );
    }

    #[test]
    fn sanitize_replaces_empty_input() {
        assert_eq!(sanitize_filename(""), "plot");
        assert_eq!(sanitize_filename("///"), "plot");
    }

    #[test]
    fn data_uri_prefix_matches_expected_mime() {
        let uri = to_data_uri("<svg/>");
        assert!(uri.starts_with("data:image/svg+xml;base64,"));
    }
}
