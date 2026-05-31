//! Extract `<text>` elements from an SVG so the GraphPanel can
//! render them itself via the same fontdue path the Console uses
//! (integer-pixel-snapped, Console-quality crispness).
//!
//! The flow:
//!   1. `extract_texts(svg)` walks the SVG source and returns a
//!      `Vec<SvgText>` describing every text label.
//!   2. `strip_text(svg)` returns the same SVG minus its `<text>`
//!      elements — that stripped SVG is what gets handed to resvg
//!      for geometry-only rasterisation.
//!   3. `GraphPanel` paints the rasterised geometry, then overlays
//!      every `SvgText` using `Frame::paint_text`.
//!
//! Parser is a forgiving state machine over the SVG string — no
//! XML library needed because our engine generates a deterministic
//! subset of SVG (no namespaces on text, no CDATA, no `<tspan>`
//! children yet). When R2 grows full SVG support we can swap to
//! roxmltree without touching call sites.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvgTextAnchor { Start, Middle, End }

#[derive(Debug, Clone)]
pub struct SvgText {
    pub x: f32,
    pub y: f32,
    pub text: String,
    /// Authored size in SVG user units (≡ px per CSS).
    pub font_size: f32,
    /// `fill` attribute value verbatim (e.g. `"black"`, `"#123456"`).
    pub color: String,
    pub anchor: SvgTextAnchor,
    /// First numeric argument of a `transform="rotate(a, ...)"`
    /// attribute, if any. Degrees, clockwise positive in SVG.
    pub rotation_deg: Option<f32>,
    pub bold: bool,
    pub italic: bool,
}

/// Walk the SVG source, return every `<text ...>...</text>` block as
/// an `SvgText`. Malformed blocks (missing `</text>`, missing `x` /
/// `y`) are skipped.
pub fn extract_texts(svg: &str) -> Vec<SvgText> {
    let mut out = Vec::new();
    let mut cursor = svg;
    loop {
        let start = match cursor.find("<text") { Some(i) => i, None => break };
        let after_start = &cursor[start + 5..];
        // Make sure this is a real `<text` element, not `<textPath`
        // or similar — next char must be whitespace or `>`.
        if !after_start.starts_with(|c: char| c.is_whitespace() || c == '>') {
            cursor = after_start;
            continue;
        }
        let gt = match after_start.find('>') { Some(i) => i, None => break };
        let attrs = &after_start[..gt];
        let body_start = &after_start[gt + 1..];
        let close = match body_start.find("</text>") { Some(i) => i, None => break };
        let body = &body_start[..close];
        if let Some(lbl) = build_label(attrs, body) {
            out.push(lbl);
        }
        cursor = &body_start[close + 7..];
    }
    out
}

/// Return the SVG source with every `<text>...</text>` removed.
/// Used to hand resvg a geometry-only document so it doesn't draw
/// the text (we draw it ourselves with snap-to-pixel glyphs).
pub fn strip_text(svg: &str) -> String {
    let mut out = String::with_capacity(svg.len());
    let mut cursor = svg;
    loop {
        let start = match cursor.find("<text") { Some(i) => i, None => break };
        out.push_str(&cursor[..start]);
        let after_start = &cursor[start..];
        let after_keyword = &after_start[5..];
        if !after_keyword.starts_with(|c: char| c.is_whitespace() || c == '>') {
            // Not a real `<text>` — keep what we matched and continue.
            out.push_str(&after_start[..5]);
            cursor = after_keyword;
            continue;
        }
        match after_start.find("</text>") {
            Some(end) => cursor = &after_start[end + 7..],
            None => { cursor = ""; break; }
        }
    }
    out.push_str(cursor);
    out
}

// ─── internals ─────────────────────────────────────────────────────

fn build_label(attrs: &str, body: &str) -> Option<SvgText> {
    let x = attr_f32(attrs, "x")?;
    let y = attr_f32(attrs, "y")?;
    let font_size = attr_size_px(attrs, "font-size").unwrap_or(12.0);
    let color = attr_str(attrs, "fill").unwrap_or_else(|| "black".to_string());
    let anchor = match attr_str(attrs, "text-anchor").as_deref() {
        Some("middle") => SvgTextAnchor::Middle,
        Some("end")    => SvgTextAnchor::End,
        _              => SvgTextAnchor::Start,
    };
    let rotation_deg = parse_rotate(attrs);
    let weight = attr_str(attrs, "font-weight").unwrap_or_default();
    let style  = attr_str(attrs, "font-style") .unwrap_or_default();
    let bold   = weight == "bold" || weight == "700";
    let italic = style == "italic";
    let text = decode_xml_entities(body.trim());
    Some(SvgText { x, y, text, font_size, color, anchor, rotation_deg, bold, italic })
}

fn attr_str(s: &str, name: &str) -> Option<String> {
    let key = format!("{}=\"", name);
    let i = s.find(&key)?;
    let after = &s[i + key.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn attr_f32(s: &str, name: &str) -> Option<f32> {
    let raw = attr_str(s, name)?;
    raw.trim().parse().ok()
}

fn attr_size_px(s: &str, name: &str) -> Option<f32> {
    let raw = attr_str(s, name)?;
    let trimmed = raw.trim_end_matches("px").trim_end_matches("pt").trim_end_matches("em");
    trimmed.parse().ok()
}

fn parse_rotate(s: &str) -> Option<f32> {
    let key = "transform=\"rotate(";
    let i = s.find(key)?;
    let after = &s[i + key.len()..];
    // First numeric argument — terminates on `,` or `)` (CSS allows
    // a comma-or-space separator).
    let end = after.find(|c: char| c == ',' || c == ')' || c == ' ').unwrap_or(after.len());
    after[..end].trim().parse().ok()
}

fn decode_xml_entities(s: &str) -> String {
    s.replace("&amp;", "&")
     .replace("&lt;",  "<")
     .replace("&gt;",  ">")
     .replace("&quot;","\"")
     .replace("&apos;","'")
}

/// Parse a CSS color into a `Color`. Supports `"#rrggbb"`,
/// `"#rrggbbaa"`, and the dozen common named colors used by our
/// plot defaults. Unknown names fall back to opaque black so a
/// typo never makes a label invisible.
pub fn parse_css_color(s: &str) -> crate::theme::Color {
    use crate::theme::Color;
    let s = s.trim();
    if let Some(rest) = s.strip_prefix('#') {
        let bytes = rest.as_bytes();
        let h = |c: u8| -> u8 {
            match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => 10 + c - b'a',
                b'A'..=b'F' => 10 + c - b'A',
                _ => 0,
            }
        };
        match bytes.len() {
            6 => return Color::rgba(h(bytes[0])*16 + h(bytes[1]),
                                    h(bytes[2])*16 + h(bytes[3]),
                                    h(bytes[4])*16 + h(bytes[5]), 255),
            8 => return Color::rgba(h(bytes[0])*16 + h(bytes[1]),
                                    h(bytes[2])*16 + h(bytes[3]),
                                    h(bytes[4])*16 + h(bytes[5]),
                                    h(bytes[6])*16 + h(bytes[7])),
            3 => return Color::rgba(h(bytes[0])*17, h(bytes[1])*17, h(bytes[2])*17, 255),
            _ => {}
        }
    }
    // Common named colors. Tiny table — the SVG defaults R2 emits
    // only use a handful, and unknown names fall through to black.
    match s.to_ascii_lowercase().as_str() {
        "black"    => Color::rgb(0, 0, 0),
        "white"    => Color::rgb(255, 255, 255),
        "red"      => Color::rgb(255, 0, 0),
        "green"    => Color::rgb(0, 128, 0),
        "blue"     => Color::rgb(0, 0, 255),
        "gray" | "grey"        => Color::rgb(128, 128, 128),
        "lightgray" | "lightgrey" => Color::rgb(211, 211, 211),
        "darkgray"  | "darkgrey"  => Color::rgb(169, 169, 169),
        "navy"     => Color::rgb(0, 0, 128),
        "steelblue"=> Color::rgb(70, 130, 180),
        "lightblue"=> Color::rgb(173, 216, 230),
        "yellow"   => Color::rgb(255, 255, 0),
        "orange"   => Color::rgb(255, 165, 0),
        "purple"   => Color::rgb(128, 0, 128),
        _ => Color::rgb(0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_simple_text() {
        let svg = r#"<svg><text x="10" y="20" font-size="14px" fill="black">Hi</text></svg>"#;
        let labels = extract_texts(svg);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].x, 10.0);
        assert_eq!(labels[0].y, 20.0);
        assert_eq!(labels[0].font_size, 14.0);
        assert_eq!(labels[0].text, "Hi");
    }

    #[test]
    fn strip_removes_text() {
        let svg = r#"<svg><rect/><text x="0" y="0">Hi</text><rect/></svg>"#;
        let stripped = strip_text(svg);
        assert!(!stripped.contains("<text"));
        assert!(stripped.contains("<rect/>"));
    }

    #[test]
    fn parses_anchor_and_rotation() {
        let svg = r#"<text x="5" y="6" text-anchor="middle" transform="rotate(-90,5,6)">Y</text>"#;
        let labels = extract_texts(svg);
        assert_eq!(labels[0].anchor, SvgTextAnchor::Middle);
        assert_eq!(labels[0].rotation_deg, Some(-90.0));
    }

    #[test]
    fn parses_bold_weight() {
        let svg = r#"<text x="0" y="0" font-weight="bold">T</text>"#;
        let labels = extract_texts(svg);
        assert!(labels[0].bold);
    }

    #[test]
    fn ignores_non_text_tags() {
        let svg = r#"<svg><textPath/><tspan>nope</tspan></svg>"#;
        let labels = extract_texts(svg);
        assert!(labels.is_empty());
    }

    #[test]
    fn parses_hex_color() {
        use crate::theme::Color;
        let c = parse_css_color("#3b82f6");
        assert_eq!(c, Color::rgba(0x3b, 0x82, 0xf6, 255));
    }

    #[test]
    fn parses_named_color() {
        use crate::theme::Color;
        assert_eq!(parse_css_color("red"),   Color::rgb(255, 0, 0));
        assert_eq!(parse_css_color("navy"),  Color::rgb(0, 0, 128));
    }
}
