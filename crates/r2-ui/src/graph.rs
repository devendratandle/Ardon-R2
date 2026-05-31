//! GraphPanel — SVG → wgpu texture for the R2 Graphics sub-window.
//!
//! Named "graph" rather than "plot" deliberately: `plot()` is an R2
//! built-in (and R's main graphics function); using "graph" for the
//! framework widget keeps the namespace clean.
//!
//! The host hands us an SVG byte slice (R2's `r2-graphics` emits SVG
//! today). We rasterize once via `usvg` + `tiny-skia` into an RGBA
//! pixmap, upload that into the renderer's atlas as an [`ImageHandle`],
//! and paint a textured quad each frame. Re-rasterization happens only
//! when `set_svg` is called with new bytes.
//!
//! No widget-side animation, no chart logic — the engine produces a
//! finished SVG, we display it. Keeps the framework decoupled from
//! plot semantics.

use crate::grid::Rect;
use crate::render::{Frame, ImageHandle, Renderer, TextAnchorKind};
use crate::svg_text::{extract_texts, parse_css_color, strip_text, SvgText, SvgTextAnchor};
use crate::theme::{Color, Theme};

/// Persistent slot dimensions for a `GraphPanel`. One slot is
/// allocated per panel on the first paint and reused for every
/// re-rasterisation — the panel writes a sub-rectangle of this
/// slot at the panel's exact display pixel size, and displays
/// that sub-rectangle 1:1. Keeps the atlas from fragmenting.
const SLOT_W: u32 = 1024;
const SLOT_H: u32 = 768;

pub struct GraphPanel {
    svg: Option<Vec<u8>>,
    /// Geometry-only SVG (text extracted and stripped). Cached so
    /// resvg gets a clean string each paint without re-parsing the
    /// full source.
    svg_geometry: Option<String>,
    /// All `<text>` labels extracted from the original SVG. Painted
    /// in `paint()` via the fontdue path, NOT through resvg — gives
    /// Console-quality crispness on every label.
    texts: Vec<SvgText>,
    /// SVG's authored viewBox dimensions, used to map text x/y
    /// coordinates from SVG space into panel space.
    svg_w: f32,
    svg_h: f32,
    /// One-shot atlas slot allocated on first paint. The actual
    /// SVG content fills the (last_w, last_h) sub-rectangle from
    /// the slot's top-left; the rest of the slot is unused.
    slot: Option<ImageHandle>,
    /// Last (raster_w, raster_h) we wrote into the slot. Used so
    /// the painter knows what sub-rectangle of the slot to draw.
    last_size: Option<(u32, u32)>,
    /// When true, the next paint will re-rasterise even if size
    /// didn't change (set by `set_svg`).
    dirty: bool,
}

impl Default for GraphPanel {
    fn default() -> Self { Self::new() }
}

impl GraphPanel {
    pub fn new() -> Self {
        Self {
            svg: None,
            svg_geometry: None,
            texts: Vec::new(),
            svg_w: 600.0, svg_h: 400.0,
            slot: None, last_size: None, dirty: false,
        }
    }

    /// Install an SVG payload. The next paint re-rasterises.
    /// Text elements are extracted up-front so resvg only sees the
    /// geometry (the GraphPanel renders text itself via fontdue).
    pub fn set_svg(&mut self, bytes: impl Into<Vec<u8>>) {
        let bytes = bytes.into();
        if let Ok(svg_str) = std::str::from_utf8(&bytes) {
            self.texts = extract_texts(svg_str);
            self.svg_geometry = Some(strip_text(svg_str));
            if let Some((w, h)) = parse_svg_dimensions(svg_str) {
                self.svg_w = w;
                self.svg_h = h;
            }
        } else {
            self.texts.clear();
            self.svg_geometry = None;
        }
        self.svg = Some(bytes);
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        self.svg = None;
        self.svg_geometry = None;
        self.texts.clear();
        self.last_size = None;
        self.dirty = false;
    }

    pub fn has_plot(&self) -> bool { self.svg.is_some() }

    /// Paint the panel — rasterizes the SVG into the atlas on the first
    /// call (or after `set_svg`), then paints a textured quad each
    /// frame. Shows a placeholder rect + label if no SVG is set or if
    /// rasterization fails.
    pub fn paint(
        &mut self,
        frame: &mut Frame,
        renderer: &mut Renderer,
        rect: Rect,
        theme: &Theme,
    ) {
        // Pixel-perfect rendering with NO atlas fragmentation:
        //   1. Allocate ONE big slot (SLOT_W × SLOT_H) on the first
        //      paint. Reuse it forever via sub-region writes.
        //   2. Each paint, rasterise the SVG to a CPU pixmap at the
        //      panel's exact displayed pixel size (× DPI). SVG uses
        //      `font-size="14px"` so text comes out at 14 physical
        //      pixels regardless of how big the panel is — no
        //      compression on resize.
        //   3. Upload the pixmap as a sub-region into the slot's
        //      top-left.
        //   4. Draw exactly the (target_w × target_h) sub-rectangle
        //      of the slot at the panel rect — 1:1 sampling, no GPU
        //      resample blur.
        let dpi = theme.dpi;
        let target_w = ((rect.w * dpi).round() as u32).max(64).min(SLOT_W);
        let target_h = ((rect.h * dpi).round() as u32).max(64).min(SLOT_H);

        if self.slot.is_none() {
            let placeholder = vec![0u8; (SLOT_W * SLOT_H * 4) as usize];
            self.slot = renderer.upload_image(SLOT_W, SLOT_H, &placeholder);
            // First allocation always triggers a fresh raster.
            self.dirty = true;
        }
        let needs_raster = match self.last_size {
            None => true,
            Some((cw, ch)) => self.dirty
                || (target_w as i32 - cw as i32).abs() > 2
                || (target_h as i32 - ch as i32).abs() > 2,
        };
        if needs_raster {
            self.dirty = false;
            // Rasterise the geometry-only SVG. Text is rendered
            // separately below via the fontdue glyph path — sharper
            // than what resvg produces, and integer-pixel-snapped.
            let geom = self.svg_geometry.as_deref()
                .or_else(|| self.svg.as_deref().and_then(|b| std::str::from_utf8(b).ok()));
            if let (Some(slot), Some(geom_str)) = (self.slot, geom) {
                if let Some((rgba, _, _)) = rasterize_svg(geom_str.as_bytes(), target_w, target_h) {
                    if renderer.replace_image_subregion(
                        slot, 0, 0, target_w, target_h, &rgba)
                    {
                        self.last_size = Some((target_w, target_h));
                    }
                }
            }
        }

        // Always paint a body background so empty / failed states still
        // look intentional.
        frame.paint_rect(rect.x, rect.y, rect.w, rect.h, theme.window_background);

        if let (Some(slot), Some((sw, sh))) = (self.slot, self.last_size) {
            // Draw exactly the (sw × sh) sub-rectangle of the slot
            // at the panel rect. Because we rasterised at the
            // panel's display pixel size, this is a 1:1 pixel-perfect
            // mapping when DPI=1 (or constant-multiplier when scaled).
            frame.paint_image_sub(slot, rect.x, rect.y, rect.w, rect.h,
                                  sw, sh, Color::WHITE);

            // ── Text overlay via fontdue. Each <text> element from
            //    the SVG is painted with integer-pixel-snapped glyphs
            //    in screen space — same path the Console uses, same
            //    crispness. Position is mapped linearly from SVG
            //    coords into panel coords (non-uniform), and font
            //    size uses the geometric-mean of x/y scales so
            //    aspect-mismatched panels don't distort text.
            if !self.texts.is_empty() && self.svg_w > 0.0 && self.svg_h > 0.0 {
                let mx = rect.w / self.svg_w;
                let my = rect.h / self.svg_h;
                let font_scale = (mx * my).sqrt().max(0.1);
                for t in &self.texts {
                    let panel_x = rect.x + t.x * mx;
                    let panel_y = rect.y + t.y * my;
                    let panel_font = (t.font_size * font_scale).max(6.0);
                    let color = parse_css_color(&t.color);
                    let anchor = match t.anchor {
                        SvgTextAnchor::Start  => TextAnchorKind::Start,
                        SvgTextAnchor::Middle => TextAnchorKind::Middle,
                        SvgTextAnchor::End    => TextAnchorKind::End,
                    };
                    frame.paint_text_anchored(
                        renderer, panel_x, panel_y, &t.text, panel_font, color,
                        anchor, t.rotation_deg.unwrap_or(0.0));
                }
            }
        } else {
            // Placeholder text — centered roughly via cell metrics.
            let (cell_w, line_h) = renderer.cell_metrics(theme.font_size);
            let msg = if self.svg.is_some() { "(plot rasterization failed)" }
                      else                  { "No plot yet" };
            let tw = msg.chars().count() as f32 * cell_w;
            let cx = rect.x + (rect.w - tw) * 0.5;
            let cy = rect.y + rect.h * 0.5 + line_h * 0.3;
            frame.paint_text(renderer, cx, cy, msg, theme.font_size,
                             Color::rgba(120, 120, 130, 200));
        }
    }
}

/// Rasterize an SVG byte slice into an RGBA buffer that **exactly**
/// matches the requested `w × h` (the SVG content is stretched to
/// fill — non-uniform scale when the panel aspect differs). Returns
/// `None` if parsing or rendering fails.
///
/// R2 plot SVGs use a fixed viewBox so stretching reads naturally as
/// "draw at the panel's resolution"; the histogram bars and axis grid
/// stay sharp because resvg works in vector space.
/// Pull `width="…"` / `height="…"` out of the SVG opening tag so
/// the panel can map text positions from SVG coords into panel
/// coords on every paint. Forgiving prefix parse — `"600"` and
/// `"600px"` both work.
fn parse_svg_dimensions(svg: &str) -> Option<(f32, f32)> {
    fn extract(svg: &str, attr: &str) -> Option<f32> {
        let needle = format!("{}=\"", attr);
        let i = svg.find(&needle)?;
        let after = &svg[i + needle.len()..];
        let end = after.find('"')?;
        let raw = &after[..end];
        let cleaned: String = raw.chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
            .collect();
        cleaned.parse().ok()
    }
    Some((extract(svg, "width")?, extract(svg, "height")?))
}

fn rasterize_svg(bytes: &[u8], w: u32, h: u32) -> Option<(Vec<u8>, u32, u32)> {
    // Straight SVG → pixmap rasterisation. Text in `font-size="Npx"`
    // *will* scale with the panel — that's a real limitation of the
    // SVG-via-resvg path. The fix is SDF text overlay (separate
    // render pass) planned for a future session. The previous
    // attempt to compensate font-size in-string distorted glyphs
    // when the panel aspect ratio didn't match the SVG's (which is
    // most of the time), so it's removed.
    let mut opt = usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();
    let tree = usvg::Tree::from_data(bytes, &opt).ok()?;

    let svg_size = tree.size();
    let sw = svg_size.width();
    let sh = svg_size.height();
    if sw <= 0.0 || sh <= 0.0 { return None; }

    let mut pixmap = tiny_skia::Pixmap::new(w, h)?;
    let transform = tiny_skia::Transform::from_scale(w as f32 / sw, h as f32 / sh);
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    Some((pixmap.data().to_vec(), w, h))
}
