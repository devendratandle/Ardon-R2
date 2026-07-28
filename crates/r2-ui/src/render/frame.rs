//! The `Frame` builder — the per-frame drawing API (rects + glyphs).

use crate::theme::Color;
use super::atlas::*;
use super::renderer::*;
use super::{TextAnchorKind, ImageHandle};

// ─── Frame builder ───────────────────────────────────────────────────

pub struct Frame {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices:  Vec<u32>,
}

impl Frame {
    /// Solid rectangle. UV samples the (0,0) opaque-white pixel.
    pub fn paint_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        let c = color_to_f32(color);
        let inv = 1.0 / ATLAS_SIZE as f32;
        let uv = [0.5 * inv, 0.5 * inv];
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            Vertex { pos: [x,     y    ], uv, color: c },
            Vertex { pos: [x + w, y    ], uv, color: c },
            Vertex { pos: [x + w, y + h], uv, color: c },
            Vertex { pos: [x,     y + h], uv, color: c },
        ]);
        self.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    /// Single glyph at (x, y) baseline. Uses `renderer.glyph_uv` to
    /// rasterize-on-demand and pack into the atlas. Call once per
    /// character per frame; the atlas keeps the bitmap for next frame.
    pub fn paint_glyph(&mut self, renderer: &mut Renderer,
                       x: f32, y_baseline: f32, ch: char,
                       size_pt: f32, color: Color)
    {
        self.paint_glyph_clipped(renderer, x, y_baseline, ch, size_pt, color, f32::INFINITY);
    }

    /// Like [`paint_glyph`] but hard-clips the quad at `clip_max_x`
    /// (right edge, in pixels). A glyph crossing the boundary is drawn
    /// only up to `clip_max_x` — the texture U coordinate is narrowed in
    /// proportion so the visible slice stays correct. This is how the
    /// console makes text slide *under* its scrollbar instead of
    /// painting over the bar or outside the console body. There is no
    /// GPU scissor (single-batch renderer), so the clip is done here on
    /// the emitted geometry.
    pub fn paint_glyph_clipped(&mut self, renderer: &mut Renderer,
                       x: f32, y_baseline: f32, ch: char,
                       size_pt: f32, color: Color, clip_max_x: f32)
    {
        self.paint_glyph_clipped_face(renderer, x, y_baseline, ch, size_pt,
                                      color, clip_max_x, crate::Face::Mono);
    }

    /// [`paint_glyph_clipped`] for an explicit typeface. Chrome (menus,
    /// window titles) passes `Face::Ui` for the proportional face; the
    /// console and tables stay on `Face::Mono` so columns keep aligning.
    pub fn paint_glyph_clipped_face(&mut self, renderer: &mut Renderer,
                       x: f32, y_baseline: f32, ch: char,
                       size_pt: f32, color: Color, clip_max_x: f32,
                       face: crate::Face)
    {
        let (g, uv) = renderer.glyph_uv_face(ch, size_pt, face);
        if g.width == 0 || g.height == 0 { return; }
        let c = color_to_f32(color);
        // Snap top-left of glyph quad to integer pixels so the
        // 1-to-1 texel→pixel mapping holds and the bilinear sampler
        // doesn't blend across pixel boundaries. This is the single
        // highest-impact sharpness fix for the font: without it,
        // baselines computed from fractional layout math land
        // between pixels and every glyph looks soft. (Tier 1.)
        let gx = (x + g.xmin as f32).round();
        let gy = (y_baseline - (g.height as f32 + g.ymin as f32)).round();
        let mut gw = g.width as f32;
        let gh = g.height as f32;
        // uv = [u_left, v_top, u_right, v_bottom].
        let (u0, v0, mut u1, v1) = (uv[0], uv[1], uv[2], uv[3]);
        if gx >= clip_max_x { return; }                 // fully past the edge
        if gx + gw > clip_max_x {
            let visible = clip_max_x - gx;
            if visible <= 0.0 { return; }
            u1 = u0 + (u1 - u0) * (visible / gw);
            gw = visible;
        }
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            Vertex { pos: [gx,      gy     ], uv: [u0, v0], color: c },
            Vertex { pos: [gx + gw, gy     ], uv: [u1, v0], color: c },
            Vertex { pos: [gx + gw, gy + gh], uv: [u1, v1], color: c },
            Vertex { pos: [gx,      gy + gh], uv: [u0, v1], color: c },
        ]);
        self.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    /// Paint a sub-rectangle of an existing image handle into the
    /// destination rect. `src_w` / `src_h` are pixel dimensions
    /// inside the handle (starting at its top-left). Useful when a
    /// large slot holds variable-sized content — e.g. a `GraphPanel`
    /// re-rasterises the SVG to the panel's exact pixel size inside
    /// a pre-allocated big slot, then displays only that sub-rect.
    pub fn paint_image_sub(&mut self, h: ImageHandle,
                           x: f32, y: f32, w: f32, ht: f32,
                           src_w: u32, src_h: u32,
                           tint: Color) {
        let inv = 1.0 / ATLAS_SIZE as f32;
        let u0 = h.atlas_x as f32 * inv;
        let v0 = h.atlas_y as f32 * inv;
        let u1 = (h.atlas_x + src_w.min(h.width))  as f32 * inv;
        let v1 = (h.atlas_y + src_h.min(h.height)) as f32 * inv;
        let c = color_to_f32(tint);
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            Vertex { pos: [x,       y     ], uv: [u0, v0], color: c },
            Vertex { pos: [x + w,   y     ], uv: [u1, v0], color: c },
            Vertex { pos: [x + w,   y + ht], uv: [u1, v1], color: c },
            Vertex { pos: [x,       y + ht], uv: [u0, v1], color: c },
        ]);
        self.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    /// Paint a previously-uploaded image into the destination rect.
    /// `tint` is multiplied component-wise with the texture sample —
    /// pass `Color::WHITE` for unmodified colors.
    pub fn paint_image(&mut self, h: ImageHandle, x: f32, y: f32, w: f32, ht: f32, tint: Color) {
        let inv = 1.0 / ATLAS_SIZE as f32;
        let u0 = h.atlas_x as f32 * inv;
        let v0 = h.atlas_y as f32 * inv;
        let u1 = (h.atlas_x + h.width)  as f32 * inv;
        let v1 = (h.atlas_y + h.height) as f32 * inv;
        let c = color_to_f32(tint);
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            Vertex { pos: [x,       y     ], uv: [u0, v0], color: c },
            Vertex { pos: [x + w,   y     ], uv: [u1, v0], color: c },
            Vertex { pos: [x + w,   y + ht], uv: [u1, v1], color: c },
            Vertex { pos: [x,       y + ht], uv: [u0, v1], color: c },
        ]);
        self.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    /// Paint a text string with optional anchor alignment (Start /
    /// Middle / End) and rotation around `(anchor_x, anchor_y)`.
    /// `rotation_deg` is clockwise-positive (SVG convention). The
    /// anchor point IS the rotation centre — matches how SVG
    /// `transform="rotate(deg, cx, cy)"` is emitted by r2-graphics
    /// for axis labels (the rotation centre is always the same as
    /// the text's `x`/`y` attributes there).
    pub fn paint_text_anchored(
        &mut self,
        renderer: &mut Renderer,
        anchor_x: f32, anchor_y: f32,
        text: &str,
        size_pt: f32,
        color: Color,
        anchor: TextAnchorKind,
        rotation_deg: f32,
    ) {
        let width = renderer.measure_text_width(text, size_pt);
        let start_offset = match anchor {
            TextAnchorKind::Start  => 0.0,
            TextAnchorKind::Middle => -width / 2.0,
            TextAnchorKind::End    => -width,
        };
        let rad = rotation_deg.to_radians();
        let cos_r = rad.cos();
        let sin_r = rad.sin();
        let rotated = rotation_deg.abs() > 1e-3;
        let c = color_to_f32(color);
        let mut pen_x = anchor_x + start_offset;
        for ch in text.chars() {
            let (g, uv) = renderer.glyph_uv(ch, size_pt);
            if g.width > 0 && g.height > 0 && ch != ' ' {
                // Glyph quad in unrotated screen space.
                let gx0 = (pen_x + g.xmin as f32).round();
                let gy0 = (anchor_y - (g.height as f32 + g.ymin as f32)).round();
                let gx1 = gx0 + g.width  as f32;
                let gy1 = gy0 + g.height as f32;
                let rot = |x: f32, y: f32| -> [f32; 2] {
                    if !rotated { return [x, y]; }
                    let dx = x - anchor_x;
                    let dy = y - anchor_y;
                    [anchor_x + dx * cos_r - dy * sin_r,
                     anchor_y + dx * sin_r + dy * cos_r]
                };
                let v0 = rot(gx0, gy0);
                let v1 = rot(gx1, gy0);
                let v2 = rot(gx1, gy1);
                let v3 = rot(gx0, gy1);
                let base = self.vertices.len() as u32;
                self.vertices.extend_from_slice(&[
                    Vertex { pos: v0, uv: [uv[0], uv[1]], color: c },
                    Vertex { pos: v1, uv: [uv[2], uv[1]], color: c },
                    Vertex { pos: v2, uv: [uv[2], uv[3]], color: c },
                    Vertex { pos: v3, uv: [uv[0], uv[3]], color: c },
                ]);
                self.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
            }
            pen_x += g.advance;
        }
    }

    /// Paint a whole string at (x, y_baseline). Returns the x-position
    /// after the last glyph (so caller can chain text).
    pub fn paint_text(&mut self, renderer: &mut Renderer,
                      x: f32, y_baseline: f32, text: &str,
                      size_pt: f32, color: Color) -> f32
    {
        self.paint_text_face(renderer, x, y_baseline, text, size_pt, color,
                             crate::Face::Mono)
    }

    /// Paint a string in the proportional UI face — for chrome: menu
    /// labels, window titles, buttons, dialog text.
    pub fn paint_text_ui(&mut self, renderer: &mut Renderer,
                         x: f32, y_baseline: f32, text: &str,
                         size_pt: f32, color: Color) -> f32
    {
        self.paint_text_face(renderer, x, y_baseline, text, size_pt, color,
                             crate::Face::Ui)
    }

    /// [`paint_text`] for an explicit typeface.
    pub fn paint_text_face(&mut self, renderer: &mut Renderer,
                           x: f32, y_baseline: f32, text: &str,
                           size_pt: f32, color: Color, face: crate::Face) -> f32
    {
        let mut pen_x = x;
        for ch in text.chars() {
            self.paint_glyph_clipped_face(renderer, pen_x, y_baseline, ch,
                                          size_pt, color, f32::INFINITY, face);
            let (g, _) = renderer.glyph_uv_face(ch, size_pt, face);
            pen_x += g.advance;
        }
        pen_x
    }
}

/// Whether the swapchain is an sRGB format. Set once at renderer init.
/// When true the hardware encodes linear shader output to sRGB on write,
/// so colors must be fed to the shader in LINEAR space; when false the
/// value is written through unchanged and must stay sRGB-encoded.
pub(crate) static SRGB_SURFACE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// sRGB 8-bit channel → linear f32, using the exact piecewise
/// IEC 61966-2-1 curve (not the pow(2.2) approximation — exactness is
/// this project's brand, and the toe matters for dark UI colors).
#[inline]
pub(crate) fn srgb_u8_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Theme color → shader vertex color.
///
/// Previously this divided by 255 and handed the result to the shader as
/// if it were linear. Because the surface is sRGB, the hardware then
/// re-encoded it, so every color displayed LIGHTER than authored (a
/// midtone authored at 0.5 landed near 0.73) — the root cause of the
/// washed-out chrome and low-contrast text. Converting to linear here
/// makes what is drawn match what the theme specifies.
///
/// Alpha is a coverage/opacity ratio, not a color channel: it is linear
/// by definition and must NOT be gamma-converted.
fn color_to_f32(c: Color) -> [f32; 4] {
    let a = c.3 as f32 / 255.0;
    if SRGB_SURFACE.load(std::sync::atomic::Ordering::Relaxed) {
        [srgb_u8_to_linear(c.0), srgb_u8_to_linear(c.1), srgb_u8_to_linear(c.2), a]
    } else {
        // Non-sRGB swapchain: the value is presented as written, so
        // converting would double-darken. Keep the raw ratio.
        [c.0 as f32 / 255.0, c.1 as f32 / 255.0, c.2 as f32 / 255.0, a]
    }
}
