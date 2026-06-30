//! Render layer — wgpu pipeline with rect + glyph primitives.
//!
//! Phase 2 Week 3 milestone: a single textured-quad pipeline that can
//! paint either solid rectangles or rasterized glyphs (from fontdue).
//! Glyphs are cached in a dynamic atlas texture; each cell stores
//! its uv-rect once rasterized. Solid rects reuse one pixel of the
//! atlas (kept opaque-white) so the same pipeline handles both.
//!
//! Public API for widgets:
//!   renderer.begin_frame() → Frame
//!   frame.paint_rect(rect, color)
//!   frame.paint_glyph(x, y, ch, color, size)
//!   renderer.submit(frame)


/// SVG's `text-anchor` attribute: where the anchor coordinate sits
/// relative to the rendered text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchorKind { Start, Middle, End }

/// Handle to an RGBA image that's been uploaded into the renderer's
/// atlas. Cheap to copy; paint with [`Frame::paint_image`].
#[derive(Debug, Clone, Copy)]
pub struct ImageHandle {
    pub atlas_x: u32, pub atlas_y: u32,
    pub width:   u32, pub height:  u32,
}

mod atlas;
mod renderer;
mod frame;

pub use atlas::GlyphInfo;
pub use renderer::Renderer;
pub use frame::Frame;
