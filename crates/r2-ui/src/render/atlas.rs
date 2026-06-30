//! Font loading, GPU vertex format, and the glyph atlas.

use std::collections::HashMap;


// Font loading — we try a small list of system monospace fonts and
// use the first one that loads. This keeps r2-ui self-contained
// without bundling a TTF asset. Order matches typical Windows /
// macOS / Linux installations.
pub(crate) fn load_system_font() -> Result<fontdue::Font, String> {
    const CANDIDATES: &[&str] = &[
        // Windows
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        // macOS
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
        // Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()) {
                return Ok(font);
            }
        }
    }
    Err("no system monospace font found (tried Consolas, Courier New, Menlo, DejaVu, Liberation)".into())
}

// ─── Vertex format ───────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    /// Position in pixels (top-left origin). The vertex shader
    /// converts to NDC using a screen-size uniform.
    pub(crate) pos: [f32; 2],
    /// UV coordinates into the atlas texture (0..1).
    pub(crate) uv:  [f32; 2],
    /// Tint color, already non-premultiplied straight-alpha.
    pub(crate) color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScreenUniform {
    pub(crate) size: [f32; 2],
    pub(crate) _pad: [f32; 2],
}

// ─── Glyph atlas ─────────────────────────────────────────────────────

// 2048-square RGBA atlas (16 MB). Headroom for several GraphPanel
// slots (each up to 1024×768) plus the glyph cache.
pub(crate) const ATLAS_SIZE: u32 = 4096;
const ATLAS_PAD:  u32 = 1;

/// Atlas + layout metrics for one rasterized glyph. Returned by
/// [`Renderer::glyph_uv`] so widgets can compute their own pen advances.
#[derive(Debug, Clone, Copy)]
pub struct GlyphInfo {
    /// Pixel rect inside the atlas (top-left, size).
    pub atlas_x: u32, pub atlas_y: u32,
    pub width: u32, pub height: u32,
    /// Layout metrics from fontdue.
    pub xmin: i32, pub ymin: i32,
    pub advance: f32,
}

pub(crate) struct Atlas {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view:    wgpu::TextureView,
    /// (size_in_pt × 100 as u32, char) → glyph info. Quantized size so
    /// we don't make a new entry for every fractional pt.
    glyphs:  HashMap<(u32, char), GlyphInfo>,
    /// Next free position in the atlas (simple shelf packer).
    pen_x:   u32,
    pen_y:   u32,
    shelf_h: u32,
}

impl Atlas {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("r2-ui atlas"),
            size:  wgpu::Extent3d { width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // RGBA8 so the same atlas + pipeline can carry both
            // grayscale glyphs (stored as white-with-alpha-coverage)
            // AND full-color image tiles (PlotPanel SVG output).
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Seed the (0,0) pixel as opaque so it can be sampled for
        // solid-color rectangles. White with full alpha → tint passes
        // through unchanged.
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        Self {
            texture, view, glyphs: HashMap::new(),
            pen_x: 1 + ATLAS_PAD,  // leave the (0,0) pixel for solid fills
            pen_y: 0,
            shelf_h: 0,
        }
    }

    /// Get (or rasterize on-demand) a glyph at the given size.
    pub(crate) fn glyph(
        &mut self,
        font: &fontdue::Font,
        ch: char,
        size_pt: f32,
        queue: &wgpu::Queue,
    ) -> GlyphInfo {
        let key = ((size_pt * 100.0) as u32, ch);
        if let Some(g) = self.glyphs.get(&key) {
            return *g;
        }
        // Rasterize.
        let (metrics, bitmap) = font.rasterize(ch, size_pt);
        let w = metrics.width  as u32;
        let h = metrics.height as u32;

        // Shelf-pack: if it doesn't fit in current shelf, start a new shelf.
        if self.pen_x + w + ATLAS_PAD > ATLAS_SIZE {
            self.pen_x  = ATLAS_PAD;
            self.pen_y += self.shelf_h + ATLAS_PAD;
            self.shelf_h = 0;
        }
        // Out of room: punt (return zero-sized glyph; caller paints nothing).
        if self.pen_y + h > ATLAS_SIZE {
            let g = GlyphInfo {
                atlas_x: 0, atlas_y: 0, width: 0, height: 0,
                xmin: metrics.xmin, ymin: metrics.ymin,
                advance: metrics.advance_width,
            };
            self.glyphs.insert(key, g);
            return g;
        }

        let ax = self.pen_x;
        let ay = self.pen_y;

        if w > 0 && h > 0 {
            // Expand grayscale coverage → RGBA: white-with-alpha-coverage.
            // (R=255, G=255, B=255, A=cov) means `sample * tint` in the
            // fragment shader produces `tint.rgb` with `tint.a * cov`.
            let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
            for &c in &bitmap {
                // Sharpen the anti-aliased coverage ramp ("stem darkening").
                // fontdue emits a soft grey edge fringe; written straight to
                // alpha it looks fuzzy at the small sizes the console/chrome
                // use on a light background. A mild gamma < 1 pushes the
                // partial-coverage edge pixels toward opaque, so stems read
                // crisp and high-contrast — matching the sharpness of the
                // resvg-rendered graphics-device text. 0.72 darkens without
                // the blocky over-thickening a more aggressive curve causes.
                let cov = c as f32 / 255.0;
                let a = (cov.powf(0.72) * 255.0 + 0.5) as u8;
                rgba.extend_from_slice(&[255, 255, 255, a]);
            }
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: ax, y: ay, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        let g = GlyphInfo {
            atlas_x: ax, atlas_y: ay,
            width: w, height: h,
            xmin: metrics.xmin, ymin: metrics.ymin,
            advance: metrics.advance_width,
        };
        self.glyphs.insert(key, g);
        self.pen_x += w + ATLAS_PAD;
        self.shelf_h = self.shelf_h.max(h);
        g
    }

    /// Allocate a rectangular region in the atlas and upload arbitrary
    /// RGBA pixel data into it. Returns the pixel rect on success.
    /// `rgba.len()` must equal `w * h * 4`.
    pub(crate) fn alloc_region(&mut self, w: u32, h: u32, rgba: &[u8], queue: &wgpu::Queue)
        -> Option<(u32, u32, u32, u32)>
    {
        if w == 0 || h == 0 || rgba.len() != (w as usize) * (h as usize) * 4 {
            return None;
        }
        // Always start a fresh shelf for image-sized allocations so we
        // don't fragment glyph shelves with one tall tile.
        if self.shelf_h > 0 {
            self.pen_x  = ATLAS_PAD;
            self.pen_y += self.shelf_h + ATLAS_PAD;
            self.shelf_h = 0;
        }
        if self.pen_x + w + ATLAS_PAD > ATLAS_SIZE { return None; }
        if self.pen_y + h > ATLAS_SIZE { return None; }
        let ax = self.pen_x;
        let ay = self.pen_y;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: ax, y: ay, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.pen_x  = ATLAS_PAD;
        self.pen_y += h + ATLAS_PAD;
        self.shelf_h = 0;
        Some((ax, ay, w, h))
    }
}

