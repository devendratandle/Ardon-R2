//! Font loading, GPU vertex format, and the glyph atlas.

use std::collections::HashMap;


// Font loading — platform-appropriate monospace candidates, first one that
// parses wins. Keeps r2-ui self-contained (no bundled TTF asset). Each OS
// leads with its *native* console font so the GUI looks at home on
// Windows / macOS / Linux; on Linux (where paths vary by distro) a bounded
// directory scan backstops the fixed list so exotic distros still get a
// monospace face rather than a startup error.
pub(crate) fn load_system_font() -> Result<fontdue::Font, String> {
    const CANDIDATES: &[&str] = &[
        // Windows — Consolas (the ClearType-era console classic), then the
        // modern Cascadia family (Windows Terminal / Win11), then the old guard.
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/CascadiaCode.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        // macOS — Menlo (Terminal.app default), Monaco (the timeless one).
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
        // Linux — the faces shipped by the major distro families.
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()) {
                return Ok(font);
            }
        }
    }
    // Last resort on Unix: scan the standard font roots for any *Mono*.ttf.
    #[cfg(not(target_os = "windows"))]
    {
        let mut roots: Vec<std::path::PathBuf> = vec![
            "/usr/share/fonts".into(),
            "/usr/local/share/fonts".into(),
        ];
        if let Some(home) = std::env::var_os("HOME") {
            roots.push(std::path::Path::new(&home).join(".local/share/fonts"));
        }
        for root in roots {
            if let Some(font) = scan_for_mono(&root, 0) { return Ok(font); }
        }
    }
    Err("no system monospace font found (tried Consolas/Cascadia, Menlo/Monaco, \
         DejaVu/Liberation/Noto/Ubuntu/FreeMono + a /usr/share/fonts scan)".into())
}

/// Bounded recursive search (≤3 levels) for a parseable `*mono*.ttf`.
#[cfg(not(target_os = "windows"))]
fn scan_for_mono(dir: &std::path::Path, depth: u8) -> Option<fontdue::Font> {
    if depth > 3 { return None; }
    let entries = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() { subdirs.push(p); continue; }
        let name = p.file_name()?.to_string_lossy().to_lowercase();
        if name.contains("mono") && name.ends_with(".ttf") {
            if let Ok(bytes) = std::fs::read(&p) {
                if let Ok(f) = fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()) {
                    return Some(f);
                }
            }
        }
    }
    for d in subdirs {
        if let Some(f) = scan_for_mono(&d, depth + 1) { return Some(f); }
    }
    None
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
    /// Glyph coverage exponent, from the active theme. <1 thickens stems
    /// (dark text on a light background, which otherwise reads thin);
    /// >1 thins them (light on dark, which otherwise blooms); 1.0 is
    /// neutral. Applied in the fragment shader to glyph alpha only.
    pub(crate) text_gamma: f32,
    pub(crate) _pad: f32,
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
            //
            // sRGB-tagged: image tiles are 8-bit sRGB pixels, so the
            // sampler must decode them to linear to match the linear
            // vertex colors (see frame::color_to_f32). Glyphs are
            // unaffected — their RGB is 255 (1.0 in both spaces) and the
            // coverage lives in alpha, which sRGB never touches.
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
            // Store fontdue's RAW coverage. The old code baked a
            // `coverage^0.58` "stem darkening" curve in here, which was a
            // counter-hack for the sRGB bug fixed in stage A — and being
            // baked per-glyph it was identical for every theme, so it would
            // make light-on-dark text bloom. The perceptual correction now
            // lives in the fragment shader as a per-theme `text_gamma`
            // uniform, where it can follow the theme's polarity.
            let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
            for &c in &bitmap {
                rgba.extend_from_slice(&[255, 255, 255, c]);
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

