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

use std::collections::HashMap;

use crate::theme::{Color, Theme};

// Font loading — we try a small list of system monospace fonts and
// use the first one that loads. This keeps r2-ui self-contained
// without bundling a TTF asset. Order matches typical Windows /
// macOS / Linux installations.
fn load_system_font() -> Result<fontdue::Font, String> {
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
struct Vertex {
    /// Position in pixels (top-left origin). The vertex shader
    /// converts to NDC using a screen-size uniform.
    pos: [f32; 2],
    /// UV coordinates into the atlas texture (0..1).
    uv:  [f32; 2],
    /// Tint color, already non-premultiplied straight-alpha.
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScreenUniform {
    size: [f32; 2],
    _pad: [f32; 2],
}

// ─── Glyph atlas ─────────────────────────────────────────────────────

// 2048-square RGBA atlas (16 MB). Headroom for several GraphPanel
// slots (each up to 1024×768) plus the glyph cache.
const ATLAS_SIZE: u32 = 2048;
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

struct Atlas {
    texture: wgpu::Texture,
    view:    wgpu::TextureView,
    /// (size_in_pt × 100 as u32, char) → glyph info. Quantized size so
    /// we don't make a new entry for every fractional pt.
    glyphs:  HashMap<(u32, char), GlyphInfo>,
    /// Next free position in the atlas (simple shelf packer).
    pen_x:   u32,
    pen_y:   u32,
    shelf_h: u32,
}

impl Atlas {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
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
    fn glyph(
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
    fn alloc_region(&mut self, w: u32, h: u32, rgba: &[u8], queue: &wgpu::Queue)
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

/// Text anchor for [`Frame::paint_text_anchored`]. Same semantics as
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

// ─── Renderer ────────────────────────────────────────────────────────

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device:  wgpu::Device,
    queue:   wgpu::Queue,
    config:  wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    // Pipeline state
    pipeline:        wgpu::RenderPipeline,
    #[allow(dead_code)] sampler:           wgpu::Sampler,
    screen_uniform:  wgpu::Buffer,
    #[allow(dead_code)] bind_group_layout: wgpu::BindGroupLayout,
    bind_group:      wgpu::BindGroup,

    // Dynamic vertex / index buffers (grow as needed).
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    vbuf_cap: u64,
    ibuf_cap: u64,

    atlas:   Atlas,
    pub font: fontdue::Font,
}

impl Renderer {
    pub async fn new(window: &'static winit::window::Window) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window)
            .map_err(|e| format!("create_surface: {}", e))?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or_else(|| "no GPU adapter".to_string())?;
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("r2-ui device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        ).await.map_err(|e| format!("request_device: {}", e))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:  size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── Glyph atlas + font ──
        let atlas = Atlas::new(&device, &queue);
        let font = load_system_font()?;

        // ── Pipeline ──
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("r2-ui sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let screen_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2-ui screen"),
            size: std::mem::size_of::<ScreenUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("r2-ui bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = make_bind_group(&device, &bind_group_layout, &screen_uniform, &atlas.view, &sampler);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("r2-ui shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("r2-ui pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("r2-ui pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vbuf_cap = 65536u64;
        let ibuf_cap = 65536u64;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2-ui vbuf"),
            size: vbuf_cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2-ui ibuf"),
            size: ibuf_cap,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            surface, device, queue, config, size,
            pipeline, sampler, screen_uniform,
            bind_group_layout, bind_group,
            vbuf, ibuf, vbuf_cap, ibuf_cap,
            atlas, font,
        })
    }

    pub fn resize(&mut self, new: winit::dpi::PhysicalSize<u32>) {
        if new.width == 0 || new.height == 0 { return; }
        self.size = new;
        self.config.width = new.width;
        self.config.height = new.height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn begin_frame(&self) -> Frame {
        Frame {
            vertices: Vec::with_capacity(2048),
            indices:  Vec::with_capacity(3072),
        }
    }

    pub fn submit(&mut self, frame: Frame, theme: &Theme) -> Result<(), wgpu::SurfaceError> {
        // Update screen-size uniform.
        let su = ScreenUniform {
            size: [self.size.width as f32, self.size.height as f32],
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.screen_uniform, 0, bytemuck::bytes_of(&su));

        // Re-make bind group if atlas grew (atlas view doesn't change,
        // but pen position does — view itself is stable).
        let vbytes = bytemuck::cast_slice::<Vertex, u8>(&frame.vertices);
        let ibytes = bytemuck::cast_slice::<u32, u8>(&frame.indices);
        if vbytes.len() as u64 > self.vbuf_cap {
            self.vbuf_cap = (vbytes.len() as u64).next_power_of_two();
            self.vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("r2-ui vbuf"),
                size: self.vbuf_cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if ibytes.len() as u64 > self.ibuf_cap {
            self.ibuf_cap = (ibytes.len() as u64).next_power_of_two();
            self.ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("r2-ui ibuf"),
                size: self.ibuf_cap,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !vbytes.is_empty() { self.queue.write_buffer(&self.vbuf, 0, vbytes); }
        if !ibytes.is_empty() { self.queue.write_buffer(&self.ibuf, 0, ibytes); }

        let surface_frame = self.surface.get_current_texture()?;
        let view = surface_frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("r2-ui frame"),
        });

        let bg = theme.mdi_background;
        let clear = wgpu::Color {
            r: bg.0 as f64 / 255.0,
            g: bg.1 as f64 / 255.0,
            b: bg.2 as f64 / 255.0,
            a: bg.3 as f64 / 255.0,
        };
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("r2-ui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if !frame.indices.is_empty() {
                rp.set_pipeline(&self.pipeline);
                rp.set_bind_group(0, &self.bind_group, &[]);
                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..frame.indices.len() as u32, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(enc.finish()));
        surface_frame.present();
        Ok(())
    }

    /// Convenience: clear-only frame (Week 2 behavior).
    pub fn render(&mut self, theme: &Theme) -> Result<(), wgpu::SurfaceError> {
        let frame = self.begin_frame();
        self.submit(frame, theme)
    }

    /// Upload an RGBA image into the atlas. Returns a handle that can
    /// be re-drawn each frame via [`Frame::paint_image`]; the upload
    /// itself happens once. Returns `None` if the atlas can't fit the
    /// requested region.
    pub fn upload_image(&mut self, w: u32, h: u32, rgba: &[u8]) -> Option<ImageHandle> {
        let (ax, ay, aw, ah) = self.atlas.alloc_region(w, h, rgba, &self.queue)?;
        Some(ImageHandle { atlas_x: ax, atlas_y: ay, width: aw, height: ah })
    }

    /// Overwrite a SUB-RECTANGLE of an existing image handle. The
    /// `(offset_x, offset_y, w, h)` rect must lie inside the handle's
    /// full dimensions, and `rgba.len()` must equal `w * h * 4`.
    /// Used by widgets that want to write a smaller pixmap into a
    /// pre-allocated big slot — e.g. `GraphPanel` rasterising the
    /// SVG at the displayed panel pixel size into a fixed
    /// 1024×768 slot so the GPU draws 1:1 and text never resamples.
    pub fn replace_image_subregion(&self, handle: ImageHandle,
                                   offset_x: u32, offset_y: u32,
                                   w: u32, h: u32, rgba: &[u8]) -> bool {
        if w == 0 || h == 0 { return false; }
        if offset_x + w > handle.width || offset_y + h > handle.height { return false; }
        if rgba.len() != (w as usize) * (h as usize) * 4 { return false; }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.atlas.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: handle.atlas_x + offset_x,
                    y: handle.atlas_y + offset_y,
                    z: 0
                },
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
        true
    }

    /// Overwrite the pixels of an existing image handle in place. The
    /// supplied `rgba` must be exactly `handle.width * handle.height
    /// * 4` bytes — same dimensions as the original allocation. Used
    /// by widgets like `GraphPanel` that re-rasterize on resize but
    /// want to reuse the same atlas slot (so the atlas doesn't fill
    /// up with discarded plot images).
    pub fn replace_image_pixels(&self, handle: ImageHandle, rgba: &[u8]) -> bool {
        let expected = (handle.width as usize) * (handle.height as usize) * 4;
        if rgba.len() != expected { return false; }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.atlas.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: handle.atlas_x, y: handle.atlas_y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(handle.width * 4),
                rows_per_image: Some(handle.height),
            },
            wgpu::Extent3d { width: handle.width, height: handle.height, depth_or_array_layers: 1 },
        );
        true
    }

    /// Sum-of-advances for a string at the given font size, in
    /// pixels. Used by text painters that need to centre or
    /// right-align text (anchor = Middle / End).
    pub fn measure_text_width(&mut self, text: &str, size_pt: f32) -> f32 {
        let mut w = 0.0f32;
        for ch in text.chars() {
            let (g, _) = self.glyph_uv(ch, size_pt);
            w += g.advance;
        }
        w
    }

    /// Cell metrics for a monospace grid: returns `(cell_width,
    /// line_height)` in pixels for the given size. Uses the advance of
    /// `'M'` for cell width and a 1.25× height factor for leading.
    pub fn cell_metrics(&mut self, size_pt: f32) -> (f32, f32) {
        let (g, _) = self.glyph_uv('M', size_pt);
        let cw = if g.advance > 0.0 { g.advance } else { size_pt * 0.6 };
        let lh = (size_pt * 1.25).ceil();
        (cw, lh)
    }

    /// Glyph access for widgets — returns metrics + atlas uv.
    /// Public so the `Frame` builder methods can use it; widgets
    /// shouldn't typically call this directly.
    pub fn glyph_uv(&mut self, ch: char, size_pt: f32) -> (GlyphInfo, [f32; 4]) {
        let g = self.atlas.glyph(&self.font, ch, size_pt, &self.queue);
        let inv = 1.0 / ATLAS_SIZE as f32;
        let uv = [
            g.atlas_x as f32 * inv,
            g.atlas_y as f32 * inv,
            (g.atlas_x + g.width)  as f32 * inv,
            (g.atlas_y + g.height) as f32 * inv,
        ];
        (g, uv)
    }
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    screen: &wgpu::Buffer,
    view:   &wgpu::TextureView,
    sampler:&wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("r2-ui bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: screen.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

// ─── Frame builder ───────────────────────────────────────────────────

pub struct Frame {
    vertices: Vec<Vertex>,
    indices:  Vec<u32>,
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
        let (g, uv) = renderer.glyph_uv(ch, size_pt);
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
        let gw = g.width as f32;
        let gh = g.height as f32;
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            Vertex { pos: [gx,      gy     ], uv: [uv[0], uv[1]], color: c },
            Vertex { pos: [gx + gw, gy     ], uv: [uv[2], uv[1]], color: c },
            Vertex { pos: [gx + gw, gy + gh], uv: [uv[2], uv[3]], color: c },
            Vertex { pos: [gx,      gy + gh], uv: [uv[0], uv[3]], color: c },
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
        let mut pen_x = x;
        for ch in text.chars() {
            self.paint_glyph(renderer, pen_x, y_baseline, ch, size_pt, color);
            let (g, _) = renderer.glyph_uv(ch, size_pt);
            pen_x += g.advance;
        }
        pen_x
    }
}

fn color_to_f32(c: Color) -> [f32; 4] {
    [
        c.0 as f32 / 255.0,
        c.1 as f32 / 255.0,
        c.2 as f32 / 255.0,
        c.3 as f32 / 255.0,
    ]
}

// ─── WGSL shader ─────────────────────────────────────────────────────
//
// Single quad pipeline. The fragment samples the atlas (grayscale)
// and multiplies by the tint color's alpha. For solid rects the
// sampled texel is 1.0 so the tint passes through unchanged; for
// glyphs the texel is the rasterized coverage so antialiasing works.

const WGSL: &str = r#"
struct Screen { size: vec2<f32>, _pad: vec2<f32> };
@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var samp:  sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1) uv:  vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var o: VsOut;
    let ndc_x =  (pos.x / screen.size.x) * 2.0 - 1.0;
    let ndc_y = -((pos.y / screen.size.y) * 2.0 - 1.0);
    o.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    o.uv = uv;
    o.color = color;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let s = textureSample(atlas, samp, in.uv);
    return s * in.color;
}
"#;
