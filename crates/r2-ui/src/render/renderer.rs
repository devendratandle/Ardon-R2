//! The wgpu `Renderer` — pipeline, buffers, atlas upload, submit.

use crate::theme::Theme;
use super::atlas::*;
use super::frame::Frame;
use super::ImageHandle;

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
    /// The monospace face — console, tables, numbers.
    pub font: fontdue::Font,
    /// The proportional UI face — menus, window titles, labels, buttons.
    /// `None` when no system UI font parsed; every lookup then falls back
    /// to the mono face, so a missing font never stops the app.
    pub ui_font: Option<fontdue::Font>,
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
        // Tell the color converter which space the swapchain expects. On an
        // sRGB surface the hardware encodes linear output, so theme colors
        // must be converted sRGB→linear before the shader; on a non-sRGB
        // fallback they must be passed through raw or they double-darken.
        super::frame::SRGB_SURFACE.store(format.is_srgb(), std::sync::atomic::Ordering::Relaxed);
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
        let ui_font = load_ui_font();

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
                    // VERTEX reads `size` for the NDC transform; FRAGMENT
                    // reads `text_gamma` for the glyph AA correction.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            atlas, font, ui_font,
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
            // Per-theme glyph AA correction (see ScreenUniform::text_gamma).
            text_gamma: theme.text_gamma,
            _pad: 0.0,
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

        // The clear value goes through the same sRGB→linear conversion as
        // every drawn color, otherwise the workspace background would be
        // the one surface not matching its theme entry.
        let bg = theme.mdi_background;
        let srgb = super::frame::SRGB_SURFACE.load(std::sync::atomic::Ordering::Relaxed);
        let conv = |v: u8| -> f64 {
            if srgb { super::frame::srgb_u8_to_linear(v) as f64 } else { v as f64 / 255.0 }
        };
        let clear = wgpu::Color {
            r: conv(bg.0),
            g: conv(bg.1),
            b: conv(bg.2),
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
        self.glyph_uv_face(ch, size_pt, Face::Mono)
    }

    /// Width of a string in the proportional UI face — chrome painters
    /// need this to centre menu labels and window titles.
    pub fn measure_text_width_ui(&mut self, text: &str, size_pt: f32) -> f32 {
        text.chars().fold(0.0, |w, ch| w + self.glyph_uv_face(ch, size_pt, Face::Ui).0.advance)
    }

    /// Glyph access for a specific typeface. `Face::Ui` falls back to the
    /// mono face when no system UI font was found, so callers never have
    /// to handle a missing font.
    pub fn glyph_uv_face(&mut self, ch: char, size_pt: f32, face: Face)
        -> (GlyphInfo, [f32; 4])
    {
        let (font, face) = match face {
            Face::Ui => match self.ui_font.as_ref() {
                Some(f) => (f, Face::Ui),
                None    => (&self.font, Face::Mono), // key by what we RASTERIZED
            },
            Face::Mono => (&self.font, Face::Mono),
        };
        let g = self.atlas.glyph(font, ch, size_pt, face, &self.queue);
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


// ─── WGSL shader ─────────────────────────────────────────────────────
//
// Single quad pipeline. The fragment samples the atlas (grayscale)
// and multiplies by the tint color's alpha. For solid rects the
// sampled texel is 1.0 so the tint passes through unchanged; for
// glyphs the texel is the rasterized coverage so antialiasing works.

const WGSL: &str = r#"
struct Screen { size: vec2<f32>, text_gamma: f32, _pad: f32 };
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
    var s = textureSample(atlas, samp, in.uv);
    // Glyph quads sample white-with-coverage-in-alpha; image tiles carry
    // real RGB. Applying the exponent to alpha only, and only where the
    // sample is pure white, corrects glyph stem weight for the theme's
    // polarity while leaving plot tiles untouched. The solid-fill pixel
    // is white with alpha 1.0, and pow(1, g) == 1, so fills are unaffected.
    if (s.r == 1.0 && s.g == 1.0 && s.b == 1.0) {
        s.a = pow(s.a, screen.text_gamma);
    }
    return s * in.color;
}
"#;
