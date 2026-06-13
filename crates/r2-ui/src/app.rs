//! App entry point and event loop.
//!
//! Opens a winit window, drives the wgpu Renderer, accumulates one
//! frame's worth of input events, and dispatches both into a user
//! closure that gets to mutate widget state and paint into the frame.

use crate::event::{from_winit, Clipboard, InputEvent, Mods, MousePos};
use crate::layout::LayoutBuilder;
use crate::menu::MenuBuilder;
use crate::render::{Frame, Renderer};
use crate::theme::Theme;

/// Per-frame context handed to the user closure. Holds the input
/// events seen since the previous frame plus a clipboard handle.
pub struct FrameCtx<'a> {
    pub events: &'a [InputEvent],
    pub clipboard: &'a mut Clipboard,
}

/// Per-frame painter signature. Called once per `RedrawRequested`
/// with the latest input events; whatever the closure draws into the
/// frame gets composited over the theme background.
pub type FrameFn = Box<dyn FnMut(&mut FrameCtx, &mut Renderer, &mut Frame, &Theme)>;

/// Top-level handle returned by `R2Ui::app("title")`. Used as a
/// builder; terminate the chain with `.run()`.
pub struct R2Ui {
    title: String,
    theme: Theme,
    font_family: String,
    font_size: f32,
    icon_bytes: Option<&'static [u8]>,
    initial_size: (u32, u32),
    #[allow(dead_code)]
    layout: Option<LayoutBuilder>,
    #[allow(dead_code)]
    menu: Option<MenuBuilder>,
    frame_fn: Option<FrameFn>,
}

impl R2Ui {
    /// Begin configuring an app with the given window title.
    pub fn app(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            theme: Theme::khaki(),
            font_family: "Consolas".into(),
            font_size: 14.0,
            icon_bytes: None,
            initial_size: (1100, 700),
            layout: None,
            menu: None,
            frame_fn: None,
        }
    }

    /// Install a per-frame closure that gets events + paint surface.
    /// This is the main hook widgets attach to.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where F: FnMut(&mut FrameCtx, &mut Renderer, &mut Frame, &Theme) + 'static
    {
        self.frame_fn = Some(Box::new(f));
        self
    }

    /// Back-compat shortcut for callers that only want to paint —
    /// they get the renderer + frame + theme but no events.
    pub fn on_paint<F>(self, mut f: F) -> Self
    where F: FnMut(&mut Renderer, &mut Frame, &Theme) + 'static
    {
        self.on_frame(move |_ctx, r, fr, th| f(r, fr, th))
    }

    pub fn theme(mut self, t: Theme) -> Self { self.theme = t; self }

    pub fn font_family(mut self, family: impl Into<String>, size: f32) -> Self {
        self.font_family = family.into();
        self.font_size = size;
        self
    }

    pub fn icon_png(mut self, bytes: &'static [u8]) -> Self {
        self.icon_bytes = Some(bytes);
        self
    }

    pub fn initial_size(mut self, width: u32, height: u32) -> Self {
        self.initial_size = (width, height);
        self
    }

    pub fn mdi<F: FnOnce(&mut LayoutBuilder)>(mut self, build: F) -> Self {
        let mut lb = LayoutBuilder::new();
        build(&mut lb);
        self.layout = Some(lb);
        self
    }

    pub fn menu<F: FnOnce(&mut MenuBuilder)>(mut self, build: F) -> Self {
        let mut mb = MenuBuilder::new();
        build(&mut mb);
        self.menu = Some(mb);
        self
    }

    /// Open the OS window and drive the event loop until the user
    /// closes it.
    pub fn run(self) -> Result<(), String> {
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::{ControlFlow, EventLoop};
        use winit::window::WindowBuilder;

        let event_loop = EventLoop::new()
            .map_err(|e| format!("EventLoop::new: {}", e))?;
        // Poll, not Wait — we redraw on a fixed cadence so input + cursor
        // animation feel responsive. Real apps can drop to Wait via a
        // dedicated knob later if power use becomes a concern.
        event_loop.set_control_flow(ControlFlow::Poll);

        // Resolution-adaptive initial size. `initial_size` is designed
        // against a 1080p/100% reference; on panels where the OS scaling
        // doesn't already enlarge logical pixels (e.g. 1440p/4K at 100%),
        // grow the window by the extra resolution factor so it occupies
        // the same visual fraction of the screen from 720p to 4K. Clamp
        // to 92% of the monitor so it always fits.
        let (init_w, init_h) = {
            let mut w = self.initial_size.0 as f64;
            let mut h = self.initial_size.1 as f64;
            if let Some(m) = event_loop.primary_monitor() {
                let os = m.scale_factor().max(0.5);
                let res = (m.size().height as f64 / 1080.0).max(1.0);
                let extra = (res / os).max(1.0); // enlargement the OS is NOT already doing
                w *= extra;
                h *= extra;
                let mon_w = m.size().width as f64 / os;
                let mon_h = m.size().height as f64 / os;
                w = w.min(mon_w * 0.92);
                h = h.min(mon_h * 0.92);
            }
            (w, h)
        };
        let window = WindowBuilder::new()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(init_w, init_h))
            .with_min_inner_size(winit::dpi::LogicalSize::new(600.0, 360.0))
            .build(&event_loop)
            .map_err(|e| format!("WindowBuilder::build: {}", e))?;

        // Icon: winit wants square RGBA. Many logo PNGs are wider than
        // they are tall; letterbox onto a transparent square so the OS
        // taskbar / Alt-Tab shows the whole image instead of a chunk.
        if let Some(bytes) = self.icon_bytes {
            if let Ok(img) = image::load_from_memory(bytes) {
                let src = img.into_rgba8();
                let (sw, sh) = (src.width(), src.height());
                // Pick a sensible square size capped at 128 px.
                let side = sw.max(sh).min(128);
                // Resize preserving aspect ratio so the longer edge = `side`.
                let scale = side as f32 / sw.max(sh) as f32;
                let new_w = (sw as f32 * scale).round() as u32;
                let new_h = (sh as f32 * scale).round() as u32;
                let resized = image::imageops::resize(
                    &src, new_w, new_h, image::imageops::FilterType::Triangle);
                // Letterbox onto a `side × side` transparent canvas.
                let mut canvas = image::RgbaImage::from_pixel(side, side,
                    image::Rgba([0, 0, 0, 0]));
                let ox = (side - new_w) / 2;
                let oy = (side - new_h) / 2;
                image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);
                if let Ok(icon) = winit::window::Icon::from_rgba(
                    canvas.into_raw(), side, side) {
                    window.set_window_icon(Some(icon));
                }
            }
        }

        let window_ref: &'static winit::window::Window =
            Box::leak(Box::new(window));

        let mut renderer = pollster::block_on(Renderer::new(window_ref))?;
        // Pick up the OS scaling factor so HiDPI / 200% displays look
        // the same proportionally as 100% — Theme carries the multiplier
        // and L4 widgets read it via theme.fs() / theme.lh() / theme.px().
        let mut theme = self.theme;
        // Resolution-AND-DPI-aware UI scale, adapting 720p → 4K. Two
        // signals: the OS scaling setting (`scale_factor`, e.g. 1.5 at
        // 150%) and the panel's physical height (a 1440p/4K screen at 100%
        // OS scaling still needs enlarging — the OS isn't asking, but the
        // pixels are tiny). Reference 1080p = 1.0×. Take whichever asks for
        // MORE enlargement; never shrink below 1.0 (720p keeps the default).
        let scale_factor = window_ref.scale_factor() as f32;
        let res_scale = window_ref
            .current_monitor()
            .map(|m| m.size().height as f32 / 1080.0)
            .unwrap_or(1.0);
        theme.set_dpi(scale_factor.max(res_scale).max(1.0));
        let mut frame_fn = self.frame_fn;
        let mut clipboard = Clipboard::new();

        // Per-frame event accumulator + latched mouse / modifier state.
        let mut pending: Vec<InputEvent> = Vec::with_capacity(32);
        let mut mouse_pos = MousePos { x: 0.0, y: 0.0 };
        let mut mods = Mods::default();

        event_loop.run(move |event, target| match event {
            Event::WindowEvent { event, .. } => {
                // Latch mouse / modifiers so MouseDown / Key events
                // carry the right snapshot.
                if let WindowEvent::CursorMoved { position, .. } = &event {
                    // PHYSICAL pixels — the renderer paints in physical
                    // coordinates (its NDC uniform is the surface size),
                    // so hit-testing must use the same space. The old
                    // to_logical conversion misaligned every click when
                    // OS scaling ≠ 100%. All visual scaling is carried by
                    // theme.dpi instead.
                    mouse_pos = MousePos { x: position.x as f32, y: position.y as f32 };
                }
                if let WindowEvent::ModifiersChanged(new) = &event {
                    let s = new.state();
                    mods = Mods {
                        shift: s.shift_key(),
                        ctrl:  s.control_key(),
                        alt:   s.alt_key(),
                    };
                }

                // Convert + buffer. Scale 1.0 = keep PHYSICAL pixels
                // (matches the renderer's coordinate space; see above).
                pending.extend(from_winit(&event, mouse_pos, mods, 1.0));

                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new) => renderer.resize(new),
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        // Re-derive the effective scale (e.g. window moved to
                        // a different-DPI monitor): same rule as startup —
                        // max(OS scaling, panel-resolution factor, 1.0).
                        let res_scale = window_ref
                            .current_monitor()
                            .map(|m| m.size().height as f32 / 1080.0)
                            .unwrap_or(1.0);
                        theme.set_dpi((scale_factor as f32).max(res_scale).max(1.0));
                    }
                    WindowEvent::RedrawRequested => {
                        let mut frame = renderer.begin_frame();
                        if let Some(p) = frame_fn.as_mut() {
                            let mut ctx = FrameCtx {
                                events: &pending,
                                clipboard: &mut clipboard,
                            };
                            p(&mut ctx, &mut renderer, &mut frame, &theme);
                        }
                        pending.clear();
                        if let Err(e) = renderer.submit(frame, &theme) {
                            eprintln!("[r2-ui] render error: {:?}", e);
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window_ref.request_redraw();
            }
            _ => {}
        }).map_err(|e| format!("event_loop.run: {}", e))?;

        Ok(())
    }
}

/// Trait that hosts implement when they want to drive R2-UI manually
/// (without the declarative `R2Ui::app` builder). Useful for embedded
/// uses where R2-UI is one panel in a larger app.
pub trait R2UiApp {
    fn frame(&mut self);
}
