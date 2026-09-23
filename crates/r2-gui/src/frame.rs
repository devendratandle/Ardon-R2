//! One frame, in order: sync graphics devices → menus and commands →
//! window chrome → console input → paint every window bottom to top →
//! popups → modal dialogs. The order is load-bearing: anything painted
//! later floats above what came before, and a modal owns the input.

use r2_graphics::device::{device_full_svg, drain_events, DeviceEvent};
use r2_ui::{
    menu_bar_height, Color, Frame, FrameCtx, GraphPanel, InputEvent, MenuBarState,
    Rect, Renderer, Theme, WindowId,
};

use crate::Gui;

impl Gui {
    pub(crate) fn frame(&mut self, ctx: &mut FrameCtx, renderer: &mut Renderer,
                        frame: &mut Frame, theme: &Theme) {
        self.frame_counter += 1;
        self.upload_logo(renderer);
        self.sync_devices(renderer, theme);
        // "The graphics window the user is currently working with" — any
        // per-window menu / save-dialog / paint dispatcher uses this.
        let graphics_id = self.graphics_window();
        let win_w = renderer.size.width  as f32;
        let win_h = renderer.size.height as f32;

        self.open_quit_dialog_on_request(ctx.close_requested);

        // While any modal is open it OWNS input: feed the widgets
        // underneath an empty event slice so clicks/keys don't leak
        // through to the console, menus, or window chrome. The dialog
        // itself (handled + painted last) reads the real `ctx.events`.
        let modal_open = self.quit_dialog.is_open() || self.settings_dialog.is_open();
        let ui_events: &[InputEvent] = if modal_open { &[] } else { ctx.events };

        // ── Workspace
        let menu_h = menu_bar_height(theme);
        let menu_rect = Rect { x: 0.0, y: 0.0, w: win_w, h: menu_h };
        let workspace = Rect { x: 0.0, y: menu_h, w: win_w, h: win_h - menu_h };
        self.mdi.set_workspace(workspace);
        self.first_layout(workspace);

        // ── The menu bar belongs to the active window. With no open
        //    device graphics_id is None → always the console menu. The
        //    OTHER menu's open popup is closed so it doesn't linger when
        //    focus switches.
        let graphics_menu = graphics_id.is_some()
            && self.mdi.z_order().last() == graphics_id;
        if graphics_menu {
            self.menu_console.open = None;
        } else {
            self.menu_graphics.open = None;
        }

        // Snapshot whether any context menu was already open BEFORE this
        // frame's events: a left-click that just landed on a popup item
        // must not also reach the grid (which would collapse the user's
        // selection before Copy can read it).
        let ctx_was_open = self.ctx_console.is_open() || self.ctx_graphics.is_open();
        if let Some(action) = self.menu_events(ui_events, menu_rect, graphics_menu,
                                               graphics_id, renderer, theme) {
            self.dispatch(&action, ctx, renderer, theme, graphics_id);
        }

        // ── MDI chrome events (drag / resize / close / max)
        self.mdi.handle_events(ui_events, theme);
        // Resize/move cursor affordance: turn the pointer into the
        // familiar ↔ ↕ ⤡ ⤢ arrows over a window's edges/corners so
        // users see windows are resizable (R/desktop behaviour).
        ctx.set_cursor(self.mdi.hover_cursor(theme));

        self.console_input(ui_events, ctx, renderer, theme, ctx_was_open);

        // ── Paint
        frame.paint_rect(workspace.x, workspace.y, workspace.w, workspace.h,
                         theme.mdi_background);
        self.active_menu(graphics_menu).paint_strip(frame, renderer, menu_rect, theme);
        self.paint_windows(ui_events, frame, renderer, theme);
        self.handle_close_buttons();

        // ── Popup + context menus — painted LAST. Drop-down floats above
        //    every sub-window; the right-click context menu floats above
        //    everything including the popup. No window can cover a menu.
        self.active_menu(graphics_menu).paint_popup(frame, renderer, menu_rect, theme);
        self.ctx_console.paint(frame, renderer, theme);
        self.ctx_graphics.paint(frame, renderer, theme);

        self.run_dialogs(ctx, frame, renderer, theme, win_w, win_h);
    }

    fn active_menu(&self, graphics_menu: bool) -> &MenuBarState {
        if graphics_menu { &self.menu_graphics } else { &self.menu_console }
    }

    /// First frame: upload the title-bar logo and attach it to the
    /// console. Atlas alloc happens once; the ImageHandle is cheap to
    /// copy after (graphics windows pick it up as they open).
    fn upload_logo(&mut self, renderer: &mut Renderer) {
        if self.logo.uploaded { return; }
        if let Some(handle) = renderer.upload_image(self.logo.w, self.logo.h, &self.logo.rgba) {
            self.logo.handle = Some(handle);
            if let Some(w) = self.mdi.window_mut(self.console_id) {
                w.icon = Some(handle);
            }
        }
        self.logo.uploaded = true;
    }

    /// Engine device events → MDI sub-windows. Each `dev.new()` produces
    /// a Created event; we spawn a fresh sub-window + GraphPanel. Plotted
    /// events refresh the matching panel. Closed events hide + drop it.
    fn sync_devices(&mut self, renderer: &Renderer, theme: &Theme) {
        for ev in drain_events() {
            match ev {
                DeviceEvent::Created(id) => {
                    // R-style near-square device window, sized
                    // PROPORTIONALLY to the actual window (adapts 720p →
                    // 4K). Cascade subsequent devices so multiple windows
                    // don't overlap identically.
                    let ww = renderer.size.width  as f32;
                    let wh = renderer.size.height as f32;
                    let n = id.0 as f32;
                    let casc = theme.px(32.0);
                    let bounds = Rect {
                        x: ww * 0.55 + (n - 1.0) * casc,
                        y: menu_bar_height(theme) + wh * 0.03 + (n - 1.0) * casc * 0.8,
                        w: ww * 0.42,
                        h: wh * 0.72,
                    };
                    let wid = self.mdi.add_window(format!("R2 Graphics — Dev {}", id.0), bounds);
                    if let Some(handle) = self.logo.handle {
                        if let Some(w) = self.mdi.window_mut(wid) {
                            w.icon = Some(handle);
                        }
                    }
                    self.devices.insert(id, (wid, GraphPanel::new()));
                }
                DeviceEvent::Plotted(id) => {
                    if let Some(svg) = device_full_svg(id) {
                        if let Some((wid, panel)) = self.devices.get_mut(&id) {
                            panel.set_svg(svg.into_bytes());
                            if let Some(w) = self.mdi.window_mut(*wid) {
                                w.visible = true;
                            }
                        }
                    }
                }
                DeviceEvent::Closed(id) => {
                    if let Some((wid, _)) = self.devices.remove(&id) {
                        if let Some(w) = self.mdi.window_mut(wid) {
                            w.visible = false;
                        }
                    }
                }
                DeviceEvent::CurrentChanged(_) => { /* z-order shift handled on click */ }
            }
        }
    }

    /// The window of the engine's current graphics device, if one is open.
    fn graphics_window(&self) -> Option<WindowId> {
        let cur = r2_graphics::device::current_device()?;
        self.devices.get(&cur).map(|(w, _)| *w)
    }

    /// One-time adaptive layout: the OS window is maximized (see r2-ui
    /// WindowBuilder), but the console deliberately takes only the LEFT
    /// half of the workspace. Graphics devices open at x = 55% (see
    /// `sync_devices`), so console and plots sit SIDE BY SIDE — no window
    /// switching, no overlap, both readable at a glance. Full height,
    /// since the vertical space is free: long transcripts stay readable.
    /// Done once; the user can then drag / resize / maximize freely.
    fn first_layout(&mut self, workspace: Rect) {
        if self.did_layout || workspace.w <= 0.0 { return; }
        self.did_layout = true;
        if let Some(w) = self.mdi.window_mut(self.console_id) {
            w.bounds = Rect {
                x: workspace.x + workspace.w * 0.015,
                y: workspace.y + workspace.h * 0.03,
                // Right edge lands at ~53.5%, clear of the 55% where
                // graphics windows start.
                w: (workspace.w * 0.52).max(200.0),
                h: (workspace.h * 0.90).max(120.0),
            };
        }
    }

    /// Menu bar + right-click context menu events. Both funnel into the
    /// SAME dispatch — one place to add a feature, two ways for the user
    /// to reach it. Both are always handled (a click is consumed by
    /// whichever it landed on); the menu bar wins a tie.
    fn menu_events(&mut self, ui_events: &[InputEvent], menu_rect: Rect, graphics_menu: bool,
                   graphics_id: Option<WindowId>, renderer: &mut Renderer,
                   theme: &Theme) -> Option<String> {
        let topmost = self.mdi.z_order().last();
        let menu = if graphics_menu { &mut self.menu_graphics } else { &mut self.menu_console };
        let mb_action = menu.handle_events(ui_events, menu_rect, renderer, theme);
        let cm_action = match topmost {
            Some(id) if id == self.console_id => {
                let content = self.mdi.window(id).map(|w| w.content_rect(theme));
                content.and_then(|c| self.ctx_console.handle_events(ui_events, c, renderer, theme))
            }
            Some(id) if graphics_id == Some(id) => {
                let content = self.mdi.window(id).map(|w| w.content_rect(theme));
                content.and_then(|c| self.ctx_graphics.handle_events(ui_events, c, renderer, theme))
            }
            _ => None,
        };
        mb_action.or(cm_action)
    }

    /// Pure z-order: for each window from bottom to top, paint its BODY →
    /// CONTENT → TITLE BAR as one unit. The next-higher window's body then
    /// cleanly covers everything below it, including the previous title
    /// strip. No leaking title bars between windows.
    fn paint_windows(&mut self, ui_events: &[InputEvent], frame: &mut Frame,
                     renderer: &mut Renderer, theme: &Theme) {
        let order: Vec<WindowId> = self.mdi.z_order().collect();
        for id in order {
            if !self.mdi.should_paint_content(id) { continue; }
            self.mdi.paint_body(id, frame, theme);
            let content = self.mdi.window(id)
                .filter(|w| w.visible)
                .map(|w| w.content_rect(theme));
            let content = match content { Some(r) => r, None => continue };
            if id == self.console_id {
                self.paint_console(content, ui_events, frame, renderer, theme);
            } else {
                self.paint_graphics(id, content, frame, renderer, theme);
            }
            self.mdi.paint_titlebar(id, frame, renderer, theme);
        }
    }

    /// Any window other than the console is a graphics device: find its
    /// GraphPanel by window id, so any number of `dev.new()` windows can
    /// be open at once.
    fn paint_graphics(&mut self, id: WindowId, content: Rect, frame: &mut Frame,
                      renderer: &mut Renderer, theme: &Theme) {
        let panel = self.devices.values_mut().find(|(w, _)| *w == id).map(|(_, p)| p);
        if let Some(panel) = panel {
            // Plot canvases stay WHITE in every theme: a plot is a document
            // (it gets saved, printed and published), not chrome. R does
            // the same.
            frame.paint_rect(content.x, content.y, content.w, content.h, Color::WHITE);
            let inner = Rect {
                x: content.x + 8.0,  y: content.y + 8.0,
                w: (content.w - 16.0).max(0.0),
                h: (content.h - 16.0).max(0.0),
            };
            panel.paint(frame, renderer, inner, theme);
        }
    }

    /// The console's close button hides it; each graphics device's routes
    /// back to the engine via `close_device`, which emits a
    /// DeviceEvent::Closed picked up next frame.
    fn handle_close_buttons(&mut self) {
        if self.mdi.take_close_requested(self.console_id) {
            if let Some(w) = self.mdi.window_mut(self.console_id) { w.visible = false; }
        }
        let device_ids: Vec<_> = self.devices.iter().map(|(d, (w, _))| (*d, *w)).collect();
        for (dev_id, wid) in device_ids {
            if self.mdi.take_close_requested(wid) {
                r2_graphics::device::close_device(Some(dev_id));
            }
        }
    }
}
