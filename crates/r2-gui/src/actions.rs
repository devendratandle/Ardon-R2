//! Menu and context-menu commands. Action strings come from `menus.rs`;
//! the menu bar and the right-click menus share this one dispatch.
//!
//! Never hold the console buffer's lock across an engine or graphics
//! call: output from the call goes through the GUI sink, which takes the
//! same lock.

use r2_graphics::device::{current_has_plot, render_to_rgba, save_plot};
use r2_ui::{FrameCtx, GridPos, Renderer, Selection, Theme, WindowId};

use crate::support::{console_wrap_width, rows_from_buffer, take_engine_svg, wrap_rows};
use crate::Gui;

impl Gui {
    pub(crate) fn dispatch(&mut self, action: &str, ctx: &mut FrameCtx, renderer: &mut Renderer,
                           theme: &Theme, graphics_id: Option<WindowId>) {
        match action {
            "file.quit"            => self.quit_requested = true,
            "edit.settings"        => self.open_settings(),
            "file.clear"           => self.buffer.lock().unwrap().clear(),
            "file.save_plot"       => self.save_plot(graphics_id, theme, false),
            "file.save_plot_png"   => self.save_plot(graphics_id, theme, true),
            "file.copy_plot"       => self.copy_plot_svg(ctx),
            "file.copy_plot_image" => self.copy_plot_image(ctx, graphics_id, theme),
            "edit.copy"            => self.copy_selection(ctx, renderer, theme),
            "edit.paste"           => self.paste(ctx),
            "edit.select_all"      => self.select_all(renderer, theme),
            "win.console" => {
                if let Some(w) = self.mdi.window_mut(self.console_id) { w.visible = true; }
            }
            "win.graphics" => {
                // Reveal every device's window. Cheap when none are open.
                for (wid, _) in self.devices.values() {
                    if let Some(w) = self.mdi.window_mut(*wid) { w.visible = true; }
                }
            }
            "help.about" => {
                let mut b = self.buffer.lock().unwrap();
                b.push_banner("Ardon-R2 — pure-Rust reimplementation of R, AGPL-3.0.");
                b.push_banner("GUI built on the r2-ui framework (winit + wgpu + fontdue).");
            }
            _ => {}
        }
    }

    /// Resolution-aware plot size: the Graphics window's current panel
    /// rect × DPI, so a 4K / 200% screen gives a 4K image and 100% a
    /// panel-sized one — exactly what the panel shows.
    fn plot_pixels(&self, graphics_id: Option<WindowId>, theme: &Theme) -> (u32, u32) {
        graphics_id
            .and_then(|gid| self.mdi.window(gid).map(|w| {
                let r = w.content_rect(theme);
                (((r.w * theme.dpi).round() as u32).max(320),
                 ((r.h * theme.dpi).round() as u32).max(240))
            }))
            .unwrap_or((1024, 768))
    }

    /// Save the current plot through a file dialog: SVG or PNG by the
    /// chosen extension, or PNG only. SVG ignores the pixel size (vector).
    fn save_plot(&self, graphics_id: Option<WindowId>, theme: &Theme, png_only: bool) {
        let has_plot = if png_only { current_has_plot() } else { take_engine_svg().is_some() };
        if !has_plot {
            self.buffer.lock().unwrap().push_output("No plot to save.");
            return;
        }
        let (sw, sh) = self.plot_pixels(graphics_id, theme);
        let dialog = rfd::FileDialog::new();
        let dialog = if png_only {
            dialog.set_title("Save R2 plot as PNG")
                .set_file_name("plot.png")
                .add_filter("PNG image", &["png"])
        } else {
            dialog.set_title("Save R2 plot")
                .set_file_name("plot.svg")
                .add_filter("SVG vector",     &["svg"])
                .add_filter("PNG image",      &["png"])
                .add_filter("All supported",  &["svg", "png"])
        };
        let Some(path) = dialog.save_file() else { return };
        let path_str = path.to_string_lossy().into_owned();
        let result = save_plot(&path_str, sw, sh);
        let what = if png_only { "PNG" } else { "plot" };
        match result {
            Ok(_)  => self.buffer.lock().unwrap()
                .push_output(&format!("Saved {} to {} ({}×{})", what, path_str, sw, sh)),
            Err(e) => self.buffer.lock().unwrap()
                .push_error(&format!("Save failed: {}", e.msg)),
        }
    }

    /// Copy the raw SVG source to the clipboard so the user can paste
    /// into an editor or vector tool.
    fn copy_plot_svg(&self, ctx: &mut FrameCtx) {
        if let Some(svg) = take_engine_svg() {
            ctx.clipboard.set_text(&svg);
            self.buffer.lock().unwrap().push_output("Plot SVG copied to clipboard.");
        } else {
            self.buffer.lock().unwrap().push_output("No plot to copy.");
        }
    }

    /// Rasterise the current plot at the Graphics window's pixel size and
    /// put the bitmap on the clipboard. Pastes into Word / Excel / Outlook
    /// / any image editor that accepts a clipboard bitmap.
    fn copy_plot_image(&self, ctx: &mut FrameCtx, graphics_id: Option<WindowId>, theme: &Theme) {
        if !current_has_plot() {
            self.buffer.lock().unwrap().push_output("No plot to copy.");
            return;
        }
        let (sw, sh) = self.plot_pixels(graphics_id, theme);
        match render_to_rgba(sw, sh) {
            Ok((rgba, w, h)) => {
                if ctx.clipboard.set_image(w, h, &rgba) {
                    self.buffer.lock().unwrap().push_output(
                        &format!("Plot copied to clipboard as {}×{} image.", w, h));
                } else {
                    self.buffer.lock().unwrap().push_error(
                        "Clipboard image copy failed (OS rejected).");
                }
            }
            Err(e) => self.buffer.lock().unwrap()
                .push_error(&format!("Rasterise failed: {}", e.msg)),
        }
    }

    /// Copy the current selection. The rows include the LIVE prompt row,
    /// because paint appends it and selections are indexed against that
    /// combined list — without it, a selection touching the last visible
    /// line fell off the end and copied nothing.
    fn copy_selection(&self, ctx: &mut FrameCtx, renderer: &mut Renderer, theme: &Theme) {
        let rows = wrap_rows(self.rows_with_prompt(theme), self.console_wrap(renderer, theme));
        if let Some(sel) = self.grid.selection {
            let text = r2_ui::grid::selection_to_text(&rows, sel);
            if !text.is_empty() {
                ctx.clipboard.set_text(&text);
            }
        }
    }

    /// Paste through the same multi-line path InputField's Ctrl+V uses:
    /// the first chunk completes the line being typed, each intermediate
    /// line auto-submits as if Enter-pressed, the final chunk stays in
    /// the editor. Identical whether the user typed Ctrl+V, picked
    /// Edit ▸ Paste, or right-clicked → Paste.
    fn paste(&mut self, ctx: &mut FrameCtx) {
        let Some(s) = ctx.clipboard.get_text() else { return };
        let s = s.replace('\r', "");
        if !s.contains('\n') {
            let f = &mut self.input;
            let pos = f.cursor;
            f.current.insert_str(pos, &s);
            f.cursor = pos + s.len();
            return;
        }
        let mut parts: Vec<String> = s.split('\n').map(String::from).collect();
        let head = parts.remove(0);
        let tail = parts.pop().unwrap_or_default();
        // Insert head into the current line, then take its full content as
        // the first submission, plus any middle lines.
        let pos = self.input.cursor;
        self.input.current.insert_str(pos, &head);
        let first = std::mem::take(&mut self.input.current);
        for line in std::iter::once(first).chain(parts) {
            self.submit(line);
        }
        self.input.current = tail;
        self.input.cursor = self.input.current.len();
    }

    /// Select the whole transcript (wrapped, to match the rows paint and
    /// selection use).
    fn select_all(&mut self, renderer: &mut Renderer, theme: &Theme) {
        let rows = rows_from_buffer(&self.buffer.lock().unwrap(), theme);
        let (cell_w, _) = renderer.cell_metrics(theme.fs());
        let ww = self.mdi.window(self.console_id)
            .map(|w| console_wrap_width(w.content_rect(theme).w, cell_w))
            .unwrap_or(0);
        let rows = wrap_rows(rows, ww);
        if !rows.is_empty() {
            let last = rows.len() - 1;
            let last_col = rows[last].len().saturating_sub(1);
            self.grid.selection = Some(Selection {
                start: GridPos { row: 0, col: 0 },
                end:   GridPos { row: last, col: last_col },
            });
        }
    }
}
