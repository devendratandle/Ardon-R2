//! The two modal dialogs: quit ("Save workspace image?") and Settings
//! (UI font size). They are handled and painted last in a frame, so they
//! dim and float above everything and own the keyboard and mouse.

use r2_ui::{DialogButton, Frame, FrameCtx, Renderer, Theme};

use crate::support::run_source;
use crate::Gui;

impl Gui {
    /// Quit path (R-style): q()/quit() set `quit_requested`, and the OS
    /// close button (✕ / Alt-F4) arrives as `close_requested`. Instead of
    /// exiting outright, pop the "Save workspace image? [Yes/No/Cancel]"
    /// confirmation. The engine flag is consumed so it fires once.
    pub(crate) fn open_quit_dialog_on_request(&mut self, close_requested: bool) {
        let quit_flag = std::mem::take(&mut self.quit_requested);
        if !(quit_flag || close_requested)
            || self.quit_dialog.is_open()
            || self.settings_dialog.is_open()
        {
            return;
        }
        let d = &mut self.quit_dialog;
        d.title = "Quit Ardon-R2".into();
        d.lines = vec!["Save workspace image?".into()];
        d.buttons = vec![
            DialogButton::new("Yes",    "quit.yes"),
            DialogButton::new("No",     "quit.no"),
            DialogButton::new("Cancel", "quit.cancel"),
        ];
        d.default_action = "quit.yes".into();
        d.cancel_action  = "quit.cancel".into();
        d.open();
    }

    pub(crate) fn open_settings(&mut self) {
        let d = &mut self.settings_dialog;
        d.title = "Settings".into();
        d.buttons = vec![
            DialogButton::new("A \u{2013}", "settings.font_dec"),
            DialogButton::new("A +",        "settings.font_inc"),
            DialogButton::new("Reset",      "settings.font_reset"),
            DialogButton::new("Close",      "settings.close"),
        ];
        d.default_action = String::new();
        d.cancel_action  = "settings.close".into();
        d.open();
    }

    pub(crate) fn run_dialogs(&mut self, ctx: &mut FrameCtx, frame: &mut Frame,
                              renderer: &mut Renderer, theme: &Theme, win_w: f32, win_h: f32) {
        if self.settings_dialog.is_open() {
            // Live body: show the current base (pre-DPI) font size.
            self.settings_dialog.lines = vec![
                format!("UI font size:  {} pt", theme.font_size_base() as i32),
                String::new(),
                "Resizes the console + window text.".into(),
                "(DPI scaling applies on top automatically.)".into(),
            ];
            let action = self.settings_dialog.handle_events(ctx.events, renderer, theme, win_w, win_h);
            match action.as_deref() {
                Some("settings.font_dec")   => ctx.set_base_font_size(theme.font_size_base() - 1.0),
                Some("settings.font_inc")   => ctx.set_base_font_size(theme.font_size_base() + 1.0),
                Some("settings.font_reset") => ctx.set_base_font_size(14.0),
                Some("settings.close")      => self.settings_dialog.close(),
                _ => {}
            }
            self.settings_dialog.paint(frame, renderer, theme, win_w, win_h);
        }

        if self.quit_dialog.is_open() {
            let action = self.quit_dialog.handle_events(ctx.events, renderer, theme, win_w, win_h);
            match action.as_deref() {
                Some("quit.yes") => {
                    // Save the session (all variables) to a default
                    // workspace file, then quit — R writes .RData;
                    // Ardon-R2 writes a .r2s session.
                    run_source("save(\".r2session.r2s\")",
                               &mut self.engine, &self.buffer, &mut self.quit_requested);
                    ctx.request_exit();
                }
                Some("quit.no")     => ctx.request_exit(),
                Some("quit.cancel") => self.quit_dialog.close(),
                _ => {}
            }
            self.quit_dialog.paint(frame, renderer, theme, win_w, win_h);
        }
    }
}
