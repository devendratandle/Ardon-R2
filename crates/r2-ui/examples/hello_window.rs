//! Minimal R2-UI example — Phase 2 Week 2 deliverable.
//!
//! Opens a native window and clears it to the khaki theme color
//! every frame. The window can be resized, minimized, maximized,
//! and closed. Proves winit + wgpu + theme are wired correctly.
//!
//! Run with:
//!   cargo run -p r2-ui --example hello_window

fn main() -> Result<(), String> {
    r2_ui::R2Ui::app("R2-UI · Phase 2 Week 2 — Hello Window")
        .theme(r2_ui::Theme::khaki())
        .initial_size(900, 600)
        .run()
}
