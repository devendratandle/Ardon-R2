//! Ardon-R2 GUI framework.
//!
//! See `docs/R2_UI_FRAMEWORK.md` for the design contract. This crate
//! is the GUI substrate that R2Gui (and any other R2-UI consumer)
//! builds against.
//!
//! ## Architecture summary
//!
//! ```text
//!   ┌────────────────────────────────────────┐
//!   │  Public API (this crate's root)        │  R2Ui, R2UiBuilder
//!   ├────────────────────────────────────────┤
//!   │  Widget layer:                         │
//!   │  CellGrid · InputField · PlotPanel     │
//!   │  Window   · MenuBar    · Dialog        │
//!   ├────────────────────────────────────────┤
//!   │  Layout layer:                         │
//!   │  Mdi · Tabs · Split · Theme · Keymap   │
//!   ├────────────────────────────────────────┤
//!   │  Render layer:                         │
//!   │  paint_rect · paint_glyph · paint_image│
//!   ├────────────────────────────────────────┤
//!   │  Substrate (deps): winit, wgpu,        │
//!   │   fontdue, arboard, image, resvg       │
//!   └────────────────────────────────────────┘
//! ```
//!
//! ## Current implementation status
//!
//! Phase 2 Week 1 of the design doc: scaffolding + public API
//! placeholders. The types compile and re-export properly, but
//! actual rendering is not yet wired through. Subsequent sessions
//! fill in the implementations one module at a time.

pub mod app;
pub mod context_menu;
pub mod event;
pub mod grid;
pub mod input;
pub mod layout;
pub mod mdi;
pub mod graph;
pub mod menu;
pub mod render;
pub mod scrollbar;
pub mod svg_text;
pub mod theme;
pub mod window;

// Re-export the public surface so callers write `r2_ui::Theme` rather
// than `r2_ui::theme::Theme`.
pub use app::{FrameCtx, FrameFn, R2Ui, R2UiApp};
pub use context_menu::{ContextItem, ContextMenu};
pub use event::{normalize_paste, Clipboard, InputEvent, KeyCode, Mods, MouseButton, MousePos};
pub use grid::{
    auto_scroll_offset, hit_test, paint_cells, paint_cells_scrolled,
    scroll_pos_to_col, scroll_pos_to_row,
    Cell, CellGrid, CellGridResponse, CellGridState, GridPos, Rect, Selection,
};
pub use input::{InputField, InputFieldResponse};
pub use layout::{Layout, LayoutBuilder, Mdi};
pub use mdi::{MdiHost, SubWindow, WindowId};
pub use menu::{MenuBar, MenuBarState, MenuBuilder, MenuItem, MenuTopLevel, MENU_BAR_HEIGHT};
pub use graph::GraphPanel;
pub use render::{Frame, GlyphInfo, ImageHandle, Renderer};
pub use scrollbar::{Scrollbar, ScrollOrientation, SCROLLBAR_THICKNESS};
pub use theme::{Color, Theme};
pub use window::{Window as UiWindow, WindowBuilder};
