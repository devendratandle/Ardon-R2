//! Layout — how sub-windows are arranged within the main frame.

use crate::window::Window;

/// Choose the overall arrangement. `Mdi` is the desktop default
/// (floating sub-windows on a workspace, RGui-style). `Tabs` swap a
/// single visible pane between named tabs (Android/mobile default).
/// `Split` divides the area into two resizable panes.
pub enum Layout {
    Mdi(Vec<Window>),
    Tabs(Vec<(String, Window)>),
    Split { horizontal: bool, a: Box<Window>, b: Box<Window> },
}

/// Builder used inside `R2Ui::app(...).mdi(|mdi| { ... })`.
pub struct LayoutBuilder {
    pub(crate) windows: Vec<Window>,
}

impl LayoutBuilder {
    pub fn new() -> Self { Self { windows: Vec::new() } }
}

impl Default for LayoutBuilder {
    fn default() -> Self { Self::new() }
}

/// Convenience type alias for the most common case (MDI workspace).
pub type Mdi = LayoutBuilder;
