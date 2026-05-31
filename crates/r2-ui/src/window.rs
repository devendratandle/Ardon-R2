//! Window — an MDI sub-window (R Console, R Graphics, etc.) inside
//! the framework's main frame. Has its own title bar, traffic-light
//! buttons, and a content body widget.

/// A sub-window's persistent state (position, size, collapsed flag).
/// The framework owns this; the host only configures defaults.
pub struct Window {
    pub title:    String,
    pub default_pos:  (f32, f32),
    pub default_size: (f32, f32),
    pub show_icon:    bool,
    pub min_max_close_buttons: bool,
    pub titlebar_drag: bool,
    pub resize_corners: bool,
    /// Optional one-time configuration: don't show this window until
    /// the host calls `mark_visible()`. Used for the Graphics sub-window
    /// which should stay hidden until the first plot.
    pub hidden_until_called: bool,
}

impl Window {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            default_pos:  (60.0, 80.0),
            default_size: (820.0, 480.0),
            show_icon:    true,
            min_max_close_buttons: true,
            titlebar_drag: true,
            resize_corners: true,
            hidden_until_called: false,
        }
    }
}

/// Mirror builder for declarative configuration inside `mdi(|m| { ... })`.
pub struct WindowBuilder {
    pub(crate) inner: Window,
}

impl WindowBuilder {
    pub fn new(title: impl Into<String>) -> Self {
        Self { inner: Window::new(title) }
    }
    pub fn default_pos(mut self, pos: (f32, f32)) -> Self {
        self.inner.default_pos = pos;
        self
    }
    pub fn default_size(mut self, size: (f32, f32)) -> Self {
        self.inner.default_size = size;
        self
    }
    pub fn hidden_until_called(mut self, h: bool) -> Self {
        self.inner.hidden_until_called = h;
        self
    }
}
