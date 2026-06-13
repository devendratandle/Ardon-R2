//! Theme — pure data, swappable at runtime, third-party-extensible.
//!
//! Themes drive every visible color in R2-UI. They're plain structs;
//! making a new theme means writing a `pub fn my_theme() -> Theme`
//! that returns one. Third-party theme crates require no API beyond
//! this struct.

/// 8-bit-per-channel RGBA color. Compact, plain data, no GPU deps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color(pub u8, pub u8, pub u8, pub u8);

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self { Color(r, g, b, 255) }
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self { Color(r, g, b, a) }
    pub const WHITE: Self = Color::rgb(255, 255, 255);
    pub const BLACK: Self = Color::rgb(0, 0, 0);
}

/// Every color and font metric the framework exposes to themes.
/// Adding a field here is a semver-major change post-1.0; adding a
/// constructor is fine.
#[derive(Debug, Clone)]
pub struct Theme {
    // ── Background colors ───────────────────────────────────────
    pub mdi_background:    Color,  // outer workspace (between menu and edges)
    pub window_background: Color,  // sub-window body
    pub menu_background:   Color,
    pub menu_text:         Color,

    // ── Sub-window chrome (active = topmost, passive = behind) ──
    pub titlebar_active:   Color,  // fill for the focused window's title strip
    pub titlebar_passive:  Color,  // fill for background windows' title strips
    pub border_active:     Color,  // 1-px border of the focused window
    pub border_passive:    Color,  // 1-px border of background windows

    // ── Console palette ─────────────────────────────────────────
    pub console_input:        Color,  // user-typed lines, the R2> prompt
    pub console_output:       Color,  // engine results
    pub console_error:        Color,  // error / warning messages
    pub console_banner:       Color,  // welcome text
    pub console_selection_bg: Color,  // selection highlight on transcript
    pub cursor:               Color,  // blinking I-beam color

    // ── Title-bar traffic-light buttons ─────────────────────────
    pub button_min:   Color,   // —  minimize/collapse
    pub button_max:   Color,   // □  maximize/restore
    pub button_close: Color,   // ✕  close

    // ── Typography ──────────────────────────────────────────────
    // `font_size` / `line_height` are the EFFECTIVE (already DPI-scaled)
    // values — every widget reads these directly, so applying the scale
    // here at the source makes the whole UI (console, menus, title bars,
    // context menus) adapt in one place. The unscaled designs live in
    // `font_size_base` / `line_height_base`; `set_dpi`/`set_font_size`
    // recompute the effective values.
    pub font_size:   f32,
    pub line_height: f32,
    font_size_base:   f32,
    line_height_base: f32,

    // ── Density scale (HiDPI / resolution scaling) ──────────────
    // Effective UI scale: max(OS scaling factor, panel-height/1080, 1).
    // 1.0 = 96 DPI / 100% on a 1080p panel. The app shell updates it at
    // startup and on monitor changes so the same widget code looks
    // proportionally identical from 720p to 4K.
    pub dpi: f32,
}

impl Theme {
    /// Effective (scaled) font size — kept for call sites that used the
    /// explicit accessor; identical to reading `font_size` now.
    #[inline] pub fn fs(&self) -> f32 { self.font_size }
    /// Effective (scaled) line height.
    #[inline] pub fn lh(&self) -> f32 { self.line_height }
    /// Scaled length helper — for any pixel constant in widget code
    /// (button sizes, paddings, bar heights, grips).
    #[inline] pub fn px(&self, v: f32) -> f32 { v * self.dpi }
    /// Set the UI scale; recomputes the effective typography from the
    /// stored base design values.
    pub fn set_dpi(&mut self, dpi: f32) {
        self.dpi = dpi.max(0.5).min(4.0);
        self.font_size   = self.font_size_base   * self.dpi;
        self.line_height = self.line_height_base * self.dpi;
    }
    /// User font preference (e.g. a future settings tab): change the BASE
    /// size; the effective size stays DPI-scaled on top of it.
    pub fn set_font_size(&mut self, base_pt: f32) {
        self.font_size_base   = base_pt.clamp(7.0, 32.0);
        self.line_height_base = (self.font_size_base * 1.3).round();
        self.font_size   = self.font_size_base   * self.dpi;
        self.line_height = self.line_height_base * self.dpi;
    }
}

impl Theme {
    /// Faint khaki workspace + classic R Console palette inside.
    /// The default theme — warm, easy on the eyes, R-faithful.
    pub fn khaki() -> Self {
        Theme {
            // Workspace = deeper khaki with a black/olive undertone.
            // 148/140/118 reads as a calm dark backdrop; white sub-
            // window bodies pop strongly against it. Menu bar a touch
            // lighter so the strip is distinguishable from the area
            // sub-windows live in.
            mdi_background:    Color::rgb(148, 140, 118),
            window_background: Color::WHITE,
            menu_background:   Color::rgb(186, 178, 156),
            menu_text:         Color::rgb(20, 20, 30),
            // Two-level faint-blue title strips, slightly more
            // saturated now so the focused window stands out against
            // the darker workspace.
            titlebar_active:   Color::rgb(200, 218, 238),
            titlebar_passive:  Color::rgb(222, 230, 242),
            border_active:     Color::rgb(110, 148, 188),
            border_passive:    Color::rgb(176, 192, 212),
            // Console palette — deeper red / blue. R-faithful weight,
            // not faint. Same hue family as before but darker.
            // Rose-red prompt + violet-blue output. R-Console-faithful
            // weight, distinct hues that read clearly on the cream
            // sub-window background.
            console_input:     Color::rgb(150, 28, 60),     // rose red
            console_output:    Color::rgb(40,  30, 130),    // violet blue
            console_error:     Color::rgb(180, 22, 30),     // deeper red
            console_banner:    Color::rgb(40,  30, 130),    // matches output
            console_selection_bg: Color::rgba(65, 105, 225, 120), // royal blue, medium
            cursor:            Color::rgb(196, 40, 40),
            button_min:        Color::rgb(40,  160, 60),    // green
            button_max:        Color::rgb(40,  120, 200),   // blue
            button_close:      Color::rgb(210, 50, 50),     // red
            font_size:   14.0,
            line_height: 18.0,
            font_size_base:   14.0,
            line_height_base: 18.0,
            dpi:         1.0,
        }
    }

    /// Faithful clone of R's original RGui Console look —
    /// pure white workspace, black on white text, traditional.
    pub fn rgui() -> Self {
        Theme {
            mdi_background:    Color::rgb(192, 192, 192),
            window_background: Color::WHITE,
            menu_background:   Color::rgb(240, 240, 240),
            menu_text:         Color::BLACK,
            titlebar_active:   Color::rgb(200, 216, 232),
            titlebar_passive:  Color::rgb(228, 232, 240),
            border_active:     Color::rgb(120, 150, 188),
            border_passive:    Color::rgb(192, 200, 216),
            console_input:     Color::rgb(196, 40, 40),
            console_output:    Color::rgb(32, 92, 168),
            console_error:     Color::rgb(208, 48, 48),
            console_banner:    Color::rgb(32, 92, 168),
            console_selection_bg: Color::rgba(65, 105, 225, 120), // royal blue, medium
            cursor:            Color::BLACK,
            button_min:        Color::rgb(40, 160, 60),
            button_max:        Color::rgb(40, 120, 200),
            button_close:      Color::rgb(210, 50, 50),
            font_size:   13.0,
            line_height: 16.0,
            font_size_base:   13.0,
            line_height_base: 16.0,
            dpi:         1.0,
        }
    }
}

impl Default for Theme {
    fn default() -> Self { Theme::khaki() }
}
