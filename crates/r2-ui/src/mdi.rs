//! MDI host — nested sub-windows inside one OS window.
//!
//! Each [`SubWindow`] is a free-floating rectangle on the MDI
//! workspace with its own title bar (title + traffic-light buttons),
//! drag-by-titlebar, resize-by-corner-grip, maximize/restore, and
//! z-order management. The host's content area for each window is
//! the body region below the title bar — widgets like `CellGridState`
//! paint into that.
//!
//! Design discipline:
//! - The MDI host knows nothing about the widgets inside its windows.
//!   The user supplies a per-window paint closure that receives the
//!   content rect.
//! - Drag/resize is driven by the same [`InputEvent`] stream as every
//!   other widget — no special winit hooks.
//! - Z-order is a single `Vec<id>`; clicking any window brings it to
//!   the front. No focus tracking beyond z-order.

use crate::event::{InputEvent, MouseButton, MousePos};
use crate::grid::Rect;
use crate::render::{Frame, ImageHandle, Renderer};
use crate::theme::{Color, Theme};

/// Unique handle for a sub-window registered with [`MdiHost`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub usize);

/// One floating sub-window. `bounds` is its position + size on the
/// MDI workspace, in pixels.
#[derive(Debug, Clone)]
pub struct SubWindow {
    pub id: WindowId,
    pub title: String,
    pub bounds: Rect,
    /// Snapshot of bounds taken at maximize, restored on un-maximize.
    pub saved_bounds: Option<Rect>,
    pub maximized: bool,
    pub visible: bool,
    pub close_requested: bool,
    /// Optional small icon shown at the left edge of the title bar.
    /// Square aspect ratio looks best; oversized images are
    /// downscaled to fit the title-bar height.
    pub icon: Option<ImageHandle>,
}

impl SubWindow {
    pub fn new(id: WindowId, title: impl Into<String>, bounds: Rect) -> Self {
        Self {
            id, title: title.into(), bounds,
            saved_bounds: None, maximized: false,
            visible: true, close_requested: false,
            icon: None,
        }
    }

    /// Content rect = bounds minus title bar + 1px chrome border.
    pub fn content_rect(&self, theme: &Theme) -> Rect {
        let tb = title_bar_height(theme);
        Rect {
            x: self.bounds.x + 1.0,
            y: self.bounds.y + tb,
            w: (self.bounds.w - 2.0).max(0.0),
            h: (self.bounds.h - tb - 1.0).max(0.0),
        }
    }
}

#[inline]
fn title_bar_height(theme: &Theme) -> f32 { (theme.line_height + 8.0).ceil() }
const BTN_SIZE: f32 = 12.0;
const BTN_GAP:  f32 =  6.0;
const RESIZE_GRIP: f32 = 18.0;   // BR corner
const EDGE_GRAB:   f32 =  6.0;   // right edge + bottom edge thickness

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragMode {
    Move,
    // 4 edges
    ResizeL, ResizeR, ResizeT, ResizeB,
    // 4 corners
    ResizeTL, ResizeTR, ResizeBL, ResizeBR,
}

#[derive(Debug, Clone, Copy)]
struct DragState {
    id: WindowId,
    mode: DragMode,
    /// Offset from the mouse to the window's top-left (Move) or to
    /// its bottom-right corner (ResizeBR), captured at MouseDown.
    anchor_dx: f32,
    anchor_dy: f32,
}

/// Owner of a set of sub-windows + their interaction state.
pub struct MdiHost {
    pub windows: Vec<SubWindow>,
    /// Workspace rect (set every frame by the caller from window size).
    pub workspace: Rect,
    /// IDs in stacking order; last element is the topmost window.
    z_order: Vec<WindowId>,
    drag: Option<DragState>,
    next_id: usize,
    /// Per-window-id flag set when the user clicks the close button.
    /// Caller checks `take_close_requested(id)` after `handle_events`.
    last_mouse: MousePos,
}

impl Default for MdiHost {
    fn default() -> Self { Self::new() }
}

impl MdiHost {
    pub fn new() -> Self {
        Self {
            windows: Vec::new(),
            workspace: Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 },
            z_order: Vec::new(),
            drag: None,
            next_id: 0,
            last_mouse: MousePos { x: 0.0, y: 0.0 },
        }
    }

    pub fn add_window(&mut self, title: impl Into<String>, bounds: Rect) -> WindowId {
        let id = WindowId(self.next_id);
        self.next_id += 1;
        self.windows.push(SubWindow::new(id, title, bounds));
        self.z_order.push(id);
        id
    }

    pub fn window(&self, id: WindowId) -> Option<&SubWindow> {
        self.windows.iter().find(|w| w.id == id)
    }
    pub fn window_mut(&mut self, id: WindowId) -> Option<&mut SubWindow> {
        self.windows.iter_mut().find(|w| w.id == id)
    }
    pub fn set_workspace(&mut self, rect: Rect) { self.workspace = rect; }

    /// Returns `true` (once) if the user clicked the close button on
    /// the given window since the last call. Self-clearing.
    pub fn take_close_requested(&mut self, id: WindowId) -> bool {
        if let Some(w) = self.window_mut(id) {
            let v = w.close_requested;
            w.close_requested = false;
            v
        } else { false }
    }

    /// Iterate windows in z-order (bottom → top), giving each a
    /// reference and its content rect. Useful for painting bodies
    /// after `paint_chrome` has drawn the frames.
    pub fn z_order(&self) -> impl Iterator<Item = WindowId> + '_ {
        self.z_order.iter().copied()
    }

    fn bring_to_front(&mut self, id: WindowId) {
        if let Some(pos) = self.z_order.iter().position(|w| *w == id) {
            self.z_order.remove(pos);
            self.z_order.push(id);
        }
    }

    /// Topmost visible window whose `bounds` contains the point.
    fn hit_topmost(&self, p: MousePos) -> Option<WindowId> {
        for id in self.z_order.iter().rev() {
            let w = self.window(*id)?;
            if w.visible && point_in(w.bounds, p) {
                return Some(*id);
            }
        }
        None
    }

    /// Toggle maximize on the given window — save / restore bounds.
    pub fn toggle_maximize(&mut self, id: WindowId) {
        let workspace = self.workspace;
        if let Some(w) = self.window_mut(id) {
            if w.maximized {
                if let Some(saved) = w.saved_bounds.take() { w.bounds = saved; }
                w.maximized = false;
            } else {
                w.saved_bounds = Some(w.bounds);
                w.bounds = workspace;
                w.maximized = true;
            }
        }
    }

    /// Walk this frame's events: update drag state, fire close/maximize
    /// button hits, re-stack on click.
    pub fn handle_events(&mut self, events: &[InputEvent], theme: &Theme) {
        // Snapshot the workspace rect once so the move/resize arms
        // can clamp bounds without re-borrowing self.
        let workspace = self.workspace;
        for ev in events {
            match *ev {
                InputEvent::MouseDown { button: MouseButton::Left, pos } => {
                    self.last_mouse = pos;
                    let hit = self.hit_topmost(pos);
                    if let Some(id) = hit {
                        self.bring_to_front(id);
                        let zone = classify_hit(self.window(id).unwrap(), pos, theme);
                        match zone {
                            HitZone::Button(ButtonHover::Min) => {
                                if let Some(w) = self.window_mut(id) { w.visible = false; }
                            }
                            HitZone::Button(ButtonHover::Max) => self.toggle_maximize(id),
                            HitZone::Button(ButtonHover::Close) => {
                                if let Some(w) = self.window_mut(id) { w.close_requested = true; }
                            }
                            HitZone::ResizeBR | HitZone::ResizeBL |
                            HitZone::ResizeTR | HitZone::ResizeTL |
                            HitZone::ResizeR  | HitZone::ResizeL  |
                            HitZone::ResizeT  | HitZone::ResizeB  => {
                                let w = self.window(id).unwrap();
                                let mode = match zone {
                                    HitZone::ResizeBR => DragMode::ResizeBR,
                                    HitZone::ResizeBL => DragMode::ResizeBL,
                                    HitZone::ResizeTR => DragMode::ResizeTR,
                                    HitZone::ResizeTL => DragMode::ResizeTL,
                                    HitZone::ResizeR  => DragMode::ResizeR,
                                    HitZone::ResizeL  => DragMode::ResizeL,
                                    HitZone::ResizeT  => DragMode::ResizeT,
                                    HitZone::ResizeB  => DragMode::ResizeB,
                                    _ => unreachable!(),
                                };
                                // For each drag mode the anchor records
                                // the gap between the mouse and the
                                // edge/corner being dragged. The drag
                                // arithmetic in MouseMoved reuses the
                                // anchor to keep the edge under the
                                // cursor as the user moves.
                                let (adx, ady) = match mode {
                                    DragMode::ResizeR | DragMode::ResizeBR | DragMode::ResizeTR
                                        => (pos.x - (w.bounds.x + w.bounds.w), 0.0),
                                    DragMode::ResizeL | DragMode::ResizeBL | DragMode::ResizeTL
                                        => (pos.x - w.bounds.x, 0.0),
                                    _ => (0.0, 0.0),
                                };
                                let (_, ady) = match mode {
                                    DragMode::ResizeB | DragMode::ResizeBR | DragMode::ResizeBL
                                        => (adx, pos.y - (w.bounds.y + w.bounds.h)),
                                    DragMode::ResizeT | DragMode::ResizeTR | DragMode::ResizeTL
                                        => (adx, pos.y - w.bounds.y),
                                    _ => (adx, ady),
                                };
                                self.drag = Some(DragState {
                                    id, mode,
                                    anchor_dx: adx, anchor_dy: ady,
                                });
                            }
                            HitZone::TitleBar => {
                                let w = self.window(id).unwrap();
                                self.drag = Some(DragState {
                                    id, mode: DragMode::Move,
                                    anchor_dx: pos.x - w.bounds.x,
                                    anchor_dy: pos.y - w.bounds.y,
                                });
                            }
                            HitZone::Body | HitZone::None => {}
                        }
                    }
                }
                InputEvent::MouseMoved(pos) => {
                    self.last_mouse = pos;
                    if let Some(d) = self.drag {
                        if let Some(w) = self.window_mut(d.id) {
                            // 8-direction resize + move. Edges adjust
                            // one dimension; corners adjust two.
                            // Left/Top edges shift the window's origin
                            // AND its dimension so the far edge stays
                            // anchored to the same screen position.
                            const MIN_W: f32 = 180.0;
                            const MIN_H: f32 =  80.0;
                            let right_x  = w.bounds.x + w.bounds.w;
                            let bottom_y = w.bounds.y + w.bounds.h;
                            match d.mode {
                                DragMode::Move => {
                                    w.bounds.x = pos.x - d.anchor_dx;
                                    w.bounds.y = pos.y - d.anchor_dy;
                                }
                                DragMode::ResizeR => {
                                    w.bounds.w = (pos.x - d.anchor_dx - w.bounds.x).max(MIN_W);
                                }
                                DragMode::ResizeL => {
                                    let new_x = pos.x - d.anchor_dx;
                                    let new_w = (right_x - new_x).max(MIN_W);
                                    w.bounds.x = right_x - new_w;
                                    w.bounds.w = new_w;
                                }
                                DragMode::ResizeB => {
                                    w.bounds.h = (pos.y - d.anchor_dy - w.bounds.y).max(MIN_H);
                                }
                                DragMode::ResizeT => {
                                    let new_y = pos.y - d.anchor_dy;
                                    let new_h = (bottom_y - new_y).max(MIN_H);
                                    w.bounds.y = bottom_y - new_h;
                                    w.bounds.h = new_h;
                                }
                                DragMode::ResizeBR => {
                                    w.bounds.w = (pos.x - d.anchor_dx - w.bounds.x).max(MIN_W);
                                    w.bounds.h = (pos.y - d.anchor_dy - w.bounds.y).max(MIN_H);
                                }
                                DragMode::ResizeBL => {
                                    let new_x = pos.x - d.anchor_dx;
                                    let new_w = (right_x - new_x).max(MIN_W);
                                    w.bounds.x = right_x - new_w;
                                    w.bounds.w = new_w;
                                    w.bounds.h = (pos.y - d.anchor_dy - w.bounds.y).max(MIN_H);
                                }
                                DragMode::ResizeTR => {
                                    w.bounds.w = (pos.x - d.anchor_dx - w.bounds.x).max(MIN_W);
                                    let new_y = pos.y - d.anchor_dy;
                                    let new_h = (bottom_y - new_y).max(MIN_H);
                                    w.bounds.y = bottom_y - new_h;
                                    w.bounds.h = new_h;
                                }
                                DragMode::ResizeTL => {
                                    let new_x = pos.x - d.anchor_dx;
                                    let new_w = (right_x - new_x).max(MIN_W);
                                    w.bounds.x = right_x - new_w;
                                    w.bounds.w = new_w;
                                    let new_y = pos.y - d.anchor_dy;
                                    let new_h = (bottom_y - new_y).max(MIN_H);
                                    w.bounds.y = bottom_y - new_h;
                                    w.bounds.h = new_h;
                                }
                            }
                            // Any user-initiated move/resize cancels maximize.
                            if w.maximized {
                                w.maximized = false;
                                w.saved_bounds = None;
                            }
                            // Clamp top so the window can never drift
                            // above the workspace (which sits below
                            // the menu bar). Other sides intentionally
                            // unclamped so the user can park a window
                            // partially off-screen if they want.
                            if w.bounds.y < workspace.y {
                                w.bounds.y = workspace.y;
                            }
                        }
                    }
                }
                InputEvent::MouseUp { button: MouseButton::Left, .. } => {
                    self.drag = None;
                }
                _ => {}
            }
        }
    }

    /// IDs of windows whose chrome should be painted this frame, in
    /// bottom-to-top order. When any window is maximized, ONLY that
    /// window's chrome is painted — neighbours are fully hidden so
    /// their title strips do not leak on top of the maximized
    /// window. Otherwise every visible window in z-order shows.
    fn paint_targets(&self) -> Vec<WindowId> {
        if let Some(mid) = self.windows.iter()
            .find(|w| w.maximized && w.visible)
            .map(|w| w.id)
        {
            return vec![mid];
        }
        self.z_order.iter().copied()
            .filter(|id| self.window(*id).map(|w| w.visible).unwrap_or(false))
            .collect()
    }

    /// Whether the caller should paint per-window content (transcript,
    /// plot) for the given id this frame. Maximized windows hide
    /// their neighbours entirely; in normal mode, every visible
    /// window is paintable.
    pub fn should_paint_content(&self, id: WindowId) -> bool {
        self.paint_targets().contains(&id)
    }

    /// Body fill + border + resize grip for ONE window. Caller uses
    /// this with the per-id paint loop so each window's entire chrome
    /// stays grouped — body, then content, then title bar — before
    /// the next-higher window paints. That gives pure z-order: front
    /// window fully covers back windows, no title-strip leaks.
    pub fn paint_body(&self, id: WindowId, frame: &mut Frame, theme: &Theme) {
        let w = match self.window(id) { Some(w) => w, None => return };
        if !w.visible { return; }
        if !self.paint_targets().contains(&id) { return; }
        let active = self.z_order.last() == Some(&id);
        paint_window_body(frame, w, theme, active);
    }

    /// Title-bar strip + icon + text + buttons for ONE window. Call
    /// AFTER painting that window's content so the title strip sits
    /// above its own body — but BEFORE the next window in z-order so
    /// the next window's body cleanly covers this one's title strip.
    pub fn paint_titlebar(&self, id: WindowId, frame: &mut Frame,
                          renderer: &mut Renderer, theme: &Theme) {
        let w = match self.window(id) { Some(w) => w, None => return };
        if !w.visible { return; }
        if !self.paint_targets().contains(&id) { return; }
        let active = self.z_order.last() == Some(&id);
        paint_window_titlebar(frame, renderer, w, theme, active);
    }

    /// Convenience wrapper for code that wants to paint every visible
    /// window in z-order using the per-id `paint_body`/`paint_titlebar`
    /// methods, without any content in between. Mostly useful for
    /// demos. The R2-Gui main loop interleaves content between body
    /// and titlebar calls to get proper z-order, so it does NOT use
    /// this helper.
    pub fn paint_chrome(&self, frame: &mut Frame, renderer: &mut Renderer, theme: &Theme) {
        let order: Vec<WindowId> = self.z_order.iter().copied().collect();
        for id in order {
            self.paint_body(id, frame, theme);
            self.paint_titlebar(id, frame, renderer, theme);
        }
    }
}

#[inline]
fn point_in(r: Rect, p: MousePos) -> bool {
    p.x >= r.x && p.x < r.x + r.w && p.y >= r.y && p.y < r.y + r.h
}

#[derive(Debug, Clone, Copy)]
enum ButtonHover { Min, Max, Close }

/// What part of a sub-window the cursor is over. Used to decide
/// click behavior (start a button press, drag-move, or drag-resize).
#[derive(Debug, Clone, Copy)]
enum HitZone {
    None,
    Button(ButtonHover),
    TitleBar,
    // 4 edges
    ResizeL, ResizeR, ResizeT, ResizeB,
    // 4 corners
    ResizeTL, ResizeTR, ResizeBL, ResizeBR,
    Body,
}

fn classify_hit(w: &SubWindow, pos: MousePos, theme: &Theme) -> HitZone {
    let tb = title_bar_height(theme);

    // Buttons (top-right of title bar) — checked first so they
    // override the title-bar drag region AND the top-right resize
    // corner that would otherwise overlap.
    let right = w.bounds.x + w.bounds.w - BTN_GAP;
    let btn_y = w.bounds.y + (tb - BTN_SIZE) * 0.5;
    let close_r = Rect { x: right - BTN_SIZE,                       y: btn_y, w: BTN_SIZE, h: BTN_SIZE };
    let max_r   = Rect { x: close_r.x - BTN_GAP - BTN_SIZE,         y: btn_y, w: BTN_SIZE, h: BTN_SIZE };
    let min_r   = Rect { x: max_r.x   - BTN_GAP - BTN_SIZE,         y: btn_y, w: BTN_SIZE, h: BTN_SIZE };
    if point_in(close_r, pos) { return HitZone::Button(ButtonHover::Close); }
    if point_in(max_r,   pos) { return HitZone::Button(ButtonHover::Max);   }
    if point_in(min_r,   pos) { return HitZone::Button(ButtonHover::Min);   }

    // Corner-resize takes priority over edge-resize.
    let in_tl = pos.x >= w.bounds.x                  && pos.x <  w.bounds.x + RESIZE_GRIP
             && pos.y >= w.bounds.y                  && pos.y <  w.bounds.y + RESIZE_GRIP;
    let in_tr = pos.x >= w.bounds.x + w.bounds.w - RESIZE_GRIP
             && pos.x <  w.bounds.x + w.bounds.w
             && pos.y >= w.bounds.y                  && pos.y <  w.bounds.y + RESIZE_GRIP;
    let in_bl = pos.x >= w.bounds.x                  && pos.x <  w.bounds.x + RESIZE_GRIP
             && pos.y >= w.bounds.y + w.bounds.h - RESIZE_GRIP
             && pos.y <  w.bounds.y + w.bounds.h;
    let in_br = pos.x >= w.bounds.x + w.bounds.w - RESIZE_GRIP
             && pos.x <  w.bounds.x + w.bounds.w
             && pos.y >= w.bounds.y + w.bounds.h - RESIZE_GRIP
             && pos.y <  w.bounds.y + w.bounds.h;
    if in_br { return HitZone::ResizeBR; }
    if in_bl { return HitZone::ResizeBL; }
    if in_tr { return HitZone::ResizeTR; }
    if in_tl { return HitZone::ResizeTL; }

    // Edges — thin grab strips along each side, excluding corners.
    let in_left   = pos.x < w.bounds.x + EDGE_GRAB
                 && pos.y >= w.bounds.y + RESIZE_GRIP
                 && pos.y <  w.bounds.y + w.bounds.h - RESIZE_GRIP;
    let in_right  = pos.x >= w.bounds.x + w.bounds.w - EDGE_GRAB
                 && pos.x <  w.bounds.x + w.bounds.w
                 && pos.y >= w.bounds.y + RESIZE_GRIP
                 && pos.y <  w.bounds.y + w.bounds.h - RESIZE_GRIP;
    let in_top    = pos.y < w.bounds.y + EDGE_GRAB
                 && pos.x >= w.bounds.x + RESIZE_GRIP
                 && pos.x <  w.bounds.x + w.bounds.w - RESIZE_GRIP;
    let in_bottom = pos.y >= w.bounds.y + w.bounds.h - EDGE_GRAB
                 && pos.y <  w.bounds.y + w.bounds.h
                 && pos.x >= w.bounds.x + RESIZE_GRIP
                 && pos.x <  w.bounds.x + w.bounds.w - RESIZE_GRIP;
    if in_right  { return HitZone::ResizeR; }
    if in_left   { return HitZone::ResizeL; }
    if in_bottom { return HitZone::ResizeB; }
    if in_top    { return HitZone::ResizeT; }

    // Title bar drag region — everything in the title bar that wasn't
    // a button or corner-resize zone.
    let titlebar = Rect { x: w.bounds.x, y: w.bounds.y, w: w.bounds.w, h: tb };
    if point_in(titlebar, pos) { return HitZone::TitleBar; }

    HitZone::Body
}

/// First pass — fill + border + resize grip. Title bar comes later.
fn paint_window_body(frame: &mut Frame, w: &SubWindow, theme: &Theme, is_active: bool) {
    let tb = title_bar_height(theme);
    let border = if is_active { theme.border_active } else { theme.border_passive };

    // Body fill.
    frame.paint_rect(w.bounds.x, w.bounds.y + tb,
                     w.bounds.w, w.bounds.h - tb, theme.window_background);

    // 1-px outer border in the active/passive color.
    frame.paint_rect(w.bounds.x, w.bounds.y,                w.bounds.w, 1.0, border);
    frame.paint_rect(w.bounds.x, w.bounds.y + w.bounds.h - 1.0, w.bounds.w, 1.0, border);
    frame.paint_rect(w.bounds.x, w.bounds.y, 1.0, w.bounds.h, border);
    frame.paint_rect(w.bounds.x + w.bounds.w - 1.0, w.bounds.y, 1.0, w.bounds.h, border);

    // Corner grips — small L-shaped marks at every corner so the
    // user sees that all four corners are draggable. Edges remain
    // grabbable (6 px strip) but unmarked to avoid visual noise.
    let corner_len = 8.0;
    let stripe     = 2.0;
    // BR
    let bx = w.bounds.x + w.bounds.w - corner_len - 2.0;
    let by = w.bounds.y + w.bounds.h - corner_len - 2.0;
    frame.paint_rect(bx, by + corner_len, corner_len + stripe, stripe, border);
    frame.paint_rect(bx + corner_len, by, stripe, corner_len + stripe, border);
    // BL
    let bx = w.bounds.x + 2.0;
    let by = w.bounds.y + w.bounds.h - corner_len - 2.0;
    frame.paint_rect(bx,                by + corner_len, corner_len + stripe, stripe, border);
    frame.paint_rect(bx,                by,              stripe, corner_len,           border);
    // TR
    let bx = w.bounds.x + w.bounds.w - corner_len - 2.0;
    let by = w.bounds.y + 2.0;
    frame.paint_rect(bx, by, corner_len + stripe, stripe, border);
    frame.paint_rect(bx + corner_len, by, stripe, corner_len, border);
    // TL
    let bx = w.bounds.x + 2.0;
    let by = w.bounds.y + 2.0;
    frame.paint_rect(bx, by, corner_len + stripe, stripe, border);
    frame.paint_rect(bx, by, stripe, corner_len + stripe, border);
}

/// Second pass — title bar strip + icon + text + buttons. Painted
/// after every window's body so back-window titles stay visible above
/// neighbour windows that overlap them.
fn paint_window_titlebar(frame: &mut Frame, renderer: &mut Renderer,
                         w: &SubWindow, theme: &Theme, is_active: bool) {
    let tb = title_bar_height(theme);
    let strip = if is_active { theme.titlebar_active } else { theme.titlebar_passive };
    let border = if is_active { theme.border_active } else { theme.border_passive };

    // Strip fill + bottom border.
    frame.paint_rect(w.bounds.x, w.bounds.y, w.bounds.w, tb, strip);
    frame.paint_rect(w.bounds.x, w.bounds.y + tb - 1.0, w.bounds.w, 1.0, border);

    // Icon + title text — identical regardless of active state. The
    // only signal of focus is the strip background color (two levels
    // of faint blue), which is enough to read at a glance without
    // making the passive window look broken.
    let _ = is_active;
    let icon_pad = 4.0;
    let icon_box = tb - 2.0 * icon_pad;
    let mut text_x = w.bounds.x + 10.0;
    if let Some(handle) = w.icon {
        let ix = w.bounds.x + icon_pad;
        let iy = w.bounds.y + icon_pad;
        frame.paint_image(handle, ix, iy, icon_box, icon_box, Color::WHITE);
        text_x = ix + icon_box + 6.0;
    }
    let baseline = w.bounds.y + tb * 0.78;
    frame.paint_text(renderer, text_x, baseline,
                     &w.title, theme.font_size, theme.menu_text);

    // Traffic-light buttons — full color always so users see the
    // same affordances on both windows. (Click still only acts on
    // the topmost window because the MDI host routes events that way.)
    let right = w.bounds.x + w.bounds.w - BTN_GAP;
    let btn_y = w.bounds.y + (tb - BTN_SIZE) * 0.5;
    let close = Rect { x: right - BTN_SIZE,                 y: btn_y, w: BTN_SIZE, h: BTN_SIZE };
    let max_  = Rect { x: close.x - BTN_GAP - BTN_SIZE,     y: btn_y, w: BTN_SIZE, h: BTN_SIZE };
    let min_  = Rect { x: max_.x  - BTN_GAP - BTN_SIZE,     y: btn_y, w: BTN_SIZE, h: BTN_SIZE };
    frame.paint_rect(min_.x,  min_.y,  min_.w,  min_.h,  theme.button_min);
    frame.paint_rect(max_.x,  max_.y,  max_.w,  max_.h,  theme.button_max);
    frame.paint_rect(close.x, close.y, close.w, close.h, theme.button_close);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Mods;

    fn mouse_down(x: f32, y: f32) -> InputEvent {
        InputEvent::MouseDown { button: MouseButton::Left, pos: MousePos { x, y } }
    }
    fn mouse_move(x: f32, y: f32) -> InputEvent {
        InputEvent::MouseMoved(MousePos { x, y })
    }
    fn mouse_up() -> InputEvent {
        InputEvent::MouseUp { button: MouseButton::Left, pos: MousePos { x: 0.0, y: 0.0 } }
    }
    fn theme() -> Theme { Theme::khaki() }

    #[test]
    fn drag_titlebar_moves_window() {
        let mut h = MdiHost::new();
        h.set_workspace(Rect { x: 0.0, y: 0.0, w: 1000.0, h: 800.0 });
        let id = h.add_window("W1", Rect { x: 100.0, y: 100.0, w: 400.0, h: 300.0 });
        // Click in the middle of the title bar (y=114 is below the
        // 6 px top-edge resize zone and inside the title strip).
        h.handle_events(&[mouse_down(200.0, 114.0), mouse_move(250.0, 144.0), mouse_up()], &theme());
        let w = h.window(id).unwrap();
        assert!((w.bounds.x - 150.0).abs() < 0.5);
        assert!((w.bounds.y - 130.0).abs() < 0.5);
    }

    #[test]
    fn close_button_sets_close_flag() {
        let mut h = MdiHost::new();
        h.set_workspace(Rect { x: 0.0, y: 0.0, w: 1000.0, h: 800.0 });
        let id = h.add_window("W1", Rect { x: 100.0, y: 100.0, w: 400.0, h: 300.0 });
        // Click at the far-right of title bar where the close button sits.
        let close_cx = 100.0 + 400.0 - BTN_GAP - BTN_SIZE * 0.5;
        let close_cy = 100.0 + title_bar_height(&theme()) * 0.5;
        h.handle_events(&[mouse_down(close_cx, close_cy), mouse_up()], &theme());
        assert!(h.take_close_requested(id));
        assert!(!h.take_close_requested(id), "flag is self-clearing");
    }

    #[test]
    fn click_brings_window_to_front() {
        let mut h = MdiHost::new();
        h.set_workspace(Rect { x: 0.0, y: 0.0, w: 1000.0, h: 800.0 });
        let a = h.add_window("A", Rect { x: 0.0,   y: 0.0,   w: 500.0, h: 400.0 });
        let b = h.add_window("B", Rect { x: 100.0, y: 100.0, w: 500.0, h: 400.0 });
        // Initial topmost is b. Click in A's exposed top-left corner.
        h.handle_events(&[mouse_down(20.0, 8.0), mouse_up()], &theme());
        // After the click, A should be topmost.
        let z: Vec<_> = h.z_order().collect();
        assert_eq!(z.last(), Some(&a));
        let _ = b; // silence
    }

    #[test]
    fn maximize_snaps_to_workspace_and_restore_returns() {
        let mut h = MdiHost::new();
        h.set_workspace(Rect { x: 0.0, y: 0.0, w: 1000.0, h: 800.0 });
        let id = h.add_window("W", Rect { x: 200.0, y: 150.0, w: 400.0, h: 300.0 });
        let orig = h.window(id).unwrap().bounds;
        h.toggle_maximize(id);
        let m = h.window(id).unwrap();
        assert!(m.maximized);
        assert!((m.bounds.w - 1000.0).abs() < 0.5);
        assert!((m.bounds.h - 800.0).abs() < 0.5);
        h.toggle_maximize(id);
        let r = h.window(id).unwrap();
        assert!(!r.maximized);
        assert!((r.bounds.x - orig.x).abs() < 0.5);
        assert!((r.bounds.w - orig.w).abs() < 0.5);
    }

    #[test]
    fn drag_resize_grip_resizes_window() {
        let mut h = MdiHost::new();
        h.set_workspace(Rect { x: 0.0, y: 0.0, w: 1000.0, h: 800.0 });
        let id = h.add_window("W", Rect { x: 100.0, y: 100.0, w: 400.0, h: 300.0 });
        // BR corner is at (500, 400) — grip is 14 px back from there.
        h.handle_events(&[mouse_down(495.0, 395.0), mouse_move(545.0, 425.0), mouse_up()], &theme());
        let w = h.window(id).unwrap();
        assert!((w.bounds.w - 450.0).abs() < 5.0);
        assert!((w.bounds.h - 330.0).abs() < 5.0);
        let _ = Mods::default();
    }
}
