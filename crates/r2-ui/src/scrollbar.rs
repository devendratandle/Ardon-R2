//! Scrollbar — vertical or horizontal track with a draggable thumb.
//!
//! Behaviorally faithful to standard desktop scrollbars: drag the
//! thumb to set position, click the track for a page jump. Hides
//! itself when the content fits the viewport. Stateful — owned by
//! the host (one vertical + one horizontal per scrollable area).
//!
//! ## Wiring
//!
//! Each frame the host:
//!   1. Computes `content_size` (total rows / max columns) and
//!      `viewport_size` (visible rows / visible columns).
//!   2. Sets `scrollbar.visible_fraction = viewport / content`.
//!   3. Calls `scrollbar.handle_events(events, track_rect)` and, if
//!      it returns `Some(new_position)`, maps that 0..1 value back
//!      to a row / column offset and writes it into
//!      `CellGridState::scroll_x` or `scroll_y_override`.
//!   4. Calls `scrollbar.paint(...)` AFTER content painting so the
//!      bar floats above the transcript.
//!
//! The widget knows nothing about cells or grids — it operates on
//! generic 0..1 fractions and pixel rects. Reusable for any
//! scrollable surface (Console transcript, future Help-pane, etc.).

use crate::event::{InputEvent, MouseButton, MousePos};
use crate::grid::Rect;
use crate::render::Frame;
use crate::theme::{Color, Theme};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollOrientation { Vertical, Horizontal }

/// Pixel thickness of the bar perpendicular to its scroll axis.
pub const SCROLLBAR_THICKNESS: f32 = 12.0;
/// Minimum thumb length so the user can always grab it on tiny
/// viewports.
const MIN_THUMB: f32 = 24.0;

pub struct Scrollbar {
    pub orientation: ScrollOrientation,
    /// 0.0 = scrolled to top / left edge, 1.0 = bottom / right edge.
    pub position: f32,
    /// Fraction of content visible in the viewport. >= 1.0 means
    /// content fits and the bar should hide.
    pub visible_fraction: f32,
    dragging: bool,
    drag_offset: f32,
    last_mouse: MousePos,
}

impl Scrollbar {
    pub fn new(orientation: ScrollOrientation) -> Self {
        Self {
            orientation,
            position: 0.0,
            visible_fraction: 1.0,
            dragging: false,
            drag_offset: 0.0,
            last_mouse: MousePos { x: 0.0, y: 0.0 },
        }
    }

    /// Whether to show the bar this frame. Content that fits in the
    /// viewport produces no scrollbar — matches every desktop app.
    pub fn is_visible(&self) -> bool { self.visible_fraction < 0.999 }

    /// Walk this frame's events. Returns `Some(new_position)` only
    /// on frames where the position actually moved, so the host can
    /// cheaply detect "did the user scroll?".
    pub fn handle_events(&mut self, events: &[InputEvent], track: Rect) -> Option<f32> {
        if !self.is_visible() { return None; }
        let prev = self.position;
        let thumb = self.thumb_rect(track);
        for ev in events {
            match *ev {
                InputEvent::MouseMoved(p) => {
                    self.last_mouse = p;
                    if self.dragging {
                        self.update_position_from_mouse(p, track);
                    }
                }
                InputEvent::MouseDown { button: MouseButton::Left, pos } => {
                    if point_in(thumb, pos) {
                        // Grab — remember where on the thumb we
                        // grabbed so the thumb doesn't jump.
                        self.dragging = true;
                        self.drag_offset = match self.orientation {
                            ScrollOrientation::Vertical   => pos.y - thumb.y,
                            ScrollOrientation::Horizontal => pos.x - thumb.x,
                        };
                    } else if point_in(track, pos) {
                        // Click on empty track — page jump. Centre
                        // the thumb on the click.
                        let len = self.thumb_length(track);
                        self.drag_offset = len * 0.5;
                        self.update_position_from_mouse(pos, track);
                    }
                }
                InputEvent::MouseUp { button: MouseButton::Left, .. } => {
                    self.dragging = false;
                }
                _ => {}
            }
        }
        if (self.position - prev).abs() > 1e-6 { Some(self.position) } else { None }
    }

    fn update_position_from_mouse(&mut self, pos: MousePos, track: Rect) {
        let thumb_len = self.thumb_length(track);
        let avail = match self.orientation {
            ScrollOrientation::Vertical   => (track.h - thumb_len).max(0.0),
            ScrollOrientation::Horizontal => (track.w - thumb_len).max(0.0),
        };
        if avail <= 0.0 { return; }
        let raw = match self.orientation {
            ScrollOrientation::Vertical   => (pos.y - track.y - self.drag_offset) / avail,
            ScrollOrientation::Horizontal => (pos.x - track.x - self.drag_offset) / avail,
        };
        self.position = raw.clamp(0.0, 1.0);
    }

    fn thumb_length(&self, track: Rect) -> f32 {
        let len = match self.orientation {
            ScrollOrientation::Vertical   => track.h,
            ScrollOrientation::Horizontal => track.w,
        };
        (len * self.visible_fraction).max(MIN_THUMB)
    }

    fn thumb_rect(&self, track: Rect) -> Rect {
        let len = self.thumb_length(track);
        match self.orientation {
            ScrollOrientation::Vertical => {
                let avail = (track.h - len).max(0.0);
                Rect { x: track.x, y: track.y + avail * self.position, w: track.w, h: len }
            }
            ScrollOrientation::Horizontal => {
                let avail = (track.w - len).max(0.0);
                Rect { x: track.x + avail * self.position, y: track.y, w: len, h: track.h }
            }
        }
    }

    /// Paint the track + thumb. Hover/drag thumb slightly darker for
    /// the standard desktop "I am grab-able" affordance.
    pub fn paint(&self, frame: &mut Frame, track: Rect, _theme: &Theme) {
        if !self.is_visible() { return; }
        // Track — faint dark wash so it reads as recessed.
        frame.paint_rect(track.x, track.y, track.w, track.h,
                         Color::rgba(0, 0, 0, 35));
        // Thumb — darker when interacted with.
        let thumb = self.thumb_rect(track);
        let active = self.dragging || point_in(thumb, self.last_mouse);
        let color = if active {
            Color::rgba(80, 100, 130, 230)
        } else {
            Color::rgba(120, 130, 150, 190)
        };
        frame.paint_rect(thumb.x, thumb.y, thumb.w, thumb.h, color);
    }
}

#[inline]
fn point_in(r: Rect, p: MousePos) -> bool {
    p.x >= r.x && p.x < r.x + r.w && p.y >= r.y && p.y < r.y + r.h
}
