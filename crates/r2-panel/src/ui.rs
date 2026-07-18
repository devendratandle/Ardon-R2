//! The app-framework widget tree — the HEADLESS core of Tier 2
//! (script-built apps, `app.*` builtins). See docs/APP_FRAMEWORK_DESIGN.md.
//!
//! Deliberately engine-independent: event handlers are stored as opaque
//! `CallbackId` slots (the engine wiring maps them to `RVal::Closure`s and
//! runs them via `Engine::call_fn`), so the entire tree — declaration,
//! layout, state, events, dirty-tracking — is unit-testable with no window,
//! no GPU, and no engine. The GUI renderer and the engine builtins are both
//! pure consumers of this model.
//!
//! Event model (locked by the design doc): single-threaded, run-to-
//! completion. The host drains `take_events()` one at a time, runs the
//! mapped closure on the engine thread, then repaints `take_dirty()`
//! widgets. No reactivity graph — callbacks explicitly `set` what changed.

use std::collections::HashMap;

/// Opaque handle to an engine-side callback (an `RVal::Closure` slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackId(pub u32);

/// A widget's current value in the state store. R-natural coercions
/// happen at the builtin boundary (`app.get`), not here.
#[derive(Debug, Clone, PartialEq)]
pub enum UiValue {
    Text(String),
    Number(f64),
    Bool(bool),
    /// Rendered table text for now (rich grid = renderer follow-up,
    /// same variant).
    Table(String),
    /// SVG captured from the widget's graphics device.
    Svg(String),
    None,
}

/// One widget. `id` is the state key; declaration order is layout order
/// (vertical stack; `Row`/`EndRow` group a horizontal run).
#[derive(Debug, Clone)]
pub enum Widget {
    Label  { id: String, text: String },
    Input  { id: String, label: String },
    Select { id: String, label: String, choices: Vec<String> },
    Check  { id: String, label: String },
    Button { id: String, text: String, on_click: Option<CallbackId> },
    /// Graded number chip; `pass_if` is an engine-side predicate closure.
    Value  { id: String, label: String, pass_if: Option<CallbackId> },
    Table  { id: String },
    Plot   { id: String, height: f32 },
    Status { id: String },
    Row,
    EndRow,
}

impl Widget {
    /// The state-store key, if this widget has one (Row/EndRow don't).
    pub fn id(&self) -> Option<&str> {
        match self {
            Widget::Label { id, .. } | Widget::Input { id, .. }
            | Widget::Select { id, .. } | Widget::Check { id, .. }
            | Widget::Button { id, .. } | Widget::Value { id, .. }
            | Widget::Table { id } | Widget::Plot { id, .. }
            | Widget::Status { id } => Some(id),
            Widget::Row | Widget::EndRow => None,
        }
    }
}

/// A queued UI event, produced by the renderer, consumed by the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiEvent {
    /// Button with this widget id was clicked; run its callback.
    Clicked { id: String, callback: CallbackId },
}

/// The whole app: widget tree + state + event queue + dirty list.
#[derive(Debug, Default)]
pub struct UiApp {
    pub title: String,
    widgets: Vec<Widget>,
    state: HashMap<String, UiValue>,
    events: Vec<UiEvent>,
    dirty: Vec<String>,
}

impl UiApp {
    pub fn new(title: impl Into<String>) -> Self {
        UiApp { title: title.into(), ..Default::default() }
    }

    /// Append a widget (declaration order = layout order). Seeds the
    /// state store so `get` before any interaction returns the initial
    /// value, and rejects duplicate ids — two widgets sharing state is
    /// always an authoring bug, better loud than weird.
    pub fn add(&mut self, w: Widget, initial: UiValue) -> Result<(), String> {
        if let Some(id) = w.id() {
            if self.state.contains_key(id) {
                return Err(format!("duplicate widget id '{}'", id));
            }
            self.state.insert(id.to_string(), initial);
        }
        self.widgets.push(w);
        Ok(())
    }

    pub fn widgets(&self) -> &[Widget] { &self.widgets }

    /// Current value of a widget (None if the id doesn't exist).
    pub fn get(&self, id: &str) -> Option<&UiValue> { self.state.get(id) }

    /// Update a widget's value and mark it dirty for repaint. Errors on
    /// unknown ids — a typo'd `app.set` must not silently do nothing.
    pub fn set(&mut self, id: &str, v: UiValue) -> Result<(), String> {
        match self.state.get_mut(id) {
            Some(slot) => {
                *slot = v;
                if !self.dirty.iter().any(|d| d == id) {
                    self.dirty.push(id.to_string());
                }
                Ok(())
            }
            None => Err(format!("no widget with id '{}'", id)),
        }
    }

    /// Renderer-side: record a click. (User edits to inputs/checks go
    /// straight through `set` — only actions queue events.)
    pub fn push_event(&mut self, ev: UiEvent) { self.events.push(ev); }

    /// Host-side: drain queued events (FIFO) to run their callbacks.
    pub fn take_events(&mut self) -> Vec<UiEvent> { std::mem::take(&mut self.events) }

    /// Host-side: drain the dirty ids to repaint after callbacks ran.
    pub fn take_dirty(&mut self) -> Vec<String> { std::mem::take(&mut self.dirty) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_app() -> UiApp {
        let mut a = UiApp::new("SPC");
        a.add(Widget::Input { id: "n".into(), label: "Sample size".into() },
              UiValue::Number(30.0)).unwrap();
        a.add(Widget::Button { id: "run".into(), text: "Compute".into(),
                               on_click: Some(CallbackId(1)) },
              UiValue::None).unwrap();
        a.add(Widget::Value { id: "cpk".into(), label: "Cpk".into(),
                              pass_if: Some(CallbackId(2)) },
              UiValue::None).unwrap();
        a
    }

    #[test]
    fn declaration_seeds_state_and_preserves_order() {
        let a = demo_app();
        assert_eq!(a.get("n"), Some(&UiValue::Number(30.0)));
        let ids: Vec<_> = a.widgets().iter().filter_map(|w| w.id()).collect();
        assert_eq!(ids, ["n", "run", "cpk"]);
    }

    #[test]
    fn duplicate_id_is_rejected() {
        let mut a = demo_app();
        let err = a.add(Widget::Label { id: "n".into(), text: "x".into() },
                        UiValue::None).unwrap_err();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn set_marks_dirty_once_and_get_reads_back() {
        let mut a = demo_app();
        a.set("cpk", UiValue::Number(1.41)).unwrap();
        a.set("cpk", UiValue::Number(1.52)).unwrap();
        assert_eq!(a.get("cpk"), Some(&UiValue::Number(1.52)));
        assert_eq!(a.take_dirty(), vec!["cpk".to_string()]); // deduped
        assert!(a.take_dirty().is_empty());                  // drained
    }

    #[test]
    fn unknown_id_set_is_a_loud_error() {
        let mut a = demo_app();
        assert!(a.set("typo", UiValue::Number(1.0)).is_err());
    }

    #[test]
    fn click_event_round_trip() {
        let mut a = demo_app();
        a.push_event(UiEvent::Clicked { id: "run".into(), callback: CallbackId(1) });
        let evs = a.take_events();
        assert_eq!(evs, vec![UiEvent::Clicked { id: "run".into(), callback: CallbackId(1) }]);
        assert!(a.take_events().is_empty());
    }

    #[test]
    fn full_interaction_cycle_headless() {
        // Simulate: user types 50 into `n`, clicks Run; the host runs the
        // callback (here: mocked as computing cpk from n) and repaints.
        let mut a = demo_app();
        a.set("n", UiValue::Number(50.0)).unwrap();          // user edit
        a.take_dirty();                                       // renderer repainted
        a.push_event(UiEvent::Clicked { id: "run".into(), callback: CallbackId(1) });
        for ev in a.take_events() {
            let UiEvent::Clicked { callback, .. } = ev;
            assert_eq!(callback, CallbackId(1));
            // (engine would call_fn the closure here; it calls app.set:)
            a.set("cpk", UiValue::Number(1.41)).unwrap();
        }
        assert_eq!(a.take_dirty(), vec!["cpk".to_string()]);
        assert_eq!(a.get("cpk"), Some(&UiValue::Number(1.41)));
    }
}
