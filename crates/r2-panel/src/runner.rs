//! The panel RUNNER — executes an action through the real Ardon-R2 engine
//! and classifies the result into a typed [`PanelResult`]. This is the
//! headless heart of the panel system: it needs no window, no GPU, and no
//! graphics device, so the whole button→result path is unit-testable on any
//! machine. The renderer (r2-gui) calls exactly this and only draws widgets
//! from what it returns.
//!
//! State persistence: one [`PanelSession`] owns one engine, so a "load
//! readings" button and a later "compute Cpk" button share variables —
//! exactly what an operator expects from a sequence of buttons.

use std::sync::Arc;

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

use crate::spec::{OutputKind, Panel, PanelApp};

/// Pass/fail grade for a `Value` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// `pass_if` evaluated true — show green.
    Pass,
    /// `pass_if` evaluated false — show red.
    Fail,
    /// No `pass_if` given — just a number.
    Neutral,
}

/// The typed outcome of pressing one button.
#[derive(Debug, Clone)]
pub enum PanelResult {
    /// A graded number. `unit` echoes the spec so the renderer can append it.
    Value { value: f64, unit: Option<String>, status: Status },
    /// A table: the engine's rendered form of a data.frame/matrix. (The
    /// skeleton passes the display text; a richer grid model is an Opus
    /// follow-up behind this same variant.)
    Table { display: String },
    /// A plot was produced. Headless runs can't rasterize, so we record the
    /// request; the GUI renderer swaps in the live graphics device here.
    Plot { note: String },
    /// Printed / free-form text output.
    Text { display: String },
    /// The script (or its pass_if) failed. Shown as an error banner, never
    /// a silent no-op — an operator must see that a task didn't run.
    Error { message: String },
}

/// A live panel session: one engine, reused across button presses so state
/// (loaded data, fitted models) persists between actions.
pub struct PanelSession {
    engine: Engine,
}

impl Default for PanelSession {
    fn default() -> Self { Self::new() }
}

impl PanelSession {
    pub fn new() -> Self {
        PanelSession { engine: Engine::new() }
    }

    /// Run every panel in an app once, in order (useful for a "run all" /
    /// headless report / test). Returns one result per button.
    pub fn run_app(&mut self, app: &PanelApp) -> Vec<(String, PanelResult)> {
        app.panels
            .iter()
            .map(|p| (p.label.clone(), self.run(p)))
            .collect()
    }

    /// Execute one action and classify its result per the panel's output kind.
    pub fn run(&mut self, panel: &Panel) -> PanelResult {
        let value = match self.eval_last(&panel.script) {
            Ok(v) => v,
            Err(e) => return PanelResult::Error { message: e },
        };

        match panel.output {
            OutputKind::Value => {
                let num = match scalar(&value) {
                    Some(n) => n,
                    None => return PanelResult::Error {
                        message: format!(
                            "action '{}' is declared `output: value` but produced a non-scalar {}",
                            panel.label, value.type_name()
                        ),
                    },
                };
                let status = match &panel.pass_if {
                    None => Status::Neutral,
                    Some(expr) => match self.eval_pass_if(num, expr) {
                        Ok(true) => Status::Pass,
                        Ok(false) => Status::Fail,
                        Err(e) => return PanelResult::Error {
                            message: format!("pass_if for '{}' failed: {}", panel.label, e),
                        },
                    },
                };
                PanelResult::Value { value: num, unit: panel.unit.clone(), status }
            }
            OutputKind::Table => PanelResult::Table { display: format!("{}", value) },
            OutputKind::Text => PanelResult::Text { display: format!("{}", value) },
            OutputKind::Plot => PanelResult::Plot {
                note: format!("plot produced by '{}'", panel.label),
            },
        }
    }

    /// Evaluate an R2 program, returning the value of its LAST statement.
    fn eval_last(&mut self, src: &str) -> Result<RVal, String> {
        let stmts = Parser::parse(src).map_err(|e| e.to_string())?;
        if stmts.is_empty() {
            return Err("empty script".into());
        }
        let mut last = RVal::Null;
        for st in &stmts {
            last = self.engine.eval(st).map_err(|e| e.msg.clone())?;
        }
        Ok(last)
    }

    /// Bind the action's numeric result as `x`, then evaluate the boolean
    /// `pass_if` expression against it. Binding goes straight into the
    /// global env (no fragile literal re-parsing), so `x` can be any value.
    fn eval_pass_if(&mut self, x: f64, expr: &str) -> Result<bool, String> {
        self.engine
            .global_env
            .set(Arc::from("x"), RVal::Numeric(vec![Some(x)].into(), Default::default()));
        let v = self.eval_last(expr)?;
        match v.as_logicals().map_err(|e| e.msg.clone())?.into_iter().next() {
            Some(Some(b)) => Ok(b),
            _ => Err("pass_if did not evaluate to TRUE/FALSE".into()),
        }
    }
}

/// Extract a scalar f64 from an RVal, or None if it isn't EXACTLY one
/// number. `scalar_f64` alone would silently take the first element of a
/// vector; a `value` panel must reject that so the operator sees an honest
/// error rather than a misleading single number.
fn scalar(v: &RVal) -> Option<f64> {
    if r2_types::rval_length(v) != 1 {
        return None;
    }
    v.scalar_f64().ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Panel;

    fn value_panel(script: &str, pass_if: Option<&str>) -> Panel {
        let mut p = Panel::new("test", script, OutputKind::Value);
        p.pass_if = pass_if.map(|s| s.to_string());
        p
    }

    #[test]
    fn value_pass_and_fail() {
        let mut s = PanelSession::new();
        // mean is 1.5 → passes ">= 1", fails ">= 2".
        match s.run(&value_panel("mean(c(1,2))", Some("x >= 1"))) {
            PanelResult::Value { value, status, .. } => {
                assert!((value - 1.5).abs() < 1e-12);
                assert_eq!(status, Status::Pass);
            }
            other => panic!("expected Value, got {:?}", other),
        }
        match s.run(&value_panel("mean(c(1,2))", Some("x >= 2"))) {
            PanelResult::Value { status, .. } => assert_eq!(status, Status::Fail),
            other => panic!("expected Value, got {:?}", other),
        }
    }

    #[test]
    fn value_without_pass_if_is_neutral() {
        let mut s = PanelSession::new();
        match s.run(&value_panel("sqrt(2)", None)) {
            PanelResult::Value { value, status, .. } => {
                assert!((value - 2f64.sqrt()).abs() < 1e-12);
                assert_eq!(status, Status::Neutral);
            }
            other => panic!("expected Value, got {:?}", other),
        }
    }

    #[test]
    fn state_persists_across_buttons() {
        // Button 1 loads data; button 2 uses it — same session.
        let mut s = PanelSession::new();
        let load = Panel::new("load", "readings <- c(10, 12, 14, 16)", OutputKind::Text);
        let _ = s.run(&load);
        match s.run(&value_panel("mean(readings)", Some("x >= 12"))) {
            PanelResult::Value { value, status, .. } => {
                assert!((value - 13.0).abs() < 1e-12);
                assert_eq!(status, Status::Pass);
            }
            other => panic!("expected Value, got {:?}", other),
        }
    }

    #[test]
    fn non_scalar_value_is_a_clear_error() {
        let mut s = PanelSession::new();
        match s.run(&value_panel("c(1,2,3)", None)) {
            PanelResult::Error { message } => assert!(message.contains("non-scalar")),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn script_error_surfaces_not_silently_swallowed() {
        let mut s = PanelSession::new();
        match s.run(&value_panel("no_such_function(1)", None)) {
            PanelResult::Error { .. } => {}
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn plot_and_table_and_text_kinds() {
        let mut s = PanelSession::new();
        match s.run(&Panel::new("p", "1+1", OutputKind::Plot)) {
            PanelResult::Plot { .. } => {}
            o => panic!("expected Plot, got {:?}", o),
        }
        match s.run(&Panel::new("t", "matrix(1:4, 2)", OutputKind::Table)) {
            PanelResult::Table { display } => assert!(!display.is_empty()),
            o => panic!("expected Table, got {:?}", o),
        }
        match s.run(&Panel::new("x", "summary(c(1,2,3,4))", OutputKind::Text)) {
            PanelResult::Text { display } => assert!(!display.is_empty()),
            o => panic!("expected Text, got {:?}", o),
        }
    }

    #[test]
    fn run_app_end_to_end() {
        let app = PanelApp::from_manifest("\
title: Demo QC
[panel]
label: Batch size
script: length(c(5,5,6,7))
output: value
pass_if: x >= 3

[panel]
label: Mean
script: mean(c(5,5,6,7))
output: value
").unwrap();
        let mut s = PanelSession::new();
        let results = s.run_app(&app);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "Batch size");
        match results[0].1 {
            PanelResult::Value { status: Status::Pass, .. } => {}
            ref o => panic!("expected Pass, got {:?}", o),
        }
    }
}
