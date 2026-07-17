//! The panel APP SPEC — the declarative data model. A panel app is a
//! title plus an ordered list of actions ("panels"). Each action is a
//! labelled button that runs one R2 script and shows its result in a typed
//! way. This is the whole contract a front-end (r2-gui over r2-ui) renders,
//! and the whole contract a QC engineer authors — no R2 fluency required to
//! PRESS the buttons, only to author the manifest once.
//!
//! Design intent (docs/PANEL_ARCHITECTURE.md): keep the model tiny and
//! declarative. The manifest says WHAT to run and HOW to display it; the
//! engine does the computing; the renderer draws widgets. Adding a new
//! output widget is a new `OutputKind` variant + a render arm — it never
//! touches the engine.

/// How an action's result should be presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    /// A single number, optionally graded pass/fail against `pass_if`.
    /// The shop-floor case: "Cpk = 1.41  ✓ (≥ 1.33)".
    Value,
    /// A tabular result (data.frame / matrix) rendered as a grid.
    Table,
    /// A plot produced by the script (control chart, histogram). The
    /// renderer hands the graphics device to r2-graphics; headless runs
    /// just record that a plot was requested.
    Plot,
    /// Free-form printed text (summary(), cat output).
    Text,
}

impl OutputKind {
    pub fn parse(s: &str) -> Option<OutputKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "value" | "number" | "scalar" => Some(OutputKind::Value),
            "table" | "grid" | "data.frame" | "dataframe" => Some(OutputKind::Table),
            "plot" | "chart" | "graph" => Some(OutputKind::Plot),
            "text" | "print" | "summary" => Some(OutputKind::Text),
            _ => None,
        }
    }
}

/// One button on the panel.
#[derive(Debug, Clone)]
pub struct Panel {
    /// Button caption the operator sees.
    pub label: String,
    /// R2 source run when the button is pressed. May be several statements;
    /// the LAST expression's value is the result.
    pub script: String,
    /// How to display the result.
    pub output: OutputKind,
    /// For `Value` outputs: a boolean R2 expression in `x` (the result),
    /// e.g. `x >= 1.33`. True → PASS (green), false → FAIL (red). Absent →
    /// neutral (just show the number).
    pub pass_if: Option<String>,
    /// Optional unit label shown after the number ("Cpk", "ppm", "%").
    pub unit: Option<String>,
    /// Optional one-line help shown under the button / as a tooltip.
    pub description: Option<String>,
}

impl Panel {
    pub fn new(label: impl Into<String>, script: impl Into<String>, output: OutputKind) -> Self {
        Panel {
            label: label.into(),
            script: script.into(),
            output,
            pass_if: None,
            unit: None,
            description: None,
        }
    }
}

/// A whole panel application: a title and its buttons.
#[derive(Debug, Clone, Default)]
pub struct PanelApp {
    pub title: String,
    pub panels: Vec<Panel>,
}

impl PanelApp {
    pub fn new(title: impl Into<String>) -> Self {
        PanelApp { title: title.into(), panels: Vec::new() }
    }

    pub fn with(mut self, panel: Panel) -> Self {
        self.panels.push(panel);
        self
    }

    /// Parse a panel app from the tiny declarative manifest format. Zero
    /// external deps by design (the workspace pulls in no TOML/serde) — a
    /// deliberately small line format that a factory engineer can hand-edit:
    ///
    /// ```text
    /// title: Line 3 Quality Control
    ///
    /// [panel]
    /// label: Daily Cpk
    /// script: cpk(readings)
    /// output: value
    /// pass_if: x >= 1.33
    /// unit: Cpk
    ///
    /// [panel]
    /// label: Control Chart
    /// script: qcc_plot(readings)
    /// output: plot
    /// ```
    ///
    /// Rules: `key: value` lines; a `[panel]` header starts a new button;
    /// lines before the first `[panel]` set app-level keys (`title`); blank
    /// lines and `#`-comments are ignored. `script:` may span multiple lines
    /// by indenting continuation lines. Swapping this for full TOML later is
    /// a loader change only — the model above is the stable contract.
    pub fn from_manifest(src: &str) -> Result<PanelApp, String> {
        let mut app = PanelApp::default();
        let mut cur: Option<Panel> = None;
        // Track the last key written so indented continuation lines append.
        let mut last_key: Option<String> = None;

        let flush = |app: &mut PanelApp, cur: &mut Option<Panel>| -> Result<(), String> {
            if let Some(p) = cur.take() {
                if p.label.is_empty() { return Err("panel missing `label`".into()); }
                if p.script.is_empty() { return Err(format!("panel '{}' missing `script`", p.label)); }
                app.panels.push(p);
            }
            Ok(())
        };

        for (lineno, raw) in src.lines().enumerate() {
            let line = raw.trim_end();
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') { continue; }

            // Indented continuation of the previous key (multi-line scripts).
            let indented = line.starts_with(' ') || line.starts_with('\t');
            if indented && last_key.is_some() && !trimmed.starts_with('[') {
                if let (Some(key), Some(p)) = (&last_key, cur.as_mut()) {
                    if key == "script" {
                        p.script.push('\n');
                        p.script.push_str(trimmed);
                        continue;
                    }
                }
            }

            if trimmed == "[panel]" {
                flush(&mut app, &mut cur)?;
                cur = Some(Panel::new("", "", OutputKind::Text));
                last_key = None;
                continue;
            }

            let (key, val) = trimmed
                .split_once(':')
                .ok_or_else(|| format!("line {}: expected `key: value`, got `{}`", lineno + 1, trimmed))?;
            let key = key.trim().to_ascii_lowercase();
            let val = val.trim().to_string();
            last_key = Some(key.clone());

            match cur.as_mut() {
                None => {
                    // App-level keys (before any [panel]).
                    match key.as_str() {
                        "title" => app.title = val,
                        _ => return Err(format!("line {}: unknown app key `{}`", lineno + 1, key)),
                    }
                }
                Some(p) => match key.as_str() {
                    "label" => p.label = val,
                    "script" => p.script = val,
                    "output" => {
                        p.output = OutputKind::parse(&val)
                            .ok_or_else(|| format!("line {}: unknown output kind `{}`", lineno + 1, val))?;
                    }
                    "pass_if" | "pass" => p.pass_if = Some(val),
                    "unit" => p.unit = Some(val),
                    "description" | "help" => p.description = Some(val),
                    _ => return Err(format!("line {}: unknown panel key `{}`", lineno + 1, key)),
                },
            }
        }
        flush(&mut app, &mut cur)?;
        if app.panels.is_empty() {
            return Err("manifest defines no [panel] blocks".into());
        }
        Ok(app)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_qc_manifest() {
        let src = "\
# Shop-floor QC panel
title: Line 3 Quality Control

[panel]
label: Daily Cpk
script: cpk(readings)
output: value
pass_if: x >= 1.33
unit: Cpk

[panel]
label: Control Chart
script: qcc_plot(readings)
output: plot
description: X-bar chart of today's readings
";
        let app = PanelApp::from_manifest(src).unwrap();
        assert_eq!(app.title, "Line 3 Quality Control");
        assert_eq!(app.panels.len(), 2);
        assert_eq!(app.panels[0].label, "Daily Cpk");
        assert_eq!(app.panels[0].output, OutputKind::Value);
        assert_eq!(app.panels[0].pass_if.as_deref(), Some("x >= 1.33"));
        assert_eq!(app.panels[0].unit.as_deref(), Some("Cpk"));
        assert_eq!(app.panels[1].output, OutputKind::Plot);
        assert_eq!(app.panels[1].description.as_deref(), Some("X-bar chart of today's readings"));
    }

    #[test]
    fn multi_line_script_continuation() {
        let src = "\
title: T
[panel]
label: Multi
script: x <- c(1,2,3)
    mean(x)
output: value
";
        let app = PanelApp::from_manifest(src).unwrap();
        assert_eq!(app.panels[0].script, "x <- c(1,2,3)\nmean(x)");
    }

    #[test]
    fn rejects_panel_without_script() {
        let src = "title: T\n[panel]\nlabel: Broken\noutput: value\n";
        assert!(PanelApp::from_manifest(src).is_err());
    }

    #[test]
    fn rejects_empty_manifest() {
        assert!(PanelApp::from_manifest("title: nothing here\n").is_err());
    }

    #[test]
    fn builder_api() {
        let app = PanelApp::new("Demo")
            .with(Panel::new("Sum", "sum(1:10)", OutputKind::Value));
        assert_eq!(app.title, "Demo");
        assert_eq!(app.panels.len(), 1);
    }
}
