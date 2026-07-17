//! # r2-panel — declarative menu/dashboard apps for Ardon-R2
//!
//! Lets someone use Ardon-R2 **without learning it**. A domain expert (QC
//! engineer, lab lead) authors a small manifest once — buttons, each running
//! a canned R2 task and showing a typed result — and an operator just presses
//! buttons: "Daily Cpk" → `1.41 ✓`, "Control Chart" → a plot, "Defect
//! Summary" → a table. This is the deployment surface that turns Ardon-R2
//! from a tool-for-users into a tool-for-a-desk.
//!
//! ## Layering (why this crate is thin and headless)
//!
//! ```text
//!   manifest (.panel)  ──parse──▶  PanelApp  (spec.rs — the stable contract)
//!         PanelSession.run(panel) ─────────▶ PanelResult   (runner.rs)
//!                                   │  runs the REAL engine, classifies output
//!   r2-gui (over r2-ui)  ◀──draws──┘  Value/Table/Plot/Text/Error widgets
//! ```
//!
//! The model + runner live here and are **fully testable with no window and
//! no GPU** (that is the Ardon-R2 foundation discipline: prove the seam
//! headless, hand the rendering breadth to the GUI). The renderer is a pure
//! consumer of [`runner::PanelResult`]; adding a widget never touches the
//! engine, and adding an engine capability never touches the renderer.
//!
//! See `docs/PANEL_ARCHITECTURE.md` for the full design and the Opus
//! follow-up queue (rich table grid, live plot device binding, TOML loader,
//! input controls).

pub mod runner;
pub mod spec;

pub use runner::{PanelResult, PanelSession, Status};
pub use spec::{OutputKind, Panel, PanelApp};
