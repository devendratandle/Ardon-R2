//! R2-UI · Phase 2 Week 3 visual smoke test.
//!
//! Opens a window and paints a synthetic transcript using
//! `paint_cells` — solid-rect selection band + glyph atlas in action.
//! Proves the Week 3 render foundation works end-to-end before we
//! wire mouse drag + keyboard selection on top.
//!
//! Run with:
//!   cargo run -p r2-ui --example cell_grid_demo

use r2_ui::{
    paint_cells, Cell, Color, GridPos, R2Ui, Rect, Selection, Theme,
};

fn main() -> Result<(), String> {
    // Build a small synthetic transcript: banner + a few input/output
    // pairs. Colors come from the active theme so it looks consistent
    // with the eventual real console widget.
    let theme = Theme::khaki();

    let banner = theme.console_banner;
    let input  = theme.console_input;
    let output = theme.console_output;
    let error  = theme.console_error;

    let lines: Vec<(&str, Color)> = vec![
        ("Ardon-R2 · R2-UI Week 3 demo",          banner),
        ("",                                       banner),
        ("R2> x <- rnorm(5, mean = 10, sd = 2)",   input),
        ("R2> print(x)",                           input),
        ("[1]  9.8243 10.4011 11.2087  8.7766 10.1330", output),
        ("R2> sqrt(-1)",                           input),
        ("[1] NaN",                                output),
        ("Warning: NaNs produced",                 error),
        ("R2> mean(x)",                            input),
        ("[1] 10.071",                             output),
    ];

    // Convert (string, color) pairs into rows of Cells.
    let rows: Vec<Vec<Cell>> = lines.iter()
        .map(|(s, col)| s.chars().map(|c| Cell::plain(c, *col)).collect())
        .collect();

    // Pin a sample selection on rows 4..=6 so we can see the band.
    let sample_selection = Some(Selection {
        start: GridPos { row: 4, col: 4  },
        end:   GridPos { row: 6, col: 8  },
    });

    R2Ui::app("R2-UI · Phase 2 Week 3 — CellGrid Demo")
        .theme(theme.clone())
        .initial_size(900, 600)
        .on_paint(move |renderer, frame, theme| {
            let size_pt = theme.font_size;
            let (cell_w, line_h) = renderer.cell_metrics(size_pt);
            // 16px inset from each edge so the grid breathes.
            let rect = Rect {
                x: 16.0, y: 16.0,
                w: renderer.size.width  as f32 - 32.0,
                h: renderer.size.height as f32 - 32.0,
            };
            paint_cells(frame, renderer, &rows, rect,
                        cell_w, line_h, size_pt, sample_selection, theme);
        })
        .run()
}
