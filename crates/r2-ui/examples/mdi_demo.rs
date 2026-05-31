//! R2-UI · Phase 2 Week 4 milestone demo.
//!
//! RGui-style desktop: menu bar at the top, MDI workspace below,
//! two floating sub-windows ("R2 Console" + "R2 Graphics") with
//! their own title bars + traffic-light buttons + drag/resize/maximize.
//! Console window hosts the full ConsoleBuffer + CellGridState +
//! InputField loop from Week 3. Graphics window hosts a GraphPanel
//! pre-loaded with a hand-rolled sample SVG (a couple of axes + a
//! polyline) to prove the SVG → resvg → atlas → wgpu pipeline works.
//!
//! Run with:
//!   cargo run -p r2-ui --example mdi_demo

use std::cell::RefCell;
use std::rc::Rc;

use r2_console::{ConsoleBuffer, LineKind, SubmitAction};
use r2_ui::{
    Cell, CellGridState, Color, InputField,
    MdiHost, MenuBarState, MenuBuilder, GraphPanel,
    R2Ui, Rect, Theme, WindowId,
    MENU_BAR_HEIGHT,
};

const SAMPLE_SVG: &str = r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="400" height="300">
  <rect x="0" y="0" width="400" height="300" fill="white"/>
  <!-- axes -->
  <line x1="50" y1="250" x2="380" y2="250" stroke="black" stroke-width="1.5"/>
  <line x1="50" y1="250" x2="50"  y2="30"  stroke="black" stroke-width="1.5"/>
  <!-- gridlines -->
  <g stroke="#cccccc" stroke-width="0.5">
    <line x1="50" y1="200" x2="380" y2="200"/>
    <line x1="50" y1="150" x2="380" y2="150"/>
    <line x1="50" y1="100" x2="380" y2="100"/>
    <line x1="50" y1="50"  x2="380" y2="50"/>
    <line x1="120" y1="30" x2="120" y2="250"/>
    <line x1="190" y1="30" x2="190" y2="250"/>
    <line x1="260" y1="30" x2="260" y2="250"/>
    <line x1="330" y1="30" x2="330" y2="250"/>
  </g>
  <!-- sample sine-ish polyline -->
  <polyline fill="none" stroke="#205ca8" stroke-width="2"
            points="60,200 100,150 140,110 180,90 220,100 260,135 300,180 340,215 380,225"/>
  <!-- title -->
  <text x="200" y="20" text-anchor="middle"
        font-family="sans-serif" font-size="14" fill="#205ca8">
    Sample R2 Graphics output
  </text>
</svg>
"##;

fn line_color(theme: &Theme, kind: LineKind) -> Color {
    match kind {
        LineKind::Input | LineKind::Continuation => theme.console_input,
        LineKind::Output                          => theme.console_output,
        LineKind::Error                           => theme.console_error,
        LineKind::Banner                          => theme.console_banner,
    }
}

fn rows_from_buffer(buf: &ConsoleBuffer, theme: &Theme) -> Vec<Vec<Cell>> {
    buf.transcript().iter()
        .map(|cl| {
            let col = line_color(theme, cl.kind);
            cl.text.chars().map(|c| Cell::plain(c, col)).collect()
        })
        .collect()
}

fn main() -> Result<(), String> {
    let theme = Theme::khaki();

    // ── App state, all shared into the FnMut closure via Rc<RefCell>.
    let buffer = Rc::new(RefCell::new(ConsoleBuffer::new()));
    {
        let mut b = buffer.borrow_mut();
        b.push_banner("Ardon-R2 · R2-UI Week 4 — MDI desktop demo");
        b.push_banner("Drag titlebars to move. Drag the BR grip to resize.");
        b.push_banner("Min/Max/Close in the title bar work. Click File/Edit to open menus.");
        b.push_banner("");
    }

    let mdi = Rc::new(RefCell::new(MdiHost::new()));
    let console_id  = mdi.borrow_mut().add_window("R2 Console",
        Rect { x: 40.0, y: 60.0, w: 620.0, h: 380.0 });
    let graphics_id = mdi.borrow_mut().add_window("R2 Graphics",
        Rect { x: 680.0, y: 60.0, w: 460.0, h: 380.0 });

    let grid_state = Rc::new(RefCell::new(CellGridState::new()));
    let input      = Rc::new(RefCell::new(InputField::new()));

    let graph = Rc::new(RefCell::new(GraphPanel::new()));
    graph.borrow_mut().set_svg(SAMPLE_SVG.as_bytes().to_vec());

    // Menu bar.
    let mut mb = MenuBuilder::new();
    mb.top("File")
        .item("New plot",  "Ctrl+N", "file.new_plot")
        .item("Save plot", "Ctrl+S", "file.save_plot")
        .item("Quit",      "Ctrl+Q", "file.quit");
    mb.top("Edit")
        .item("Copy",      "Ctrl+C", "edit.copy")
        .item("Paste",     "Ctrl+V", "edit.paste")
        .item("Select all","Ctrl+A", "edit.select_all");
    mb.top("Windows")
        .item("R2 Console",  "", "win.console")
        .item("R2 Graphics", "", "win.graphics");
    mb.top("Help")
        .item("About R2-UI", "", "help.about");
    let menu_state = Rc::new(RefCell::new(MenuBarState::new(mb.bar)));

    let frame_counter = Rc::new(RefCell::new(0u64));

    R2Ui::app("R2-UI · Phase 2 Week 4 — MDI Desktop")
        .theme(theme.clone())
        .initial_size(1200, 760)
        .on_frame({
            let buffer        = buffer.clone();
            let mdi           = mdi.clone();
            let grid_state    = grid_state.clone();
            let input         = input.clone();
            let graph         = graph.clone();
            let menu_state    = menu_state.clone();
            let frame_counter = frame_counter.clone();
            move |ctx, renderer, frame, theme| {
                *frame_counter.borrow_mut() += 1;
                let win_w = renderer.size.width  as f32;
                let win_h = renderer.size.height as f32;

                // ── Workspace = full window minus menu bar.
                let menu_rect = Rect { x: 0.0, y: 0.0, w: win_w, h: MENU_BAR_HEIGHT };
                let workspace = Rect {
                    x: 0.0, y: MENU_BAR_HEIGHT,
                    w: win_w, h: win_h - MENU_BAR_HEIGHT,
                };
                mdi.borrow_mut().set_workspace(workspace);

                // ── Menu bar events + action dispatch.
                if let Some(action) = menu_state.borrow_mut().handle_events(
                    ctx.events, menu_rect, renderer, theme)
                {
                    match action.as_str() {
                        "edit.select_all" => {
                            let rows = rows_from_buffer(&buffer.borrow(), theme);
                            if !rows.is_empty() {
                                let last = rows.len() - 1;
                                let last_col = rows[last].len().saturating_sub(1);
                                grid_state.borrow_mut().selection = Some(r2_ui::Selection {
                                    start: r2_ui::GridPos { row: 0, col: 0 },
                                    end:   r2_ui::GridPos { row: last, col: last_col },
                                });
                            }
                        }
                        _ => {
                            buffer.borrow_mut().push_output(&format!("[menu action] {}", action));
                        }
                    }
                }

                // ── MDI events (drag/resize/close/maximize).
                mdi.borrow_mut().handle_events(ctx.events, theme);

                // ── Per-window event routing for content. We only route
                //     to widgets when their window is topmost & visible.
                let topmost = mdi.borrow().z_order().last();
                if topmost == Some(console_id) {
                    // InputField + transcript event handling.
                    let mut input_mut = input.borrow_mut();
                    let resp = input_mut.handle_events(ctx.events, ctx.clipboard);
                    if let Some(line) = resp.submitted {
                        let action = buffer.borrow_mut().submit_line(line.clone());
                        match action {
                            SubmitAction::Submit(src) => {
                                buffer.borrow_mut()
                                    .push_output(&format!("(would eval) {}", src.trim()));
                            }
                            SubmitAction::Continue | SubmitAction::Empty => {}
                        }
                    }
                    if resp.history_up {
                        if let Some(s) = buffer.borrow_mut().history_up()   { input_mut.set_line(s); }
                    }
                    if resp.history_down {
                        if let Some(s) = buffer.borrow_mut().history_down() { input_mut.set_line(s); }
                    }
                    input_mut.set_prompt(buffer.borrow().current_prompt());

                    let rows = rows_from_buffer(&buffer.borrow(), theme);
                    let (cell_w, line_h) = renderer.cell_metrics(theme.font_size);
                    let content = mdi.borrow().window(console_id).map(|w| w.content_rect(theme));
                    if let Some(content) = content {
                        let transcript_rect = Rect {
                            x: content.x + 8.0, y: content.y + 8.0,
                            w: content.w - 16.0,
                            h: content.h - line_h - 20.0,
                        };
                        let _copied = grid_state.borrow_mut().handle_events(
                            ctx.events, &rows, transcript_rect,
                            cell_w, line_h, ctx.clipboard,
                        );
                    }
                }

                // ── Paint background → menu → MDI chrome → window bodies.
                frame.paint_rect(workspace.x, workspace.y, workspace.w, workspace.h, theme.mdi_background);
                menu_state.borrow().paint(frame, renderer, menu_rect, theme);
                mdi.borrow().paint_chrome(frame, renderer, theme);

                // ── Window bodies in z-order so topmost paints last.
                let order: Vec<WindowId> = mdi.borrow().z_order().collect();
                for id in order {
                    let visible_content = mdi.borrow()
                        .window(id)
                        .filter(|w| w.visible)
                        .map(|w| w.content_rect(theme));
                    let content = match visible_content { Some(r) => r, None => continue };

                    if id == console_id {
                        let (cell_w, line_h) = renderer.cell_metrics(theme.font_size);
                        let transcript_rect = Rect {
                            x: content.x + 8.0, y: content.y + 8.0,
                            w: content.w - 16.0,
                            h: content.h - line_h - 20.0,
                        };
                        let prompt_rect = Rect {
                            x: content.x + 8.0,
                            y: transcript_rect.y + transcript_rect.h + 4.0,
                            w: content.w - 16.0,
                            h: line_h + 6.0,
                        };
                        let rows = rows_from_buffer(&buffer.borrow(), theme);
                        grid_state.borrow().paint(frame, renderer, &rows, transcript_rect,
                                                  cell_w, line_h, theme.font_size, theme);
                        // Prompt divider.
                        frame.paint_rect(prompt_rect.x, prompt_rect.y - 1.0,
                                         prompt_rect.w, 1.0,
                                         Color::rgba(40, 40, 40, 60));
                        let cursor_on = (*frame_counter.borrow() / 30) % 2 == 0;
                        input.borrow().paint(frame, renderer, prompt_rect,
                                             cell_w, line_h, theme.font_size, theme, cursor_on);
                    } else if id == graphics_id {
                        // Margin inside the panel.
                        let inner = Rect {
                            x: content.x + 8.0,  y: content.y + 8.0,
                            w: (content.w - 16.0).max(0.0),
                            h: (content.h - 16.0).max(0.0),
                        };
                        graph.borrow_mut().paint(frame, renderer, inner, theme);
                    }
                }

                // ── Close-button handling.
                if mdi.borrow_mut().take_close_requested(console_id) {
                    if let Some(w) = mdi.borrow_mut().window_mut(console_id) { w.visible = false; }
                }
                if mdi.borrow_mut().take_close_requested(graphics_id) {
                    if let Some(w) = mdi.borrow_mut().window_mut(graphics_id) { w.visible = false; }
                }
            }
        })
        .run()
}
