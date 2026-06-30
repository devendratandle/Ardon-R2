//! Menu-bar and right-click context-menu construction for the GUI.
//! Pure builders moved out of main.rs so main() reads as window setup +
//! the event/paint loop. Action strings share one namespace with the
//! central dispatch in main's frame closure.

use std::cell::RefCell;
use std::rc::Rc;

use r2_ui::{ContextItem, ContextMenu, MenuBarState, MenuBuilder};

/// Console window menu — focused on the REPL workflow.
pub(crate) fn console_menu() -> Rc<RefCell<MenuBarState>> {
    let mut mb = MenuBuilder::new();
    mb.top("File")
        .item("Clear console", "",       "file.clear")
        .item("Quit",          "Ctrl+Q", "file.quit");
    mb.top("Edit")
        .item("Copy",          "Ctrl+C", "edit.copy")
        .item("Paste",         "Ctrl+V", "edit.paste")
        .item("Select all",    "Ctrl+A", "edit.select_all")
        .item("Settings…",     "",       "edit.settings");
    mb.top("Windows")
        .item("Show Console",  "", "win.console")
        .item("Show Graphics", "", "win.graphics");
    mb.top("Help")
        .item("About Ardon-R2", "", "help.about");
    Rc::new(RefCell::new(MenuBarState::new(mb.bar)))
}

/// Graphics window menu — viewer-only (no Paste; a plot pane is output).
pub(crate) fn graphics_menu() -> Rc<RefCell<MenuBarState>> {
    let mut mb = MenuBuilder::new();
    mb.top("File")
        .item("Save plot as SVG…", "Ctrl+S", "file.save_plot")
        .item("Save plot as PNG…", "",       "file.save_plot_png")
        .item("Copy plot as image","",       "file.copy_plot_image")
        .item("Copy plot SVG",     "",       "file.copy_plot")
        .item("Quit",              "Ctrl+Q", "file.quit");
    mb.top("Windows")
        .item("Show Console",      "",       "win.console")
        .item("Show Graphics",     "",       "win.graphics");
    mb.top("Help")
        .item("About Ardon-R2",    "",       "help.about");
    Rc::new(RefCell::new(MenuBarState::new(mb.bar)))
}

/// Console right-click context menu.
pub(crate) fn console_context() -> Rc<RefCell<ContextMenu>> {
    Rc::new(RefCell::new(ContextMenu::new(vec![
        ContextItem::new("Copy",       "edit.copy"),
        ContextItem::new("Paste",      "edit.paste"),
        ContextItem::new("Select all", "edit.select_all"),
        ContextItem::separator(),
        ContextItem::new("Clear console", "file.clear"),
    ])))
}

/// Graphics right-click context menu.
pub(crate) fn graphics_context() -> Rc<RefCell<ContextMenu>> {
    Rc::new(RefCell::new(ContextMenu::new(vec![
        ContextItem::new("Save plot as SVG…",   "file.save_plot"),
        ContextItem::new("Save plot as PNG…",   "file.save_plot_png"),
        ContextItem::separator(),
        ContextItem::new("Copy plot as image",  "file.copy_plot_image"),
        ContextItem::new("Copy plot SVG",       "file.copy_plot"),
    ])))
}
