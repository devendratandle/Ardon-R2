Ardon-R2 — Linux (x86_64)
=========================

This archive contains two binaries:

  R2Gui   the desktop GUI (RGui-style: console + graphics windows)
  r2      the command-line REPL / script runner

Running
-------
  ./R2Gui              launch the desktop GUI
  ./r2                 start the interactive CLI
  ./r2 myscript.r2     run a script

Requirements
------------
The GUI needs a graphical session (X11 or Wayland) and working OpenGL
drivers (mesa is fine). If R2Gui fails to start on a headless server,
use the CLI (./r2) instead — it has no display requirement.

Optional: install system-wide
------------------------------
  mkdir -p ~/.local/bin
  cp R2Gui r2 ~/.local/bin/           # put binaries on PATH
  cp ardon-r2.png ~/.local/share/icons/   # if present
  desktop-file-install --dir=~/.local/share/applications Ardon-R2.desktop

Ardon-R2 is pure Rust, AGPL-3.0. https://github.com/devendratandle/Ardon-R2
