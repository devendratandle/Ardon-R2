Ardon-R2 — macOS (Apple Silicon / arm64)
=========================================

Ardon-R2.app   the desktop GUI (drag into /Applications if you like)
r2             the command-line REPL (inside the .app at
               Ardon-R2.app/Contents/MacOS/r2, also copied here)

First launch: Gatekeeper
------------------------
This build is NOT code-signed or notarized (no paid Apple Developer
ID yet). macOS will refuse the first launch with a security warning.
To open it anyway:

  • Right-click (or Control-click) Ardon-R2.app -> Open -> Open, OR
  • Run once in Terminal:
        xattr -dr com.apple.quarantine Ardon-R2.app

After the first approval it opens normally like any other app.

CLI
---
  ./r2                  start the interactive REPL
  ./r2 myscript.r2      run a script

Ardon-R2 is pure Rust, AGPL-3.0. https://github.com/devendratandle/Ardon-R2
