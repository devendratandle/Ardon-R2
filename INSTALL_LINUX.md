# Installing Ardon-R2 on Linux

Works on **Linux Mint**, Ubuntu, Debian, Pop!_OS, and most apt-based distros.
Ardon-R2 ships two programs — both install on Linux:

| Program | What it is | Needs system libraries? |
|---------|------------|--------------------------|
| `r2`    | the CLI / REPL (run scripts, interactive console) | **No** — single self-contained binary |
| `R2Gui` | the desktop GUI (windowed console, data grid, plots) | **Yes** — windowing/OpenGL runtime libs |

You have three ways to install. Pick one.

---

## Option 1 — One-shot script (easiest)

This downloads the prebuilt binaries, installs the GUI's system libraries for
you, puts both programs on your `PATH`, and adds **Ardon-R2** to your
application menu.

```bash
# Download the installer and run it (installs BOTH the CLI and the GUI)
curl -fsSL https://raw.githubusercontent.com/devendratandle/Ardon-R2/main/scripts/install-linux.sh -o install-linux.sh
chmod +x install-linux.sh
./install-linux.sh
```

That's it. Then verify:

```bash
echo 'cat(mean(c(1,2,3)), "\n")' | r2     # prints: 2
R2Gui                                       # opens the GUI window
```

### Script options

```bash
./install-linux.sh --cli                  # CLI only
./install-linux.sh --gui                  # GUI only
./install-linux.sh --source               # build from source instead of downloading
./install-linux.sh --version v0.3.3       # install a specific release
./install-linux.sh --prefix "$HOME/.local"   # install WITHOUT sudo (per-user)
```

> Installing into `/usr/local` (the default) asks for your `sudo` password.
> Use `--prefix "$HOME/.local"` for a password-free, per-user install — just
> make sure `~/.local/bin` is on your `PATH`.

---

## Option 2 — Prebuilt binaries by hand

If you'd rather not run a script.

### 2a. CLI (`r2`) — no dependencies

```bash
cd ~/Downloads
wget https://github.com/devendratandle/Ardon-R2/releases/download/v0.3.3/r2-linux-x86_64.tar.gz
tar -xzf r2-linux-x86_64.tar.gz
sudo install -m 755 r2 /usr/local/bin/r2
r2 --help    # or just:  r2
```

### 2b. GUI (`R2Gui`) — install runtime libraries first

```bash
# Windowing + OpenGL runtime libraries
sudo apt-get update
sudo apt-get install -y \
  libxkbcommon0 libwayland-client0 libwayland-cursor0 libwayland-egl1 \
  libxcb1 libx11-6 libxrandr2 libxi6 libxcursor1 libgl1 libegl1

# Download and install the GUI
wget https://github.com/devendratandle/Ardon-R2/releases/download/v0.3.3/R2Gui-linux-x86_64.tar.gz
tar -xzf R2Gui-linux-x86_64.tar.gz
sudo install -m 755 R2Gui /usr/local/bin/R2Gui
R2Gui
```

---

## Option 3 — Build from source

Use this if the prebuilt binary complains about glibc (older Mint releases),
or you want to build the latest `main`.

```bash
# 1. Install the Rust toolchain (Rust 1.70+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 2. Get the source
git clone https://github.com/devendratandle/Ardon-R2.git
cd Ardon-R2

# 3a. Build the CLI (no system deps)
cargo build --release -p r2-repl --bin r2
sudo install -m 755 target/release/r2 /usr/local/bin/r2

# 3b. Build the GUI (needs the -dev windowing packages)
sudo apt-get update
sudo apt-get install -y \
  libxkbcommon-dev libwayland-dev libxcb1-dev libx11-dev \
  libxrandr-dev libxi-dev libxcursor-dev libgl1-mesa-dev
cargo build --release -p r2-gui
sudo install -m 755 target/release/R2Gui /usr/local/bin/R2Gui
```

---

## Verify the install

```bash
# CLI
echo 'x <- c(1,2,3,4,5); cat("mean:", mean(x), " sd:", sd(x), "\n")' | r2

# Run a script file
printf 'cat("hello from Ardon-R2\\n")\n' > hello.R
r2 hello.R

# GUI
R2Gui
```

---

## Uninstall

```bash
sudo rm -f /usr/local/bin/r2 /usr/local/bin/R2Gui
sudo rm -rf /usr/local/lib/ardon-r2
sudo rm -f /usr/local/share/applications/ardon-r2.desktop
# (adjust the prefix if you installed with --prefix "$HOME/.local")
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `r2: command not found` after install | Your bin dir isn't on `PATH`. Add `export PATH="/usr/local/bin:$PATH"` (or `~/.local/bin`) to `~/.bashrc`, then `source ~/.bashrc`. |
| `version `GLIBC_2.xx' not found` | The prebuilt binary is newer than your distro's glibc (older Mint). **Build from source** (Option 3) — it always matches your system. |
| `R2Gui` exits with an `libGL`/`libxkbcommon`/`wayland` error | A GUI runtime library is missing. Re-run the apt command in **Option 2b** (or `./install-linux.sh --gui`). |
| GUI is black / won't render over SSH | The GUI needs a real display (X11/Wayland session). It won't run headless. |
| A package name in the apt list "has no installation candidate" | Package names vary slightly across Mint releases. The one-shot script skips missing names automatically; by hand, drop the offending package and install the rest, or build from source. |

> **Note:** prebuilt binaries are built on `ubuntu-latest`. If you're on an
> older Linux Mint (20.x / Ubuntu 20.04 base) and hit a glibc error, Option 3
> (build from source) is the reliable path.
