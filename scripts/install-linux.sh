#!/usr/bin/env bash
#
# Ardon-R2 — one-shot Linux installer (CLI `r2` + GUI `R2Gui`)
#
# Works on Linux Mint / Ubuntu / Debian and most apt-based distros.
# Installs prebuilt release binaries by default; can also build from source.
#
# Quick use:
#   chmod +x install-linux.sh
#   ./install-linux.sh              # install BOTH cli + gui (prebuilt)
#   ./install-linux.sh --cli        # CLI only
#   ./install-linux.sh --gui        # GUI only
#   ./install-linux.sh --source     # build from source instead of downloading
#   ./install-linux.sh --version v0.3.3
#   ./install-linux.sh --prefix "$HOME/.local"   # install without sudo
#
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
REPO="devendratandle/Ardon-R2"
PREFIX="${PREFIX:-/usr/local}"     # binaries go in $PREFIX/bin
COMPONENTS="both"                  # cli | gui | both
MODE="prebuilt"                    # prebuilt | source
VERSION=""                         # empty ⇒ latest release
FALLBACK_VERSION="v0.3.3"          # used if the GitHub API is unreachable

# ── Pretty output ─────────────────────────────────────────────────────
c_grn='\033[0;32m'; c_yel='\033[0;33m'; c_red='\033[0;31m'; c_dim='\033[2m'; c_off='\033[0m'
say()  { printf "${c_grn}==>${c_off} %s\n" "$*"; }
warn() { printf "${c_yel}warning:${c_off} %s\n" "$*" >&2; }
die()  { printf "${c_red}error:${c_off} %s\n" "$*" >&2; exit 1; }

usage() {
  sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ── Parse args ────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --cli)     COMPONENTS="cli" ;;
    --gui)     COMPONENTS="gui" ;;
    --both)    COMPONENTS="both" ;;
    --source)  MODE="source" ;;
    --prebuilt) MODE="prebuilt" ;;
    --version) VERSION="${2:-}"; shift ;;
    --prefix)  PREFIX="${2:-}"; shift ;;
    -h|--help) usage ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

need() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

# sudo only when we can't write to the target and aren't root
SUDO=""
if [ "$(id -u)" -ne 0 ] && [ ! -w "$PREFIX/bin" ] 2>/dev/null; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
fi

BIN_DIR="$PREFIX/bin"
APP_DIR="$PREFIX/lib/ardon-r2"     # GUI binary + assets live here

# ── apt dependency installation (GUI only) ────────────────────────────
apt_install() {
  command -v apt-get >/dev/null 2>&1 || {
    warn "apt-get not found — this looks like a non-Debian distro."
    warn "Install these libraries with your package manager, then re-run with --gui:"
    warn "  libxkbcommon libwayland libxcb libX11 libXrandr libXi libXcursor libGL libEGL"
    return 1
  }
  say "Installing GUI ${1} libraries via apt (needs your password)…"
  $SUDO apt-get update -qq
  # Install each package independently so one unavailable name on a given
  # Mint/Ubuntu release doesn't abort the whole batch.
  local ok=1
  for pkg in "$@"; do
    [ "$pkg" = "runtime" ] && continue
    [ "$pkg" = "build" ] && continue
    if ! $SUDO apt-get install -y "$pkg" >/dev/null 2>&1; then
      warn "could not install '$pkg' (name may differ on this release) — continuing"
      ok=0
    fi
  done
  [ "$ok" -eq 1 ] || warn "some GUI libs were skipped; if R2Gui fails to start, install the missing libGL/wayland/xcb runtime packages."
}

gui_runtime_deps() {
  apt_install runtime \
    libxkbcommon0 libwayland-client0 libwayland-cursor0 libwayland-egl1 \
    libxcb1 libx11-6 libxrandr2 libxi6 libxcursor1 libgl1 libegl1
}
gui_build_deps() {
  apt_install build \
    libxkbcommon-dev libwayland-dev libxcb1-dev libx11-dev \
    libxrandr-dev libxi-dev libxcursor-dev libgl1-mesa-dev
}

# ── Resolve the version to install ────────────────────────────────────
resolve_version() {
  [ -n "$VERSION" ] && return
  if command -v curl >/dev/null 2>&1; then
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
      | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')" || true
  fi
  [ -n "$VERSION" ] || { VERSION="$FALLBACK_VERSION"; warn "could not query latest release; using $VERSION"; }
}

install_bin() {  # install_bin <src-file> <dest-name>
  $SUDO install -D -m 755 "$1" "$BIN_DIR/$2"
  say "installed $2 → $BIN_DIR/$2"
}

# ── Prebuilt download path ────────────────────────────────────────────
download_extract() {  # download_extract <asset-name> <out-dir>
  local asset="$1" out="$2"
  local url="https://github.com/$REPO/releases/download/$VERSION/$asset"
  say "downloading $asset ($VERSION)…"
  curl -fSL --retry 3 -o "$out/$asset" "$url" \
    || die "download failed: $url (does the $VERSION release have this asset?)"
  tar -xzf "$out/$asset" -C "$out"
}

install_prebuilt() {
  need curl; need tar
  resolve_version
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

  if [ "$COMPONENTS" = "cli" ] || [ "$COMPONENTS" = "both" ]; then
    download_extract "r2-linux-x86_64.tar.gz" "$tmp"
    local r2; r2="$(find "$tmp" -type f -name r2 | head -n1)"
    [ -n "$r2" ] || die "r2 binary not found inside the CLI tarball"
    install_bin "$r2" "r2"
  fi

  if [ "$COMPONENTS" = "gui" ] || [ "$COMPONENTS" = "both" ]; then
    gui_runtime_deps
    download_extract "R2Gui-linux-x86_64.tar.gz" "$tmp"
    local gui; gui="$(find "$tmp" -type f -name R2Gui | head -n1)"
    [ -n "$gui" ] || die "R2Gui binary not found inside the GUI tarball"
    $SUDO install -d "$APP_DIR"
    $SUDO install -m 755 "$gui" "$APP_DIR/R2Gui"
    $SUDO ln -sf "$APP_DIR/R2Gui" "$BIN_DIR/R2Gui"
    say "installed R2Gui → $APP_DIR/R2Gui (symlinked into $BIN_DIR)"
    make_desktop_entry
  fi
}

# ── From-source path ──────────────────────────────────────────────────
install_source() {
  need git
  command -v cargo >/dev/null 2>&1 || die "Rust/cargo not found. Install it with:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source \$HOME/.cargo/env"
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  say "cloning $REPO…"
  git clone --depth 1 ${VERSION:+--branch "$VERSION"} "https://github.com/$REPO.git" "$tmp/src" \
    || git clone --depth 1 "https://github.com/$REPO.git" "$tmp/src"
  cd "$tmp/src"

  if [ "$COMPONENTS" = "cli" ] || [ "$COMPONENTS" = "both" ]; then
    say "building CLI (cargo build --release -p r2-repl --bin r2)…"
    cargo build --release -p r2-repl --bin r2
    install_bin "target/release/r2" "r2"
  fi
  if [ "$COMPONENTS" = "gui" ] || [ "$COMPONENTS" = "both" ]; then
    gui_build_deps
    say "building GUI (cargo build --release -p r2-gui)…"
    cargo build --release -p r2-gui
    $SUDO install -d "$APP_DIR"
    $SUDO install -m 755 "target/release/R2Gui" "$APP_DIR/R2Gui"
    $SUDO ln -sf "$APP_DIR/R2Gui" "$BIN_DIR/R2Gui"
    say "installed R2Gui → $APP_DIR/R2Gui (symlinked into $BIN_DIR)"
    make_desktop_entry
  fi
}

# ── Desktop launcher (so R2Gui appears in the app menu) ───────────────
make_desktop_entry() {
  local apps_dir="$PREFIX/share/applications"
  $SUDO install -d "$apps_dir"
  printf '%s\n' \
    "[Desktop Entry]" \
    "Type=Application" \
    "Name=Ardon-R2" \
    "Comment=Pure-Rust R — statistical computing GUI" \
    "Exec=$BIN_DIR/R2Gui" \
    "Terminal=false" \
    "Categories=Science;Education;Development;" \
    | $SUDO tee "$apps_dir/ardon-r2.desktop" >/dev/null
  say "added application-menu entry"
}

# ── Run ───────────────────────────────────────────────────────────────
say "Ardon-R2 installer — components=$COMPONENTS mode=$MODE prefix=$PREFIX"
case "$MODE" in
  prebuilt) install_prebuilt ;;
  source)   install_source ;;
esac

echo
say "Done. Verify with:"
case "$COMPONENTS" in
  cli)  printf "  ${c_dim}echo 'cat(mean(c(1,2,3)), \"\\\\n\")' | r2${c_off}\n" ;;
  gui)  printf "  ${c_dim}R2Gui${c_off}   (or launch 'Ardon-R2' from the menu)\n" ;;
  both) printf "  ${c_dim}echo 'cat(mean(c(1,2,3)), \"\\\\n\")' | r2${c_off}   and   ${c_dim}R2Gui${c_off}\n" ;;
esac
if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
  warn "$BIN_DIR is not on your PATH. Add this to ~/.bashrc:"
  printf "  ${c_dim}export PATH=\"%s:\$PATH\"${c_off}\n" "$BIN_DIR"
fi
