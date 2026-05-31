# Session B — Multi-device graphics (`dev.new()` etc.)

**Goal:** R-style multiple graphics windows. Every `dev.new()` opens a
fresh native sub-window backed by an independent SVG device.
`dev.list()` / `dev.set()` / `dev.off()` / `dev.cur()` work the same
as in R. The current device receives every subsequent `plot()`,
`hist()`, etc.

**Estimated size:** ~380 LoC across three crates, one focused
session. Behavior-cloning only — no R source copied.

---

## Architecture

```
   ┌────────────────────────────────────────────────────────┐
   │ r2-graphics::device                                     │
   │   DeviceId(u32)                                         │
   │   DeviceTable { devices: BTreeMap<DeviceId, PlotDev>,   │
   │                 current: Option<DeviceId>,              │
   │                 next_id: u32,                           │
   │                 pending_events: Vec<DeviceEvent> }      │
   │   thread_local DEVICE_TABLE: RefCell<DeviceTable>       │
   │                                                         │
   │   pub fn new_device() -> DeviceId                       │
   │   pub fn set_device(id) -> Option<DeviceId>             │
   │   pub fn close_device(id) -> Option<DeviceId>           │
   │   pub fn list_devices() -> Vec<DeviceId>                │
   │   pub fn current_device() -> Option<DeviceId>           │
   │   pub fn with_current(|d| ...)  ← used by every plot fn │
   │   pub fn drain_events() -> Vec<DeviceEvent>             │
   │                                                         │
   │   enum DeviceEvent {                                    │
   │     Created(DeviceId),  Closed(DeviceId),               │
   │     Plotted(DeviceId),  CurrentChanged(DeviceId),       │
   │   }                                                     │
   ├────────────────────────────────────────────────────────┤
   │ r2-engine                                               │
   │   bi_dev_new   → DeviceId as RVal::Integer              │
   │   bi_dev_set   → previous current id                    │
   │   bi_dev_off   → previous current id, or NULL           │
   │   bi_dev_list  → integer vector of open device ids      │
   │   bi_dev_cur   → integer scalar (current id or 0)       │
   ├────────────────────────────────────────────────────────┤
   │ r2-gui                                                  │
   │   active_devices: HashMap<DeviceId,(WindowId,GraphPanel)>│
   │   every frame: drain_events() and react:                │
   │     Created(id)  → MdiHost::add_window("R2 Graphics N") │
   │                   + new GraphPanel + assign logo icon   │
   │     Closed(id)   → MdiHost::window_mut(...).visible=false│
   │                   + remove from map                     │
   │     Plotted(id)  → fetch device's SVG, set on panel,    │
   │                   raise the window (z-order)            │
   │     CurrentChanged(id) → bring to front, no other change│
   └────────────────────────────────────────────────────────┘
```

---

## Step-by-step implementation order

### Step 1 — r2-graphics refactor (~150 LoC)

1. Define `DeviceId(u32)`. `Copy + Eq + Hash + Display`.
2. Build `DeviceTable` with the `devices` map, `current` pointer,
   `next_id` counter, and `pending_events` Vec.
3. Move the existing thread-local `DEVICE` (single `PlotDevice`) into
   `DEVICE_TABLE.devices[default_id]` — first call to any device fn
   lazily creates device id 1 to preserve old behaviour.
4. Rewrite `with_device(|d| ...)` to delegate to
   `with_current(|d| ...)`. Existing call-sites stay unchanged.
5. Add new public fns: `new_device`, `set_device`, `close_device`,
   `list_devices`, `current_device`, `drain_events`.
6. Each public fn pushes the right `DeviceEvent` so the GUI can
   observe state changes without polling.
7. Per-device temp file: `%TMP%/r2/dev_<id>.svg` (Win) or
   `$TMPDIR/r2/dev_<id>.svg` (POSIX). `save_to_file` already exists;
   re-use it inside `new_device` / `Plotted` event.

**Validation:** existing `bi_plot` and friends must still produce
the same file output. Run `cargo test -p r2-graphics` after; the
existing tests should pass without edits.

### Step 2 — r2-engine builtins (~80 LoC)

Add four R-faithful builtins in `r2-engine::lib`:

| Builtin | Signature | Returns |
|---|---|---|
| `dev.new()` | no args | `DeviceId` as `RVal::Integer` scalar |
| `dev.set(n)` | int id | previous current id (or NULL if none) |
| `dev.off()` / `dev.off(n)` | optional int id | previous current id |
| `dev.list()` | no args | `RVal::Integer` vector of open ids |
| `dev.cur()` | no args | `RVal::Integer` scalar — current id or 0 |

Wire each to the matching `r2-graphics` function. Print a small
"Created device <n>" / "Closed device <n>" via the OutputSink so the
CLI behaves the same as R.

### Step 3 — r2-gui wiring (~120 LoC)

In `r2-gui::main`:

1. Replace the single `graph: Rc<RefCell<GraphPanel>>` + `graphics_id`
   with `active_devices: Rc<RefCell<HashMap<DeviceId, (WindowId, GraphPanel)>>>`.
2. Each frame, after engine eval, call `r2_graphics::device::drain_events()`:
   * `Created(id)`: `MdiHost::add_window(format!("R2 Graphics — Dev {}", id.0), ...)`,
     new `GraphPanel`, assign logo icon, insert into map.
   * `Closed(id)`: hide / remove the window, drop from map.
   * `Plotted(id)`: get device's `full_svg()`, call panel.set_svg, bring
     window to front.
   * `CurrentChanged(id)`: bring its window to front, no other change.
3. Window close button (✕) routes to `r2_graphics::device::close_device(id)`
   so the engine learns about the close without the user calling
   `dev.off()`.
4. Replace `graphics_id` references in menu dispatch with "the current
   device's window id" — pulled from `current_device()`.

---

## Behavior notes (matches R)

- New devices spawn at offset positions so they don't all stack on
  top of each other. Use `40 * (id - 1) % 200` for x and y offsets.
- Window title shows the device id: `"R2 Graphics — Dev 2"`.
- Closing the last device leaves no current device; `dev.cur()` returns 0.
- Closing all devices does NOT terminate the GUI; the Graphics
  windows simply go away.
- `plot()` with no current device implicitly calls `dev.new()` first.

## Files touched

| Crate | File | Approx LoC |
|---|---|---|
| r2-graphics | `device.rs` | +150, -30 |
| r2-engine   | `lib.rs` (builtins section) | +80 |
| r2-gui      | `main.rs` | +120, -60 |

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Existing single-device behavior breaks | Step 1's `with_device` shim forwards to `with_current` so every existing `bi_plot`/`bi_hist`/overlay call works unchanged |
| Thread-local refactor exposes a data-race latent in current code | r2-graphics has always been thread-local — same property preserved |
| Event queue grows unbounded if GUI never drains | Cap `pending_events` at 256 entries, drop oldest |

## Acceptance test (R script)

```r
plot(1:10)              # opens device 1
dev.new()               # opens device 2 (separate window)
hist(rnorm(1000))       # goes into device 2
dev.set(1)              # bring device 1 to front
points(1:10, 1:10 + 1)  # adds to device 1
dev.list()              # → c(1L, 2L)
dev.off(2)              # closes device 2 window
dev.list()              # → c(1L)
dev.off()               # closes device 1 — no graphics windows now
```

All five lines must work end-to-end before session B is considered shipped.
