# R2 IDE protocol (`r2 --ide`), version 1

How a separate IDE uses Ardon-R2 as its engine — the way RStudio drives an
R session. The IDE is its own program and links no R2 code: it starts the
installed `r2` executable with `--ide` and talks to it over the child
process's stdin and stdout, **one JSON object per line** in each direction
(UTF-8, `\n`-terminated). R2 does all statistics, formatting and plotting;
the IDE displays what comes back.

Source: `crates/r2-repl/src/ide/`. Tests: `crates/r2-repl/tests/ide_protocol.rs`.

## 1. Requests and replies

Every request is an object with `"op"` and (except `interrupt`) a numeric
`"id"`. Every reply echoes the id and has `"ok"`:

```json
→ {"id":7,"op":"value","code":"sd(trial$score, na.rm = TRUE)"}
← {"id":7,"ok":true,"value":{"type":"numeric","length":1,"values":[7.1240805405141865]}}
```

Requests run one at a time, in order. While an `eval` runs, R2 also writes
**stream lines** for it (§2.2) before its reply. `"ok":false` comes with an
`"error"` message.

## 2. Operations

### 2.1 `hello`
`{"id":1,"op":"hello"}` → `{"id":1,"ok":true,"r2":"0.4.2","protocol":1}`.
The IDE should send it first and check `protocol` (and `r2` against its minimum).

### 2.2 `eval` — console input
`{"id":2,"op":"eval","code":"mean(x); hist(x)"}`

Runs `code` as the console does: each top-level statement in turn, visible
values printed (R's auto-print rule), stopping at the first error. Output is
sent **as it is printed**, one line per chunk:

```json
← {"id":2,"stream":"out","text":"[1] 80.01144\n"}
← {"id":2,"stream":"err","text":"Error: object 'y' not found\n"}
```

then the reply:

| field | meaning |
|---|---|
| `"incomplete":true` | the input is unfinished (open bracket, string or `{`) — nothing ran; the IDE shows a `+` prompt and sends the whole input again once completed |
| `"error"` | a statement failed |
| `"interrupted"` | stopped by `interrupt` |
| `"plot"` | present only when the current plot changed: the whole plot as an SVG document |

Warnings arrive on the `err` stream after the output. Plots are never
written to files or opened in a browser in this mode.

### 2.3 `value` — a value, quietly
`{"id":3,"op":"value","code":"hist(trial$score, plot = FALSE)"}` →
`{"id":3,"ok":true,"value":{…}}` (§4). Nothing is printed and nothing goes
to the console. Used for everything the IDE shows from data (Sakha's cards
and statistics): the IDE asks R2's own functions and computes nothing itself.

### 2.4 `env` — the Environment pane
`{"id":4,"op":"env"}` →
`{"id":4,"ok":true,"rows":[{"name":"trial","group":"Data","summary":"120 obs. of 9 variables","viewable":true}, …]}`

`group` is `Data`, `Values` or `Functions`; rows are sorted by group, then
name; names starting with `.` are hidden. `viewable` means `view` can show it.

### 2.5 `view` — a page of the data viewer
`{"id":5,"op":"view","code":"trial","offset":0,"limit":200}` →

```json
{"id":5,"ok":true,"shape":"120 rows × 9 columns","nrow":120,"ncol":9,
 "headers":["id","score",…],"types":["chr","num",…],"widths":[4,5,…],
 "right":[false,true,…],"offset":0,
 "rows":[{"label":"1","cells":["P001","81.02",…]}, …]}
```

Cells are formatted as `print()` formats the column (one common number
format per numeric column), cut to `widths` characters. `limit` defaults to 200.

### 2.6 `interrupt`
`{"op":"interrupt"}` — no id, no reply. Read on its own thread, so it works
while an `eval` runs: the running code stops at the next safe point and its
reply says `"interrupted":true`. Variables bound before the stop remain.

### 2.7 `quit`
`{"id":9,"op":"quit"}` → `{"id":9,"ok":true}`, then R2 exits. Closing
stdin also ends the session.

## 3. Lifetime

One `r2 --ide` process is one R2 session (one global environment). The IDE
starts it in the working folder (a project's folder), restarts it if it
exits unexpectedly, and may run several for several sessions.

## 4. Values

Each value is an object with `"type"`:

| type | fields |
|---|---|
| `NULL` | — |
| `numeric`, `integer` | `length`, `values` (numbers), `names` if named |
| `logical` | `length`, `values` (`true`/`false`) |
| `character` | `length`, `values` (strings), `names` if named |
| `factor` | `length`, `levels`, `codes` (1-based) |
| `list` | `length`, `items`: `[{"name":…,"value":…}]` (`name` may be null) |
| `object` | `class` (e.g. `"histogram"`), `fields`: `{name: value}` |
| `data.frame` | `nrow`, `ncol`, `columns`: `[{"name":…,"value":…}]` (values only while nrow × ncol ≤ 100 000) |
| `matrix` | `nrow`, `ncol`, `data` (column-major) |
| other | `text`: the printed form |

Missing values: NA is `null`; NaN is `"NaN"`; infinities are `"Inf"` /
`"-Inf"`. Numbers keep full double precision. A vector longer than 100 000
elements is cut and marked `"truncated":true` (its `length` is the full
length) — for large data, ask for a summary or a `view` page instead.

## 5. Compatibility

`protocol` is bumped only for breaking changes. New operations and new
reply fields may be added within a version; an IDE ignores fields it does
not know. `r2 --json` (one expression per line, display text back) is a
separate, unchanged mode.
