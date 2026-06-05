# Ardon-R2 — working agreement for AI assistants

Ardon-R2 is a pure-Rust reimplementation of R (AGPL-3.0). Solo maintainer
(Devendra Tandale); no other contributors. These rules are **locked** —
follow them every session. The maintainer should not have to repeat them.

## 1. Git & push policy (READ THIS FIRST)

**Never push, fast-forward the remote, tag, or delete a remote branch
without the maintainer's explicit permission for that specific action.**
Asking "should I push?" and waiting is mandatory. Pushing without an OK is
a serious violation.

Separate the cheap act from the expensive one:

| Action | Frequency | When |
|---|---|---|
| **Local commit** | Often — free | Any logical checkpoint / WIP. No permission needed. |
| **Push to `main`** | Rare | Only for a **major feature** or a **version update**, *after* it is **tested and verified locally**, *and* with explicit maintainer permission. |
| **Tag / version bump / installer** | Rarest | Only at an accumulated **stable milestone**, only when the maintainer says so. |

Rationale: we work fast. Frequent pushes create noise, burn 3-platform CI,
and make the history hard to follow. Commit locally as much as you like;
push deliberately. CI is a *confirmation* of locally-green work, not a
discovery tool.

## 2. Branching

**Single branch: `main`.** No long-lived `dev`. Commit straight to `main`
as local checkpoints. For a genuinely risky change, use a *short-lived*
feature branch and merge back when green — created on demand, deleted
after. Releases are cut by tagging `main` (`v*` → `release.yml` builds the
Windows Inno installer + binaries).

## 3. Before any push (the local CI-equivalent)

Run and confirm green locally first:
- `cargo build` (workspace; CI excludes `r2-ui` / `r2-gui`)
- `cargo test` on at least the crates you touched
- Docs updated in the same change (CHANGELOG.md, README.md, FUNCTIONS.md,
  docs/ as relevant)
Present a short checklist, then **wait for the maintainer's OK** to push.

## 4. Commits

- End-state: only the maintainer's name in authorship. **Do not add a
  `Co-Authored-By: Claude` trailer.** AI involvement is disclosed in the
  README, not per-commit.
- Don't bump version strings unless explicitly told to.

## 5. Docs that are the source of truth

- `CHANGELOG.md` — authoritative release history.
- `docs/ARCHITECTURE.md` — current state + future phases only (completed
  phase narrative is archived under `code-history/`, gitignored).
- Keep `docs/ARCHITECTURE.md` honest: when a roadmap phase ships, move it
  from "next" to "shipped" in the same change.

## 6. Project shape (orientation)

~25 crates. Hot path: `r2-parser` → `r2-types` (RVal, the `Reals`
columnar storage, the `r2dterminal` output sink `r2_types::out`) →
`r2-engine` (builtins, eval, JIT call path) → domain crates (`r2-stats`,
`r2-ml`, `r2-linalg`, `r2-data`, `r2-graphics`) → `r2-kernel`
(Oracle-dispatched reduce/map/binary/par_for; Rayon lives here, never in
builtins) → `r2-arrow` (columnar buffers + memory-mapped out-of-core) →
`r2-oracle` (hardware-scaled serial/Rayon dispatch). Frontends:
`r2-repl` (CLI) and `r2-gui` install the output sink + graphics device.
