//! The R2 datastore core — append-only segment store over immutable files.
//! See docs/DATASTORE_DESIGN.md. This module is the HEADLESS foundation:
//! segment naming, per-node sequence recovery, union listing, schema
//! fingerprinting, and sync diffing — everything except the actual column
//! codec (parquet lives in r2-arrow; the `db.*` builtins wire the two).
//!
//! The invariant every function here protects: **segments are immutable
//! once named** — a table is a grow-only set of files, so replica merge is
//! file-set union and no conflict is possible. Writes are tmp+rename so a
//! crash can only leave an invisible `.tmp` orphan, never a torn segment.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

pub const SEGMENT_EXT: &str = "parquet";

/// A store handle: the root directory + this writer's node identity and
/// recovered per-node sequence counter. Cheap; holds no locks, no daemon.
#[derive(Debug)]
pub struct Store {
    pub root: PathBuf,
    pub node: String,
    /// Next sequence number for THIS node (max existing + 1, per table —
    /// tracked lazily at write time).
    _priv: (),
}

/// Sanitize a node name: lowercase alnum and '-' only, non-empty.
pub fn sanitize_node(raw: &str) -> String {
    let s: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { "node".into() } else { s }
}

impl Store {
    /// Open (create-if-missing) a store rooted at `root`.
    pub fn open(root: impl Into<PathBuf>, node: &str) -> Result<Store, String> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| format!("db.open: cannot create {}: {}", root.display(), e))?;
        Ok(Store { root, node: sanitize_node(node), _priv: () })
    }

    fn table_dir(&self, table: &str) -> PathBuf { self.root.join(table) }

    /// List table names (folders under the root).
    pub fn tables(&self) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        for e in fs::read_dir(&self.root).map_err(|e| e.to_string())?.flatten() {
            if e.path().is_dir() {
                if let Some(n) = e.file_name().to_str() { out.push(n.to_string()); }
            }
        }
        out.sort();
        Ok(out)
    }

    /// All committed segments of a table, sorted by (node, seq) — the
    /// stable default row order across the whole store.
    pub fn segments(&self, table: &str) -> Result<Vec<Segment>, String> {
        let dir = self.table_dir(table);
        let mut segs = Vec::new();
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => return Ok(segs), // absent table = empty table
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some(SEGMENT_EXT) { continue; }
            if let Some(seg) = Segment::parse(&p) { segs.push(seg); }
        }
        segs.sort_by(|a, b| (a.node.as_str(), a.seq).cmp(&(b.node.as_str(), b.seq)));
        Ok(segs)
    }

    /// Reserve the next segment path for an append to `table`: scans THIS
    /// node's existing segments for max seq (crash-safe recovery — the
    /// counter lives in the filenames, nowhere else), returns
    /// (final_path, tmp_path). The caller writes the codec bytes to tmp,
    /// then calls [`commit_segment`].
    pub fn next_segment(&self, table: &str) -> Result<(PathBuf, PathBuf), String> {
        let dir = self.table_dir(table);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let max_seq = self
            .segments(table)?
            .iter()
            .filter(|s| s.node == self.node)
            .map(|s| s.seq)
            .max()
            .unwrap_or(0);
        let rand4 = {
            // Four hex chars from time+pid — a collision guard for two
            // processes on one PC, not cryptography.
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0);
            format!("{:04x}", (t ^ std::process::id()) & 0xFFFF)
        };
        let name = format!("{}-{:06}-{}.{}", self.node, max_seq + 1, rand4, SEGMENT_EXT);
        let fin = dir.join(&name);
        let tmp = dir.join(format!("{}.tmp", name));
        Ok((fin, tmp))
    }
}

/// Commit a fully-written tmp file as an immutable segment (fsync+rename).
pub fn commit_segment(tmp: &Path, fin: &Path) -> Result<(), String> {
    // fsync the tmp so the rename never publishes unflushed data.
    if let Ok(f) = fs::File::open(tmp) { let _ = f.sync_all(); }
    fs::rename(tmp, fin).map_err(|e| format!("db.write: commit failed: {}", e))
}

/// One committed segment file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub path: PathBuf,
    pub node: String,
    pub seq: u64,
}

impl Segment {
    /// Parse `{node}-{seq:06}-{rand4}.parquet`; None for foreign files
    /// (which are thereby ignored by every listing — junk-tolerant).
    pub fn parse(path: &Path) -> Option<Segment> {
        let stem = path.file_stem()?.to_str()?;
        let mut parts = stem.rsplitn(3, '-');
        let _rand = parts.next()?;
        let seq: u64 = parts.next()?.parse().ok()?;
        let node = parts.next()?.to_string();
        if node.is_empty() { return None; }
        Some(Segment { path: path.to_path_buf(), node, seq })
    }

    pub fn file_name(&self) -> String {
        self.path.file_name().unwrap_or_default().to_string_lossy().into_owned()
    }
}

// ─── Schema fingerprint ──────────────────────────────────────────────

/// Loud-error schema check: first write fixes (name, type) per column;
/// every later write must match exactly. The fingerprint is derived from
/// the data.frame at the builtin layer; here it's just ordered pairs.
pub fn check_schema(
    table: &str,
    existing: &[(String, String)],
    incoming: &[(String, String)],
) -> Result<(), String> {
    if existing == incoming { return Ok(()); }
    Err(format!(
        "db.write('{}'): schema mismatch.\n  table has:  {}\n  data has:   {}\n\
         (schema is fixed by the first write; corrections are new rows, \
         new columns need a new table)",
        table, fmt_schema(existing), fmt_schema(incoming)
    ))
}

fn fmt_schema(s: &[(String, String)]) -> String {
    if s.is_empty() { return "<empty>".into(); }
    s.iter().map(|(n, t)| format!("{}:{}", n, t)).collect::<Vec<_>>().join(", ")
}

// ─── Sync (conflict-free file-set union) ─────────────────────────────

/// One direction of a sync: which of `from`'s segment files are missing
/// in `to`, per table. Copying is the caller's job (tmp+rename on the
/// receiving side); this function is the correctness core: the diff of
/// two grow-only sets.
pub fn sync_missing(from: &Store, to: &Store) -> Result<Vec<(String, Vec<Segment>)>, String> {
    let mut plan = Vec::new();
    for table in from.tables()? {
        let have: std::collections::HashSet<String> =
            to.segments(&table)?.iter().map(|s| s.file_name()).collect();
        let missing: Vec<Segment> = from
            .segments(&table)?
            .into_iter()
            .filter(|s| !have.contains(&s.file_name()))
            .collect();
        if !missing.is_empty() { plan.push((table, missing)); }
    }
    Ok(plan)
}

/// Execute a one-way sync (copy missing whole files, tmp+rename).
/// Returns (table, files_copied) rows. Idempotent: a second run copies 0.
pub fn sync_push(from: &Store, to: &Store) -> Result<Vec<(String, usize)>, String> {
    let mut out = Vec::new();
    for (table, missing) in sync_missing(from, to)? {
        let dir = to.root.join(&table);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut n = 0usize;
        for seg in missing {
            let dst = dir.join(seg.file_name());
            let tmp = dir.join(format!("{}.tmp", seg.file_name()));
            fs::copy(&seg.path, &tmp).map_err(|e| format!("db.sync: {}", e))?;
            commit_segment(&tmp, &dst)?;
            n += 1;
        }
        out.push((table, n));
    }
    Ok(out)
}

/// Test/dev helper: write raw bytes as a committed segment (stands in for
/// the parquet codec; the real `db.write` builtin encodes a data.frame).
pub fn write_raw_segment(store: &Store, table: &str, bytes: &[u8]) -> Result<Segment, String> {
    let (fin, tmp) = store.next_segment(table)?;
    let mut f = fs::File::create(&tmp).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    drop(f);
    commit_segment(&tmp, &fin)?;
    Segment::parse(&fin).ok_or_else(|| "internal: unparseable segment name".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store(name: &str, node: &str) -> Store {
        let dir = std::env::temp_dir().join(format!("r2store-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        Store::open(dir, node).unwrap()
    }

    #[test]
    fn seq_recovers_from_filenames_across_reopen() {
        let s = tmp_store("seq", "desk1");
        write_raw_segment(&s, "readings", b"a").unwrap();
        write_raw_segment(&s, "readings", b"b").unwrap();
        // "Crash" and reopen: the counter lives in the filenames.
        let s2 = Store::open(s.root.clone(), "desk1").unwrap();
        let seg = write_raw_segment(&s2, "readings", b"c").unwrap();
        assert_eq!(seg.seq, 3);
        assert_eq!(s2.segments("readings").unwrap().len(), 3);
    }

    #[test]
    fn union_order_is_node_then_seq_and_junk_is_ignored() {
        let s = tmp_store("union", "n01");
        write_raw_segment(&s, "t", b"x").unwrap();
        // A second node's segments land by file copy (as sync would).
        let other = tmp_store("union-other", "n07");
        write_raw_segment(&other, "t", b"y").unwrap();
        sync_push(&other, &s).unwrap();
        // Junk files must be invisible.
        fs::write(s.root.join("t").join("notes.txt"), b"junk").unwrap();
        fs::write(s.root.join("t").join("broken.parquet.tmp"), b"junk").unwrap();
        let segs = s.segments("t").unwrap();
        let order: Vec<(String, u64)> = segs.iter().map(|g| (g.node.clone(), g.seq)).collect();
        assert_eq!(order, vec![("n01".into(), 1), ("n07".into(), 1)]);
    }

    #[test]
    fn crash_leaves_only_invisible_tmp() {
        let s = tmp_store("crash", "n1");
        let (_fin, tmp) = s.next_segment("t").unwrap();
        fs::write(&tmp, b"half-written").unwrap(); // no commit = crash
        assert!(s.segments("t").unwrap().is_empty());
    }

    #[test]
    fn schema_mismatch_is_loud_and_exact_match_passes() {
        let a = vec![("ts".into(), "character".into()), ("v".into(), "numeric".into())];
        assert!(check_schema("t", &a, &a).is_ok());
        let b = vec![("ts".into(), "character".into()), ("v".into(), "integer".into())];
        let e = check_schema("t", &a, &b).unwrap_err();
        assert!(e.contains("schema mismatch") && e.contains("v:integer"));
    }

    #[test]
    fn sync_is_union_and_idempotent() {
        let nodal = tmp_store("sync-nodal", "desk1");
        let host = tmp_store("sync-host", "host");
        write_raw_segment(&nodal, "readings", b"r1").unwrap();
        write_raw_segment(&nodal, "readings", b"r2").unwrap();
        write_raw_segment(&host, "readings", b"h1").unwrap();
        // push: host gains desk1's two segments, keeps its own.
        let n1 = sync_push(&nodal, &host).unwrap();
        assert_eq!(n1, vec![("readings".to_string(), 2)]);
        assert_eq!(host.segments("readings").unwrap().len(), 3);
        // idempotent: second run copies nothing.
        let n2 = sync_push(&nodal, &host).unwrap();
        assert!(n2.is_empty());
        // pull direction completes the union on the nodal side too.
        sync_push(&host, &nodal).unwrap();
        assert_eq!(nodal.segments("readings").unwrap().len(), 3);
    }

    #[test]
    fn node_name_sanitization() {
        assert_eq!(sanitize_node("QC Desk #7"), "qc-desk--7");
        assert_eq!(sanitize_node("!!!"), "node");
    }
}
