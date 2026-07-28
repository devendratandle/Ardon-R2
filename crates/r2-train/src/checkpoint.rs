//! Checkpointing — survive a crash without losing the run.
//!
//! Long training is interrupted by things a safe language cannot prevent:
//! power loss, OOM-kill, a full disk, ECC errors, a failed node. In a
//! cluster that risk *grows with scale* — 100 nodes at one-month
//! individual MTBF gives a cluster MTBF of hours. Checkpointing is
//! insurance against hardware, not against our bugs.
//!
//! **Optimizer state is part of the checkpoint, not an extra.** Adam
//! carries momentum, variance and a step counter; a "weights-only" resume
//! silently starts a differently-conditioned optimization that merely
//! shares the current weights. The test suite proves both halves of that:
//! a full resume is *identical* to an uninterrupted run, and a
//! weights-only resume *diverges*.
//!
//! **Crash-safety is structural.** Writes go to a temporary file, are
//! flushed, and then renamed — a rename is atomic, so a crash mid-write
//! leaves the previous checkpoint untouched rather than a half-written
//! file that loads as garbage. Rotation keeps N recent checkpoints, so
//! even a corrupted newest one is recoverable.
//!
//! **Cost.** Weights are the small part: Adam adds two f32s per
//! parameter, so a resumable checkpoint is ~3× the weight size. Written
//! on a background thread it need not stall compute at all — there is no
//! GC and no stop-the-world pause to schedule around.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::optim::Adam;

const MAGIC: &[u8; 8] = b"R2CKPT\0\x01";
/// Bumped when the layout changes; an old file is then refused by name
/// rather than misread.
const VERSION: u32 = 1;

/// A training snapshot: everything needed to continue exactly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    /// Global step, so a resumed run reports honest progress.
    pub step: u64,
    /// Named f32 tensors: parameters AND optimizer moments.
    pub tensors: BTreeMap<String, Vec<f32>>,
    /// Named scalars: learning rate, loss, RNG seed, β's — anything the
    /// loop needs to reproduce its next step.
    pub scalars: BTreeMap<String, f64>,
}

impl Checkpoint {
    pub fn new(step: u64) -> Self { Checkpoint { step, ..Default::default() } }

    pub fn put_tensor(&mut self, name: &str, data: &[f32]) {
        self.tensors.insert(name.to_string(), data.to_vec());
    }
    pub fn put_scalar(&mut self, name: &str, v: f64) {
        self.scalars.insert(name.to_string(), v);
    }
    pub fn tensor(&self, name: &str) -> Result<&[f32], String> {
        self.tensors.get(name).map(|v| v.as_slice())
            .ok_or_else(|| format!("checkpoint: missing tensor '{}'", name))
    }
    pub fn scalar(&self, name: &str) -> Result<f64, String> {
        self.scalars.get(name).copied()
            .ok_or_else(|| format!("checkpoint: missing scalar '{}'", name))
    }

    /// Bytes this checkpoint will occupy — so a training loop can report
    /// its cost instead of surprising the operator.
    pub fn size_bytes(&self) -> usize {
        self.tensors.values().map(|v| v.len() * 4).sum::<usize>()
    }

    // ── Optimizer round-trip ──

    /// Store an optimizer's full state under `prefix`.
    pub fn put_adam(&mut self, prefix: &str, opt: &Adam) {
        self.put_tensor(&format!("{prefix}.m"), &opt.m);
        self.put_tensor(&format!("{prefix}.v"), &opt.v);
        self.put_scalar(&format!("{prefix}.t"), opt.t as f64);
        self.put_scalar(&format!("{prefix}.lr"), opt.lr as f64);
        self.put_scalar(&format!("{prefix}.beta1"), opt.beta1 as f64);
        self.put_scalar(&format!("{prefix}.beta2"), opt.beta2 as f64);
        self.put_scalar(&format!("{prefix}.eps"), opt.eps as f64);
    }

    /// Rebuild an optimizer exactly — including `t`, without which bias
    /// correction restarts and the next step is the wrong size.
    pub fn get_adam(&self, prefix: &str) -> Result<Adam, String> {
        let m = self.tensor(&format!("{prefix}.m"))?.to_vec();
        let v = self.tensor(&format!("{prefix}.v"))?.to_vec();
        if m.len() != v.len() {
            return Err(format!("checkpoint: '{prefix}' m/v length mismatch"));
        }
        Ok(Adam {
            m, v,
            t: self.scalar(&format!("{prefix}.t"))? as u64,
            lr: self.scalar(&format!("{prefix}.lr"))? as f32,
            beta1: self.scalar(&format!("{prefix}.beta1"))? as f32,
            beta2: self.scalar(&format!("{prefix}.beta2"))? as f32,
            eps: self.scalar(&format!("{prefix}.eps"))? as f32,
        })
    }

    // ── Persistence ──

    /// Serialize to bytes. Little-endian throughout so a checkpoint moves
    /// between machines unchanged.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.size_bytes() + 256);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.step.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        for (name, data) in &self.tensors {
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(data.len() as u64).to_le_bytes());
            for x in data { b.extend_from_slice(&x.to_le_bytes()); }
        }
        b.extend_from_slice(&(self.scalars.len() as u32).to_le_bytes());
        for (name, v) in &self.scalars {
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// Parse bytes. Every length is checked against what remains, so a
    /// truncated file is an error naming the problem — never a panic and
    /// never a silently short tensor.
    pub fn from_bytes(b: &[u8]) -> Result<Checkpoint, String> {
        let mut r = Reader { b, i: 0 };
        if r.take(8)? != MAGIC {
            return Err("checkpoint: bad magic (not an R2 checkpoint)".into());
        }
        let ver = r.u32()?;
        if ver != VERSION {
            return Err(format!("checkpoint: version {} not supported (expected {})",
                               ver, VERSION));
        }
        let mut c = Checkpoint::new(r.u64()?);
        let n_t = r.u32()? as usize;
        for _ in 0..n_t {
            let name = r.string()?;
            let len = r.u64()? as usize;
            let mut data = Vec::with_capacity(len.min(1 << 20));
            for _ in 0..len { data.push(r.f32()?); }
            c.tensors.insert(name, data);
        }
        let n_s = r.u32()? as usize;
        for _ in 0..n_s {
            let name = r.string()?;
            c.scalars.insert(name, r.f64()?);
        }
        if r.i != b.len() {
            return Err(format!("checkpoint: {} trailing bytes", b.len() - r.i));
        }
        Ok(c)
    }

    /// Write atomically: temp file → flush → rename. A crash during the
    /// write cannot damage the previous checkpoint, because the rename is
    /// the only moment the new one becomes visible.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir).map_err(|e| format!("checkpoint: {}", e))?;
            }
        }
        let tmp = path.with_extension("tmp");
        {
            let mut f = fs::File::create(&tmp)
                .map_err(|e| format!("checkpoint: create {}: {}", tmp.display(), e))?;
            f.write_all(&self.to_bytes()).map_err(|e| format!("checkpoint: write: {}", e))?;
            f.sync_all().map_err(|e| format!("checkpoint: sync: {}", e))?;
        }
        fs::rename(&tmp, path).map_err(|e| format!("checkpoint: commit: {}", e))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Checkpoint, String> {
        let mut f = fs::File::open(path.as_ref())
            .map_err(|e| format!("checkpoint: open {}: {}", path.as_ref().display(), e))?;
        let mut b = Vec::new();
        f.read_to_end(&mut b).map_err(|e| format!("checkpoint: read: {}", e))?;
        Checkpoint::from_bytes(&b)
    }

    /// Save as `dir/step-<n>.ckpt`, keeping only the newest `keep`.
    /// Rotation matters: if only one checkpoint existed and it were
    /// corrupted by a bad disk, the run would be unrecoverable.
    pub fn save_rotating(&self, dir: impl AsRef<Path>, keep: usize) -> Result<PathBuf, String> {
        let dir = dir.as_ref();
        let path = dir.join(format!("step-{:09}.ckpt", self.step));
        self.save(&path)?;
        if keep > 0 {
            let mut found = list_checkpoints(dir)?;
            while found.len() > keep {
                let (_, oldest) = found.remove(0);
                let _ = fs::remove_file(oldest); // best effort; never fatal
            }
        }
        Ok(path)
    }
}

/// Checkpoints in `dir` as (step, path), oldest first.
pub fn list_checkpoints(dir: impl AsRef<Path>) -> Result<Vec<(u64, PathBuf)>, String> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(dir.as_ref()) {
        Ok(r) => r,
        Err(_) => return Ok(out), // no directory yet = no checkpoints
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("ckpt") { continue; }
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if let Some(n) = stem.strip_prefix("step-").and_then(|s| s.parse::<u64>().ok()) {
            out.push((n, p));
        }
    }
    out.sort_by_key(|(n, _)| *n);
    Ok(out)
}

/// Load the newest checkpoint in `dir`, or `None` if there is none —
/// the "resume if possible, else start fresh" call a training loop makes
/// at startup.
pub fn load_latest(dir: impl AsRef<Path>) -> Result<Option<Checkpoint>, String> {
    match list_checkpoints(dir)?.pop() {
        Some((_, p)) => Ok(Some(Checkpoint::load(p)?)),
        None => Ok(None),
    }
}

struct Reader<'a> { b: &'a [u8], i: usize }

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.i + n > self.b.len() {
            return Err(format!("checkpoint: truncated (wanted {} bytes at {}, have {})",
                               n, self.i, self.b.len()));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        let s = self.take(n)?;
        String::from_utf8(s.to_vec()).map_err(|_| "checkpoint: name is not UTF-8".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("r2ckpt-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Deterministic gradient of a convex bowl, so any trajectory
    /// difference comes from the optimizer, not from data noise.
    fn grad(p: &[f32], target: &[f32]) -> Vec<f32> {
        p.iter().zip(target).map(|(a, t)| 2.0 * (a - t)).collect()
    }

    #[test]
    fn resume_is_identical_to_an_uninterrupted_run() {
        // THE INVARIANT: training 10 steps == training 5, saving,
        // reloading, then training 5 more. This is what makes "train in
        // parts and join them" exact rather than approximate.
        let target = vec![1.0f32, -2.0, 0.5, 3.0];
        let dir = tmpdir("resume");

        // Uninterrupted reference.
        let mut p_ref = vec![0.0f32; 4];
        let mut opt_ref = Adam::new(4, 0.05);
        for _ in 0..10 {
            let g = grad(&p_ref, &target);
            opt_ref.step(&mut p_ref, &g).unwrap();
        }

        // Interrupted: 5 steps, checkpoint, reload into FRESH objects, 5 more.
        let mut p = vec![0.0f32; 4];
        let mut opt = Adam::new(4, 0.05);
        for _ in 0..5 {
            let g = grad(&p, &target);
            opt.step(&mut p, &g).unwrap();
        }
        let mut ck = Checkpoint::new(5);
        ck.put_tensor("params", &p);
        ck.put_adam("opt", &opt);
        let path = ck.save_rotating(&dir, 3).unwrap();

        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.step, 5);
        let mut p2 = loaded.tensor("params").unwrap().to_vec();
        let mut opt2 = loaded.get_adam("opt").unwrap();
        assert_eq!(opt2.t, 5, "step counter must survive (bias correction depends on it)");
        for _ in 0..5 {
            let g = grad(&p2, &target);
            opt2.step(&mut p2, &g).unwrap();
        }

        assert_eq!(p2, p_ref, "resumed params must match the uninterrupted run exactly");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn weights_only_resume_diverges() {
        // The counter-example that justifies storing optimizer state: keep
        // the weights, drop momentum/variance, and the trajectory changes.
        let target = vec![1.0f32, -2.0, 0.5, 3.0];
        let mut p_ref = vec![0.0f32; 4];
        let mut opt_ref = Adam::new(4, 0.05);
        for _ in 0..10 {
            let g = grad(&p_ref, &target);
            opt_ref.step(&mut p_ref, &g).unwrap();
        }

        let mut p = vec![0.0f32; 4];
        let mut opt = Adam::new(4, 0.05);
        for _ in 0..5 {
            let g = grad(&p, &target);
            opt.step(&mut p, &g).unwrap();
        }
        opt.reset_state(); // weights-only "resume"
        for _ in 0..5 {
            let g = grad(&p, &target);
            opt.step(&mut p, &g).unwrap();
        }
        assert_ne!(p, p_ref, "dropping optimizer state must change the trajectory");
    }

    #[test]
    fn round_trips_tensors_and_scalars_exactly() {
        let mut c = Checkpoint::new(42);
        c.put_tensor("a", &[1.0, -2.5, f32::MIN_POSITIVE]);
        c.put_tensor("b", &[]);
        c.put_scalar("loss", 0.1234567890123);
        let back = Checkpoint::from_bytes(&c.to_bytes()).unwrap();
        assert_eq!(back, c, "serialization must be lossless");
        assert_eq!(back.size_bytes(), 12);
    }

    #[test]
    fn corrupt_and_truncated_files_are_rejected() {
        let mut c = Checkpoint::new(1);
        c.put_tensor("a", &[1.0, 2.0, 3.0]);
        let good = c.to_bytes();

        assert!(Checkpoint::from_bytes(&good[..good.len() - 3])
            .unwrap_err().contains("truncated"));
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(Checkpoint::from_bytes(&bad_magic).unwrap_err().contains("bad magic"));
        let mut bad_ver = good.clone();
        bad_ver[8] = 99;
        assert!(Checkpoint::from_bytes(&bad_ver).unwrap_err().contains("not supported"));
        let mut extra = good.clone();
        extra.push(0);
        assert!(Checkpoint::from_bytes(&extra).unwrap_err().contains("trailing"));
        assert!(Checkpoint::from_bytes(&[]).is_err());
    }

    #[test]
    fn a_failed_write_cannot_destroy_the_previous_checkpoint() {
        // The atomic-rename property, observed: after a good save, a stray
        // .tmp file (what a crash mid-write leaves) does not affect what
        // loads.
        let dir = tmpdir("atomic");
        let path = dir.join("step-000000001.ckpt");
        let mut c1 = Checkpoint::new(1);
        c1.put_tensor("w", &[1.0, 2.0]);
        c1.save(&path).unwrap();

        fs::write(path.with_extension("tmp"), b"garbage half-write").unwrap();
        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.tensor("w").unwrap(), &[1.0, 2.0]);
        assert!(list_checkpoints(&dir).unwrap().len() == 1, ".tmp must not count as a checkpoint");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rotation_keeps_the_newest_and_load_latest_finds_it() {
        let dir = tmpdir("rotate");
        for step in 1..=5u64 {
            let mut c = Checkpoint::new(step);
            c.put_tensor("w", &[step as f32]);
            c.save_rotating(&dir, 2).unwrap();
        }
        let found = list_checkpoints(&dir).unwrap();
        assert_eq!(found.len(), 2, "rotation must keep exactly `keep` files");
        assert_eq!(found.iter().map(|(s, _)| *s).collect::<Vec<_>>(), vec![4, 5]);

        let latest = load_latest(&dir).unwrap().unwrap();
        assert_eq!(latest.step, 5);
        assert_eq!(latest.tensor("w").unwrap(), &[5.0]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_latest_on_an_empty_or_missing_dir_is_none_not_an_error() {
        // A training loop calls this at startup; "no checkpoint yet" is
        // the normal first-run case, not a failure.
        assert!(load_latest(std::env::temp_dir().join("r2ckpt-does-not-exist")).unwrap().is_none());
        let dir = tmpdir("empty");
        assert!(load_latest(&dir).unwrap().is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_fields_report_what_is_missing() {
        let c = Checkpoint::new(0);
        assert!(c.tensor("nope").unwrap_err().contains("missing tensor 'nope'"));
        assert!(c.scalar("nope").unwrap_err().contains("missing scalar 'nope'"));
        assert!(c.get_adam("opt").is_err());
    }
}
