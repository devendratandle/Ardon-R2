//! Capability policy — the security boundary for server-hosted sessions.
//!
//! Agent/LLM-driven sessions execute UNTRUSTED generated code, so the
//! server creates their engines with a default-deny policy and the
//! operator grants capabilities per session. Enforcement happens at ONE
//! choke point — `call_fn`'s builtin dispatch — driven by category name
//! lists, so every present and future named builtin in a category is
//! covered without per-builtin edits. Interactive CLI/GUI engines use
//! `Policy::allow_all()` (unchanged behavior).

/// What a session is allowed to touch. Fields are grant flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// Reading files/dirs: read.csv/read.table/readLines/load/source/
    /// list.files/file.exists/mmap/parquet readers…
    pub fs_read: bool,
    /// Writing files: write.csv/write.table/writeLines/save/plot devices
    /// that write to disk…
    pub fs_write: bool,
    /// Process environment + host introspection: Sys.getenv/Sys.setenv.
    pub env_access: bool,
    /// Installing/loading addon packages from disk (code loading).
    pub install: bool,
}

impl Default for Policy {
    fn default() -> Self { Policy::allow_all() }
}

impl Policy {
    /// Interactive sessions: everything allowed (the user owns the machine).
    pub fn allow_all() -> Self {
        Policy { fs_read: true, fs_write: true, env_access: true, install: true }
    }
    /// Server sessions start here: pure compute only. The operator grants
    /// capabilities explicitly at session creation.
    pub fn deny_all() -> Self {
        Policy { fs_read: false, fs_write: false, env_access: false, install: false }
    }

    const FS_READ: &'static [&'static str] = &[
        "read.csv", "read.table", "readLines", "load", "source", "scan",
        "file.exists", "list.files", "readRDS", "read.parquet", "mmap",
        "getwd",
    ];
    const FS_WRITE: &'static [&'static str] = &[
        "write.csv", "write.table", "writeLines", "save", "saveRDS",
        "write.parquet", "save_plot", "save.plot", "pdf", "svg", "png",
        "dev.off", "setwd", "file.remove", "unlink",
    ];
    const ENV_ACCESS: &'static [&'static str] = &[
        "Sys.getenv", "Sys.setenv",
    ];
    const INSTALL: &'static [&'static str] = &[
        "install.packages", "library", "require", "uninstall", "detach",
    ];

    /// The single dispatch-time check: `Some(reason)` blocks the call.
    pub fn deny_reason(&self, builtin: &str) -> Option<String> {
        let cat = if !self.fs_read && Self::FS_READ.contains(&builtin) {
            "fs_read"
        } else if !self.fs_write && Self::FS_WRITE.contains(&builtin) {
            "fs_write"
        } else if !self.env_access && Self::ENV_ACCESS.contains(&builtin) {
            "env_access"
        } else if !self.install && Self::INSTALL.contains(&builtin) {
            "install"
        } else {
            return None;
        };
        Some(format!(
            "'{builtin}' denied by session policy (capability '{cat}' not granted)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deny_all_blocks_every_category_allow_all_blocks_nothing() {
        let locked = Policy::deny_all();
        for f in ["read.csv", "write.csv", "Sys.getenv", "install.packages", "source"] {
            assert!(locked.deny_reason(f).is_some(), "{f} should be denied");
        }
        assert!(locked.deny_reason("sum").is_none(), "pure compute stays allowed");
        assert!(locked.deny_reason("lm").is_none());
        let open = Policy::allow_all();
        for f in ["read.csv", "write.csv", "Sys.getenv", "install.packages", "sum"] {
            assert!(open.deny_reason(f).is_none());
        }
    }
    #[test]
    fn partial_grant() {
        let p = Policy { fs_read: true, ..Policy::deny_all() };
        assert!(p.deny_reason("read.csv").is_none());
        assert!(p.deny_reason("write.csv").is_some());
    }
}
