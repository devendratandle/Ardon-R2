//! Layered function registry — how user-defined functions, addon
//! packages, base packages, and core primitives resolve in priority
//! order at lookup time.
//!
//! Resolution order (top wins):
//!   1. User-defined functions in the global environment
//!   2. Last-loaded addon package
//!   3. ... earlier addon packages ...
//!   4. Base libraries (stats, graphics, utils, base)
//!   5. CORE primitives (IMMUTABLE — addons cannot mask these)
//!
//! `pkg::func()` bypasses resolution — direct namespace access.
//! `detach(pkg)` removes a layer; everything below is naturally restored.

use std::collections::HashMap;

use crate::BuiltinFn;

#[derive(Clone)]
pub struct PackageLayer {
    pub name: String,
    pub tier: PackageTier,
    pub functions: HashMap<String, BuiltinFn>,
    pub exports: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PackageTier {
    /// CANNOT be masked or detached.
    Core,
    /// CAN be masked by addon, CAN be detached.
    Base,
    /// CAN be masked by later addon, CAN be detached.
    Addon,
}

pub struct FunctionRegistry {
    /// pub(crate) so the Engine eval loop in `lib.rs` can iterate
    /// layers directly. External crates use the methods.
    pub(crate) layers: Vec<PackageLayer>,
}

impl Default for FunctionRegistry {
    fn default() -> Self { Self::new() }
}

impl FunctionRegistry {
    pub fn new() -> Self { FunctionRegistry { layers: Vec::new() } }

    pub fn add_layer(&mut self, layer: PackageLayer) { self.layers.push(layer); }

    pub fn remove_layer(&mut self, name: &str) -> Result<Vec<String>, String> {
        let pos = self.layers.iter().position(|l| l.name == name);
        match pos {
            Some(i) => {
                if self.layers[i].tier == PackageTier::Core {
                    return Err(format!("cannot detach core package '{}'", name));
                }
                let removed = self.layers.remove(i);
                let restored: Vec<String> = removed.exports.iter()
                    .filter(|f| self.resolve(f).is_some())
                    .cloned().collect();
                Ok(restored)
            }
            None => Err(format!("package '{}' not loaded", name)),
        }
    }

    /// Resolve a function name. Core always wins for core names; for
    /// everything else, top-of-stack down.
    pub fn resolve(&self, name: &str) -> Option<(BuiltinFn, &str)> {
        // Core is immutable — check first.
        for layer in &self.layers {
            if layer.tier == PackageTier::Core {
                if let Some(f) = layer.functions.get(name) {
                    return Some((*f, &layer.name));
                }
            }
        }
        // Then search last-loaded first (addons mask base).
        for layer in self.layers.iter().rev() {
            if layer.tier == PackageTier::Core { continue; }
            if let Some(f) = layer.functions.get(name) {
                return Some((*f, &layer.name));
            }
        }
        None
    }

    /// Direct namespace: `pkg::func()` bypasses the search order.
    pub fn resolve_in_package(&self, pkg: &str, name: &str) -> Option<BuiltinFn> {
        self.layers.iter().find(|l| l.name == pkg)
            .and_then(|l| l.functions.get(name).copied())
    }

    pub fn is_core(&self, name: &str) -> bool {
        self.layers.iter().any(|l| l.tier == PackageTier::Core && l.functions.contains_key(name))
    }

    /// For an incoming addon, return `(masked_name, masking_pkg)` for
    /// each name that would shadow an already-loaded function.
    pub fn check_masks(&self, new_exports: &[String]) -> Vec<(String, String)> {
        let mut masks = Vec::new();
        for name in new_exports {
            if let Some((_, from)) = self.resolve(name) {
                if !self.is_core(name) { masks.push((name.clone(), from.to_string())); }
            }
        }
        masks
    }

    /// R-style search-path output: `.GlobalEnv` first, then addons
    /// (most-recently-loaded first), then base layers, then core.
    pub fn search_path(&self) -> Vec<String> {
        let mut path = vec![".GlobalEnv".to_string()];
        for layer in self.layers.iter().rev() {
            if layer.tier != PackageTier::Core {
                path.push(format!("package:{}", layer.name));
            }
        }
        path.push("package:core".to_string());
        path
    }
}
