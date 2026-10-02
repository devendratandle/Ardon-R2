// build.rs — generate r2.ico from assets/logo.png and embed it as the
// Windows exe icon. Runs once per build; cargo re-runs whenever the
// source PNG changes (rerun-if-changed line below).
//
// What this does:
//   1. Open ../../assets/logo.png  (the full brand logo, 1344×784).
//   2. Crop the left portion containing just the "R2" mark.
//   3. Trim transparent / white borders to a tight bounding box.
//   4. Pad to a square with transparent background.
//   5. Resample to seven standard icon sizes
//        16, 24, 32, 48, 64, 128, 256
//   6. Write the multi-resolution .ico to OUT_DIR/r2.ico.
//   7. On Windows, embed it as the exe icon via winresource.
//
// On non-Windows platforms the .ico file is still produced (useful for
// future Linux/.deb desktop entries) but no embedding step runs.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Project root is two levels up from crates/r2-repl/.
    let project_root = manifest_dir.parent().unwrap().parent().unwrap();
    let logo_path = project_root.join("assets").join("logo.png");

    println!("cargo:rerun-if-changed={}", logo_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    if !logo_path.exists() {
        println!("cargo:warning=assets/logo.png not found at {} — building without a custom icon", logo_path.display());
        return;
    }

    let img = match image::open(&logo_path) {
        Ok(im) => im.into_rgba8(),
        Err(e) => {
            println!("cargo:warning=could not decode {}: {} — skipping icon", logo_path.display(), e);
            return;
        }
    };

    // ── Crop to the R2 mark ─────────────────────────────────────────
    // The source logo has the R2 mark + "Ardon" wordmark beneath, and
    // sometimes a small monochrome R2 badge in the lower-right corner.
    // We want JUST the big R2 mark.
    //
    // Strategy: keep the FULL width (so we never clip the "2" on the
    // right edge — its exact horizontal extent varies between source
    // PNGs) and crop only vertically to drop the "Ardon" text and the
    // bottom-right badge. trim_to_content() then finds the tight
    // bounding box of the R2 mark's actual pixels.
    let (w, h) = (img.width(), img.height());
    let crop_w = w;                            // full width — let trim find the right edge
    let crop_h = (h as f32 * 0.65) as u32;     // drop bottom ~35% (Ardon wordmark + small badge)
    let cropped = image::imageops::crop_imm(&img, 0, 0, crop_w, crop_h).to_image();

    // ── Auto-trim non-transparent / non-near-white border ───────────
    let trimmed = trim_to_content(&cropped);

    // ── Pad to a square (transparent background) ────────────────────
    let square = pad_to_square(&trimmed);

    // ── Debug PNG (lets us eyeball the cropped result without
    //    needing to extract from .ico, which Windows tools mangle).
    let debug_png = project_root.join("installer").join("icon-256-preview.png");
    let preview_256 = image::imageops::resize(&square, 256, 256, image::imageops::FilterType::Lanczos3);
    let _ = preview_256.save(&debug_png);

    // ── Generate multi-resolution ICO ────────────────────────────────
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let ico_path = out_dir.join("r2.ico");
    write_multires_ico(&square, &ico_path);

    // ── Also copy to a stable path for the installer (installer/R2.iss
    // references this).
    let installer_ico = project_root.join("installer").join("r2.ico");
    let _ = std::fs::create_dir_all(installer_ico.parent().unwrap());
    let _ = std::fs::copy(&ico_path, &installer_ico);

    // ── Embed in the Windows exe ─────────────────────────────────────
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon(ico_path.to_str().unwrap());
        res.set("ProductName",  "Ardon-R2");
        res.set("FileDescription", "Ardon-R2 — statistics & AI training, reimagined");
        res.set("CompanyName",  "Devendra Tandale");
        res.set("LegalCopyright", "AGPL-3.0");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed icon resource: {}", e);
        }
    }
    #[cfg(not(windows))]
    {
        let _ = ico_path; // silence unused warning
    }
}

// Find the tight bounding box of pixels that are neither fully
// transparent nor near-white, then return that subimage.
fn trim_to_content(img: &image::RgbaImage) -> image::RgbaImage {
    let (w, h) = (img.width(), img.height());
    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0u32;
    let mut y1 = 0u32;
    for (x, y, p) in img.enumerate_pixels() {
        let [r, g, b, a] = p.0;
        let near_white = r > 240 && g > 240 && b > 240;
        if a > 8 && !near_white {
            if x < x0 { x0 = x; }
            if y < y0 { y0 = y; }
            if x > x1 { x1 = x; }
            if y > y1 { y1 = y; }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        // Nothing to trim — return as-is.
        return img.clone();
    }
    // Add a small padding so glyphs aren't flush with the edges.
    let pad = 8;
    let nx0 = x0.saturating_sub(pad);
    let ny0 = y0.saturating_sub(pad);
    let nx1 = (x1 + pad).min(w - 1);
    let ny1 = (y1 + pad).min(h - 1);
    image::imageops::crop_imm(img, nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1).to_image()
}

// Center the image on a square canvas with transparent background.
fn pad_to_square(img: &image::RgbaImage) -> image::RgbaImage {
    let (w, h) = (img.width(), img.height());
    let side = w.max(h);
    let mut out = image::ImageBuffer::from_fn(side, side, |_, _| image::Rgba([0, 0, 0, 0]));
    let ox = (side - w) / 2;
    let oy = (side - h) / 2;
    image::imageops::overlay(&mut out, img, ox as i64, oy as i64);
    out
}

fn write_multires_ico(square: &image::RgbaImage, out_path: &std::path::Path) {
    let sizes = [16u32, 24, 32, 48, 64, 128, 256];
    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);
    for size in sizes {
        let resized = image::imageops::resize(
            square, size, size, image::imageops::FilterType::Lanczos3);
        let rgba: Vec<u8> = resized.into_raw();
        let frame = ico::IconImage::from_rgba_data(size, size, rgba);
        icon_dir.add_entry(ico::IconDirEntry::encode(&frame).expect("encode ico frame"));
    }
    let file = std::fs::File::create(out_path).expect("create r2.ico");
    icon_dir.write(file).expect("write r2.ico");
}
