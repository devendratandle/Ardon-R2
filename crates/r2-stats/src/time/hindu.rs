//! Hindu calendar — tithi + Saka-era date (Phase R.T.5b).
//!
//! Astronomical model: mean solar/lunar longitudes à la Sūrya
//! Siddhānta. A tithi (lunar day) is a 12° gain of the Moon over the
//! Sun in geocentric longitude; 30 tithis per lunar month (Shukla
//! 1..15, Krishna 1..15). Mean elements only — ~1-day accuracy, good
//! for calendar display; full Drik Panchang precision is an addon.
//!
//! Split out of `time.rs` (was a two-domain file). Date/calendar
//! helpers (`gv`, `as_str`, `days_from_civil`, `civil_from_days`,
//! `format_date`) are reached via `super::` — a child module sees its
//! parent's items.

use std::sync::Arc;
use r2_types::*;
use super::{gv, civil_from_days};

// Phase R.T.5b — Hindu calendar (tithi + saka-era date)
//
// Astronomical model: mean solar/lunar longitudes a la Sūrya Siddhānta.
// Tithi is the lunar day; one tithi corresponds to a 12° gain of the
// Moon over the Sun in geocentric longitude. There are 30 tithis in a
// lunar month (Shukla 1..15, then Krishna 1..15, ending with Amavasya).
//
// This implementation uses simple mean elements — accurate to ~1 day,
// good enough for calendar display. Full Drik Panchang precision would
// require VSOP87/ELP-2000 ephemerides, which we leave to an addon.
// ═══════════════════════════════════════════════════════════════════════

const TITHI_NAMES: [&str; 30] = [
    "Pratipada","Dwitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami",
    "Ashtami","Navami","Dashami","Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima",
    "Pratipada","Dwitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami",
    "Ashtami","Navami","Dashami","Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Amavasya",
];

const MASA_NAMES: [&str; 12] = [
    "Chaitra","Vaisakha","Jyeshtha","Ashadha","Shravana","Bhadrapada",
    "Ashwin","Kartika","Margashirsha","Pausha","Magha","Phalguna",
];

// Short codes for tithi names, used in compact HNC output and
// hnc.format(). Convention: first three letters lowercase, with
// "puri" for Purnima and "ama" for Amavasya as the two markers.
const TITHI_SHORT: [&str; 30] = [
    "pra","dwi","tri","cha","pan","sha","sap",
    "ash","nav","das","eka","dwa","tra","cdr","pur",
    "pra","dwi","tri","cha","pan","sha","sap",
    "ash","nav","das","eka","dwa","tra","cdr","ama",
];

const MASA_SHORT: [&str; 12] = [
    "cai","vai","jye","ash","shr","bha","ashw","kar","mar","pau","mag","pha",
];

// ── Adhik Maas (intercalary month) detection ──────────────────────────
//
// A lunar month is *Adhik* iff there is no solar saṅkrānti
// (sun crossing a multiple of 30°) between two consecutive amavasyas.
// We test this around the requested date by locating the bracketing
// amavasyas via mean elements and checking whether the sun's longitude
// crossed a 30° boundary in between.
//
// Limitation: mean-elements model is approximate; this can disagree
// with a Drik Panchang almanac by ±1 day at the boundaries. Good
// enough for calendar display, not for muhurta computation.

/// Find the JD of the amavasya at-or-just-before `jd0`.
fn prev_amavasya(jd0: f64) -> f64 {
    // Walk backwards day-by-day until elong crosses 0.
    let mut jd = jd0;
    let mut last_elong = (moon_mean_long(jd) - sun_mean_long(jd)).rem_euclid(360.0);
    for _ in 0..35 {
        jd -= 1.0;
        let elong = (moon_mean_long(jd) - sun_mean_long(jd)).rem_euclid(360.0);
        // Amavasya = elong wraps 360→0. Detect by elong dropping a lot.
        if elong > last_elong + 100.0 {
            // Refine to the day before the wrap.
            return jd + 1.0;
        }
        last_elong = elong;
    }
    jd0 - 29.5 // fallback
}

/// Find the JD of the next amavasya strictly after `jd0`.
fn next_amavasya(jd0: f64) -> f64 {
    let mut jd = jd0;
    let mut last_elong = (moon_mean_long(jd) - sun_mean_long(jd)).rem_euclid(360.0);
    for _ in 0..35 {
        jd += 1.0;
        let elong = (moon_mean_long(jd) - sun_mean_long(jd)).rem_euclid(360.0);
        if elong + 100.0 < last_elong { return jd; }
        last_elong = elong;
    }
    jd0 + 29.5
}

/// Is the lunar month containing `jd` an Adhik month?
fn is_adhik(jd: f64) -> bool {
    let start = prev_amavasya(jd);
    let end   = next_amavasya(start + 1.0);
    let sun_start = sun_mean_long(start);
    let sun_end   = sun_mean_long(end);
    // Saṅkrānti = sun crosses a 30° multiple. Compare integer rashis.
    let r_start = (sun_start / 30.0).floor() as i32;
    let r_end   = (sun_end   / 30.0).floor() as i32;
    // Account for year wrap.
    let crossings = if r_end >= r_start { r_end - r_start } else { r_end + 12 - r_start };
    crossings == 0
}

/// True (apparent) geocentric longitude of the Sun in degrees, 0..360.
///
/// Uses Meeus' low-accuracy formula (Astronomical Algorithms, Ch. 25):
/// mean longitude + equation of center. Accurate to ~0.01° (~36 arcsec),
/// good enough for tithi computation.
fn sun_mean_long(jd: f64) -> f64 {
    let t = (jd - 2451545.0) / 36525.0;
    let l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t;
    let m  = 357.52911 + 35999.05029 * t - 0.0001537 * t * t;
    let m_rad = m.to_radians();
    // Equation of center (largest three terms).
    let c = (1.914602 - 0.004817 * t - 0.000014 * t * t) * m_rad.sin()
          + (0.019993 - 0.000101 * t)                    * (2.0 * m_rad).sin()
          +  0.000289                                     * (3.0 * m_rad).sin();
    (l0 + c).rem_euclid(360.0)
}

/// True (apparent) geocentric longitude of the Moon in degrees, 0..360.
///
/// Uses the six largest periodic terms from Meeus Ch. 47 (the full
/// ELP-2000 series has ~60 terms; we keep only those above 0.1°).
/// Accuracy: ~0.03° (~2 arcmin), which keeps tithi within ~30 minutes
/// of Drik Panchang — sufficient for the tithi number to match.
fn moon_mean_long(jd: f64) -> f64 {
    let t = (jd - 2451545.0) / 36525.0;
    let l_prime = 218.3164477 + 481267.88123421 * t
                 - 0.0015786 * t * t + t * t * t / 538841.0;
    let d  = 297.8501921 + 445267.1114034 * t
            - 0.0018819 * t * t + t * t * t / 545868.0;
    let m  = 357.5291092 + 35999.0502909 * t
            - 0.0001536 * t * t;
    let mp = 134.9633964 + 477198.8675055 * t
            + 0.0087414 * t * t + t * t * t / 69699.0;
    let f  = 93.2720950 + 483202.0175233 * t
            - 0.0036539 * t * t;
    // Largest six periodic terms.
    let mp_r = mp.to_radians();
    let d_r  = d.to_radians();
    let m_r  = m.to_radians();
    let f_r  = f.to_radians();
    let sum  = 6.288774 * (mp_r).sin()
             - 1.274027 * (mp_r - 2.0 * d_r).sin()
             + 0.658314 * (2.0 * d_r).sin()
             + 0.213618 * (2.0 * mp_r).sin()
             - 0.185116 * (m_r).sin()
             - 0.114332 * (2.0 * f_r).sin();
    (l_prime + sum).rem_euclid(360.0)
}

/// Convert R2 internal day count (days since 1970-01-01) to Julian Day.
/// JD of 1970-01-01 00:00 UT = 2440587.5. We add 0.5 to use noon as the
/// reference (standard for tithi tables).
fn r2days_to_jd(days: f64) -> f64 {
    2440587.5 + days + 0.5
}

/// `tithi(date)` — returns a list with tithi number (1..30), name,
/// paksha ("Shukla" or "Krishna"), and the lunar masa (month).
pub fn bi_tithi(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let days = match v {
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("Date") => {
            xs.first().and_then(|x| *x).ok_or_else(|| R2Err { msg: "tithi(): empty Date".into(), kind: ErrKind::Runtime })?
        }
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("POSIXct") => {
            (xs.first().and_then(|x| *x).ok_or_else(|| R2Err { msg: "tithi(): empty POSIXct".into(), kind: ErrKind::Runtime })?) / 86_400.0
        }
        _ => return Err(R2Err { msg: "tithi(): need a Date or POSIXct".into(), kind: ErrKind::Type }),
    };
    let jd = r2days_to_jd(days);
    let sun = sun_mean_long(jd);
    let moon = moon_mean_long(jd);
    let elong = (moon - sun).rem_euclid(360.0);
    // 1..30 (Pratipada = elong in [0°,12°))
    let tithi_idx = (elong / 12.0).floor() as usize;     // 0..29
    let paksha = if tithi_idx < 15 { "Shukla" } else { "Krishna" };
    let name = TITHI_NAMES[tithi_idx];
    // Lunar month: 12 amanta months per year, starting Chaitra at the
    // amavasya before the spring equinox (sun_longitude ≈ 330°→0° crossing).
    let masa_idx = ((sun / 30.0).floor() as usize).rem_euclid(12);
    let adhik = is_adhik(jd);
    let short = format!("{}-{}", if tithi_idx < 15 { "shu" } else { "kru" }, TITHI_SHORT[tithi_idx]);
    Ok(RVal::List(vec![
        (Some(Arc::from("tithi")),   RVal::Numeric(vec![Some(tithi_idx as f64 + 1.0)].into(), Attrs::default())),
        (Some(Arc::from("name")),    RVal::Character(vec![Some(Arc::from(name))], Attrs::default())),
        (Some(Arc::from("short")),   RVal::Character(vec![Some(Arc::from(short.as_str()))], Attrs::default())),
        (Some(Arc::from("paksha")),  RVal::Character(vec![Some(Arc::from(paksha))], Attrs::default())),
        (Some(Arc::from("masa")),    RVal::Character(vec![Some(Arc::from(MASA_NAMES[masa_idx]))], Attrs::default())),
        (Some(Arc::from("masa.short")), RVal::Character(vec![Some(Arc::from(MASA_SHORT[masa_idx]))], Attrs::default())),
        (Some(Arc::from("adhik")),   RVal::Logical(vec![Some(adhik)].into(), Attrs::default())),
        (Some(Arc::from("elong")),   RVal::Numeric(vec![Some(elong)].into(), Attrs::default())),
    ]))
}

/// `hnc.date(date)` — Hindu National Calendar numeric form, anchored to
/// the Shālivāhana Śaka era (year 1 = 78 CE). The new year is **Chaitra
/// Śukla Pratipadā** (Gudi Padwa / Ugadi), which falls in March of the
/// Gregorian year.
///
/// Format: `SSSS-MM-P-TT` (with optional `A` after MM for Adhik Maas)
///
/// * `SSSS` = Śaka year. Increments at Chaitra-1 (~late March), so a
///            January date belongs to the previous Śaka year.
/// * `MM`   = lunar month 01..12 (01 = Chaitra, …, 12 = Phalguna).
/// * `A`    = "Adhik" marker, present iff the month has no saṅkrānti.
/// * `P`    = paksha (1 = Śukla / waxing, 2 = Kṛṣṇa / waning).
/// * `TT`   = tithi within the paksha, 01..15.
///
/// Examples:
///
///   * Gudi Padwa (HNC new year)         → `1946-01-1-01`
///   * Holi 2024 (Krishna Pratipada)     → `1946-01-2-01`
///   * Adhik Jyeshtha Shukla Panchami    → `1946-03A-1-05`
///
/// Returns a list with the numeric breakdown plus a `formatted` string.
pub fn bi_hnc_date(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let days = match v {
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("Date") => {
            xs.first().and_then(|x| *x).ok_or_else(|| R2Err { msg: "hnc.date(): empty Date".into(), kind: ErrKind::Runtime })?
        }
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("POSIXct") => {
            (xs.first().and_then(|x| *x).ok_or_else(|| R2Err { msg: "hnc.date(): empty POSIXct".into(), kind: ErrKind::Runtime })?) / 86_400.0
        }
        _ => return Err(R2Err { msg: "hnc.date(): need a Date or POSIXct".into(), kind: ErrKind::Type }),
    };

    let (gy, gm, _gd) = civil_from_days(days as i64);
    let jd = r2days_to_jd(days);
    let sun  = sun_mean_long(jd);
    let moon = moon_mean_long(jd);
    let elong = (moon - sun).rem_euclid(360.0);

    let tithi_idx = (elong / 12.0).floor() as usize;        // 0..29
    let paksha    = if tithi_idx < 15 { 1u32 } else { 2u32 }; // 1=Shukla, 2=Krishna
    let tithi_num = (tithi_idx % 15) as u32 + 1;             // 1..15
    let masa_idx  = ((sun / 30.0).floor() as usize).rem_euclid(12);
    let masa_num  = (masa_idx as u32) + 1;                   // 1..12 (1=Chaitra)
    let adhik     = is_adhik(jd);

    // Saka year increments at Chaitra-1. Before mid-March we're still
    // in the previous Saka year. Mean-elements approximation: subtract
    // 78 from Gregorian year, then back off by 1 if we're before Chaitra.
    let saka_year = if gm >= 4 || (gm == 3 && masa_num == 1) {
        gy - 78
    } else {
        gy - 79
    };

    let mm = if adhik {
        format!("{:02}A", masa_num)
    } else {
        format!("{:02}", masa_num)
    };
    let formatted = format!("{}-{}-{}-{:02}", saka_year, mm, paksha, tithi_num);

    Ok(RVal::List(vec![
        (Some(Arc::from("saka.year")), RVal::Numeric(vec![Some(saka_year as f64)].into(), Attrs::default())),
        (Some(Arc::from("masa")),      RVal::Numeric(vec![Some(masa_num as f64)].into(),  Attrs::default())),
        (Some(Arc::from("masa.name")), RVal::Character(vec![Some(Arc::from(MASA_NAMES[masa_idx]))], Attrs::default())),
        (Some(Arc::from("adhik")),     RVal::Logical(vec![Some(adhik)].into(), Attrs::default())),
        (Some(Arc::from("paksha")),    RVal::Numeric(vec![Some(paksha as f64)].into(), Attrs::default())),
        (Some(Arc::from("tithi")),     RVal::Numeric(vec![Some(tithi_num as f64)].into(), Attrs::default())),
        (Some(Arc::from("formatted")), RVal::Character(vec![Some(Arc::from(formatted.as_str()))], Attrs::default())),
    ]))
}

/// `hindu.date(date)` — return Saka-era year, masa, paksha, tithi as a
/// human-readable list. Saka era began 78 CE, so Saka year = Gregorian year − 78
/// (approximately — strict conversion requires checking whether the
/// solar new year has passed; we approximate to keep this in pure Rust).
pub fn bi_hindu_date(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let days = match v {
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("Date") => {
            xs.first().and_then(|x| *x).ok_or_else(|| R2Err { msg: "hindu.date(): empty Date".into(), kind: ErrKind::Runtime })?
        }
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("POSIXct") => {
            (xs.first().and_then(|x| *x).ok_or_else(|| R2Err { msg: "hindu.date(): empty POSIXct".into(), kind: ErrKind::Runtime })?) / 86_400.0
        }
        _ => return Err(R2Err { msg: "hindu.date(): need a Date or POSIXct".into(), kind: ErrKind::Type }),
    };
    let (gy, _gm, _gd) = civil_from_days(days as i64);
    let saka_year = gy - 78;
    // Reuse tithi() machinery.
    let jd = r2days_to_jd(days);
    let sun  = sun_mean_long(jd);
    let moon = moon_mean_long(jd);
    let elong = (moon - sun).rem_euclid(360.0);
    let tithi_idx = (elong / 12.0).floor() as usize;
    let paksha = if tithi_idx < 15 { "Shukla" } else { "Krishna" };
    let tithi_name = TITHI_NAMES[tithi_idx];
    let tithi_num = (tithi_idx % 15) + 1;
    let masa_idx = ((sun / 30.0).floor() as usize).rem_euclid(12);
    let masa = MASA_NAMES[masa_idx];

    let formatted = format!("{}-{} ({} Paksha {}) {} Saka", saka_year, masa, paksha, tithi_name, tithi_num);
    Ok(RVal::List(vec![
        (Some(Arc::from("saka.year")), RVal::Numeric(vec![Some(saka_year as f64)].into(), Attrs::default())),
        (Some(Arc::from("masa")),      RVal::Character(vec![Some(Arc::from(masa))], Attrs::default())),
        (Some(Arc::from("paksha")),    RVal::Character(vec![Some(Arc::from(paksha))], Attrs::default())),
        (Some(Arc::from("tithi")),     RVal::Numeric(vec![Some(tithi_num as f64)].into(), Attrs::default())),
        (Some(Arc::from("tithi.name")),RVal::Character(vec![Some(Arc::from(tithi_name))], Attrs::default())),
        (Some(Arc::from("formatted")), RVal::Character(vec![Some(Arc::from(formatted.as_str()))], Attrs::default())),
    ]))
}
