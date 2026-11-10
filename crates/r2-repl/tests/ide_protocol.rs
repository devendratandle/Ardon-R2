//! `r2 --ide`, driven the way an IDE drives it: the real binary, JSON
//! lines in and out (docs/IDE_PROTOCOL.md).

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Run requests (each `None` is a pause) and return every line R2 wrote.
fn session(requests: &[Option<&str>]) -> Vec<String> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_r2")).arg("--ide")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
        .spawn().expect("start r2 --ide");
    let mut stdin = child.stdin.take().unwrap();
    for r in requests {
        match r {
            Some(line) => { writeln!(stdin, "{line}").unwrap(); stdin.flush().unwrap(); }
            None => std::thread::sleep(Duration::from_millis(400)),
        }
    }
    drop(stdin);
    let out = BufReader::new(child.stdout.take().unwrap()).lines().map(|l| l.unwrap()).collect();
    child.wait().unwrap();
    out
}

fn reply(lines: &[String], id: u32) -> &str {
    let tag = format!("{{\"id\":{id},\"ok\"");
    lines.iter().find(|l| l.starts_with(&tag)).unwrap_or_else(|| panic!("no reply {id} in {lines:#?}"))
}

#[test]
fn console_values_environment_and_viewer() {
    let out = session(&[
        Some(r#"{"id":1,"op":"hello"}"#),
        Some(r#"{"id":2,"op":"eval","code":"d <- data.frame(x = c(81.5, 77, 92.25, NA), g = c(\"a\", \"b\", \"a\", \"b\"))"}"#),
        Some(r#"{"id":3,"op":"eval","code":"mean(d$x, na.rm = TRUE); invisible(5)"}"#),
        Some(r#"{"id":4,"op":"eval","code":"f <- function(v) {"}"#),
        Some(r#"{"id":5,"op":"value","code":"summary(d$x)"}"#),
        Some(r#"{"id":6,"op":"env"}"#),
        Some(r#"{"id":7,"op":"view","code":"d","offset":1,"limit":2}"#),
        Some(r#"{"id":8,"op":"eval","code":"stop(\"boom\")"}"#),
        Some(r#"{"id":9,"op":"frobnicate"}"#),
        Some(r#"{"id":10,"op":"quit"}"#),
    ]);
    assert!(reply(&out, 1).contains(&format!("\"r2\":\"{}\",\"protocol\":1", env!("CARGO_PKG_VERSION"))));
    // visible values print on the out stream; invisible() does not
    let streams: Vec<&String> = out.iter().filter(|l| l.starts_with("{\"id\":3,\"stream\"")).collect();
    assert_eq!(streams.len(), 1, "{streams:?}");
    assert!(streams[0].contains("\"stream\":\"out\"") && streams[0].contains("83.58333"));
    assert!(reply(&out, 4).contains("\"incomplete\":true"));
    // summary(): R's values at full precision, NA count included
    assert!(reply(&out, 5).contains("\"values\":[77.0,79.25,81.5,83.58333333333333,86.875,92.25,1.0]"));
    assert!(reply(&out, 6).contains("{\"name\":\"d\",\"group\":\"Data\",\"summary\":\"4 obs. of 2 variables\",\"viewable\":true}"));
    let page = reply(&out, 7);
    assert!(page.contains("\"nrow\":4") && page.contains("\"headers\":[\"x\",\"g\"]"));
    assert!(page.contains("{\"label\":\"2\",\"cells\":[\"77.00\",\"b\"]}") && page.contains("{\"label\":\"3\",\"cells\":[\"92.25\",\"a\"]}"));
    assert!(!page.contains("\"label\":\"1\""), "the page starts at offset 1");
    assert!(out.iter().any(|l| l.starts_with("{\"id\":8,\"stream\":\"err\"") && l.contains("boom")));
    assert!(reply(&out, 8).contains("\"error\":true"));
    assert!(reply(&out, 9).contains("unknown op"));
    assert_eq!(reply(&out, 10), "{\"id\":10,\"ok\":true}");
}

#[test]
fn interrupt_stops_a_running_loop_and_the_session_goes_on() {
    let out = session(&[
        Some(r#"{"id":1,"op":"eval","code":"i <- 0; repeat { i <- i + 1 }"}"#),
        None,
        Some(r#"{"op":"interrupt"}"#),
        Some(r#"{"id":2,"op":"value","code":"i > 0"}"#),
        Some(r#"{"id":3,"op":"quit"}"#),
    ]);
    assert!(reply(&out, 1).contains("\"interrupted\":true"));
    assert!(reply(&out, 2).contains("\"values\":[true]"));
}

#[test]
fn a_plot_comes_back_as_svg_and_only_when_it_changes() {
    let out = session(&[
        Some(r#"{"id":1,"op":"eval","code":"hist(c(1, 2, 2, 3))"}"#),
        Some(r#"{"id":2,"op":"eval","code":"x <- 1"}"#),
        Some(r#"{"id":3,"op":"value","code":"hist(c(1, 2, 2, 3), plot = FALSE)$counts"}"#),
        Some(r#"{"id":4,"op":"quit"}"#),
    ]);
    assert!(reply(&out, 1).contains("\"plot\":\"<svg"));
    assert!(!reply(&out, 2).contains("\"plot\""));
    assert!(reply(&out, 3).contains("\"values\":[1.0,2.0,0.0,1.0]"), "GNU R: hist(c(1,2,2,3))$counts is 1 2 0 1");
    assert!(!out.iter().any(|l| l.contains("browser")), "no browser viewer under an IDE");
}

/// Ardon Studio's Gview reads the reply's plot: it must be the whole picture,
/// overlays included, with the title, subtitle, axis names and plotting region
/// marked so they can be picked and edited.
#[test]
fn the_reply_plot_is_the_whole_picture_with_its_parts_marked() {
    let out = session(&[
        Some(r#"{"id":1,"op":"eval","code":"plot(1:10, main = \"Main title\", xlab = \"The x\", ylab = \"The y\", xlim = c(0, 20))"}"#),
        Some(r#"{"id":2,"op":"eval","code":"lines(1:10, 10:1)"}"#),
        Some(r#"{"id":3,"op":"eval","code":"abline(h = 5); points(3, 3); text(5, 8, \"LABEL\"); title(sub = \"Sub title\")"}"#),
        Some(r#"{"id":4,"op":"eval","code":"hist(c(1, 2, 2, 3, 9))"}"#),
        Some(r#"{"id":5,"op":"eval","code":"lines(density(c(1, 2, 2, 3, 9)))"}"#),
        Some(r#"{"id":6,"op":"quit"}"#),
    ]);
    let first = reply(&out, 1);
    assert!(first.contains(r#"class=\"main\""#) && first.contains("Main title"));
    assert!(first.contains(r#"class=\"xlab\""#) && first.contains(r#"class=\"ylab\""#));
    // xlim = c(0, 20), padded 4 % on each side as R's par("usr") is
    assert!(first.contains(r#"class=\"plot-region\" data-usr=\"-0.8 20.8"#), "{first}");
    let after_lines = reply(&out, 2);
    assert!(after_lines.contains("\"plot\":") && after_lines.len() > first.len(), "an overlay sends the whole picture again");
    let after_text = reply(&out, 3);
    assert!(after_text.contains("LABEL") && after_text.contains("Sub title"), "text() and title(sub =) are in the reply");
    assert!(after_text.contains("Main title"), "the plot under the overlays is still there");
    let hist = reply(&out, 5);
    assert!(hist.contains(r#"class=\"plot-region\""#) && hist.contains("<path"),
        "lines(density()) draws over the histogram");
}
