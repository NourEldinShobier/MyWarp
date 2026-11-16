// Tiny `warp.exe` that opens a Warp OSS window (or tab with `--tab`) in a directory.
// Usable from the Explorer address bar, Win+R, any shell, and the Explorer context menu.
// Lives in `<app>\bin` and starts `<app>\warp-oss.exe` with a `warposs://action/...` link.
#![windows_subsystem = "windows"]

use std::env;
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

const DETACHED_PROCESS: u32 = 0x0000_0008;

fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn main() {
    let mut tab = false;
    let mut target: Option<String> = None;
    for arg in env::args().skip(1) {
        if arg == "--tab" {
            tab = true;
        } else if target.is_none() {
            target = Some(arg);
        }
    }

    let cwd = env::current_dir().ok();
    let path: Option<PathBuf> = match target {
        Some(raw) => {
            // `"%V"` with a drive root arrives as `D:"` because `\"` escapes the quote.
            let mut raw = raw.trim_end_matches('"').to_owned();
            if raw.len() == 2 && raw.ends_with(':') {
                raw.push('\\');
            }
            let p = PathBuf::from(raw);
            if p.is_absolute() {
                Some(p)
            } else {
                cwd.map(|c| c.join(&p)).or(Some(p))
            }
        }
        None => cwd,
    };

    let exe = env::current_exe()
        .ok()
        .and_then(|e| e.parent().and_then(|bin| bin.parent()).map(|app| app.join("warp-oss.exe")));
    let Some(exe) = exe else { return };

    let mut url = format!("warposs://action/new_{}", if tab { "tab" } else { "window" });
    if let Some(p) = path {
        url.push_str("?path=");
        url.push_str(&percent_encode(&p.to_string_lossy()));
    }
    let _ = Command::new(exe).arg(url).creation_flags(DETACHED_PROCESS).spawn();
}
