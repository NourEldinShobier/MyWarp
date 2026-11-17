//! Windows shell launcher: `warp.exe [--tab] [path]` opens Warp via its URL scheme.
//! Compiled by bundle.ps1 and installed as `bin\warp.exe` and `bin\warp.exe`.
#![windows_subsystem = "windows"]

use std::path::PathBuf;
use std::process::Command;

const APP_EXE: &str = "warp.exe";
const URL_SCHEME: &str = "warp";

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
    let mut new_tab = false;
    let mut path = None;
    for arg in std::env::args().skip(1) {
        if arg == "--tab" {
            new_tab = true;
        } else {
            // A quoted `"C:\"` reaches us as `C:"` because `\"` escapes the quote, so strip it.
            let arg = arg.trim_end_matches('"').to_string();
            if !arg.is_empty() {
                path = Some(arg);
            }
        }
    }

    // The launcher lives in `<app>\bin`; the app executable is one level up.
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent()?.parent().map(|d| d.join(APP_EXE)))
        .unwrap_or_else(|| PathBuf::from(APP_EXE));

    let mut cmd = Command::new(exe);
    if let Some(path) = path {
        let mut p = PathBuf::from(path);
        if p.is_relative() {
            if let Ok(cwd) = std::env::current_dir() {
                p = cwd.join(p);
            }
        }
        let action = if new_tab { "new_tab" } else { "new_window" };
        cmd.arg(format!(
            "{URL_SCHEME}://action/{action}?path={}",
            percent_encode(&p.to_string_lossy())
        ));
    }
    if cmd.spawn().is_err() {
        std::process::exit(1);
    }
}
