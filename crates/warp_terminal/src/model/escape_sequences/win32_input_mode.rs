use warpui::event::KeyEventDetails;
use warpui::keymap::Keystroke;

use super::{ModeProvider, TermMode, C0};

const LEFT_ALT_PRESSED: u32 = 0x2;
const LEFT_CTRL_PRESSED: u32 = 0x8;
const SHIFT_PRESSED: u32 = 0x10;
const ENHANCED_KEY: u32 = 0x100;

/// Encodes a Ctrl/Alt-modified keystroke as a ConPTY win32-input-mode key event
/// (`CSI Vk ; Sc ; Uc ; Kd ; Cs ; Rc _`) so the console sees the real virtual key instead of
/// bytes that ConPTY has to guess from (e.g. Ctrl+J as Ctrl+Enter).
///
/// Returns None unless the child enabled mode 9001 and the platform supplied a virtual key,
/// in which case the caller falls back to the regular encodings.
/// shortcut: only Ctrl/Alt chords and key-down are sent, since the rest already work through
/// the legacy bytes; upgrade if other keys turn out to be misread.
pub fn maybe_win32_input_escape_sequence(
    keystroke: &Keystroke,
    details: &KeyEventDetails,
    chars: &str,
    mode_provider: &(impl ModeProvider + ?Sized),
) -> Option<Vec<u8>> {
    if !mode_provider.is_term_mode_set(TermMode::WIN32_INPUT) {
        return None;
    }
    let win32 = details.win32?;
    // Windows reports AltGr as Ctrl+Alt; keep the typed character (e.g. the euro sign).
    if !(keystroke.ctrl || keystroke.alt) || keystroke.cmd || (keystroke.ctrl && details.right_alt)
    {
        return None;
    }

    let mut control_key_state = 0;
    if keystroke.ctrl {
        control_key_state |= LEFT_CTRL_PRESSED;
    }
    if keystroke.alt {
        control_key_state |= LEFT_ALT_PRESSED;
    }
    if keystroke.shift {
        control_key_state |= SHIFT_PRESSED;
    }
    // Page up/down, End, Home, arrows, Insert and Delete.
    if matches!(win32.vk, 0x21..=0x28 | 0x2D | 0x2E) {
        control_key_state |= ENHANCED_KEY;
    }
    let unicode_char = chars.encode_utf16().next().unwrap_or(0);

    let sequence = format!(
        "[{};{};{};1;{};1_",
        win32.vk, win32.scan_code, unicode_char, control_key_state
    );
    Some([&[C0::ESC], sequence.as_bytes()].concat())
}
