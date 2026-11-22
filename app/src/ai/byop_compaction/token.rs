//! Token estimation, aligned with opencode `packages/opencode/src/util/token.ts`.
//!
//! ```ts
//! const CHARS_PER_TOKEN = 4
//! export function estimate(input: string) {
//!   return Math.max(0, Math.round((input || "").length / CHARS_PER_TOKEN))
//! }
//! ```
//!
//! Uses `chars().count()` instead of `len()` so multi-byte UTF-8 characters do not skew the estimate wildly.
//! In JS, opencode's `.length` is 1 for BMP characters, which matches chars().count() in most cases;
//! for emoji outside the BMP, JS gives 2 (UTF-16 surrogate pair) while Rust chars().count() gives 1 —
//! this small deviation has no practical effect on head/tail splitting.
use super::consts::CHARS_PER_TOKEN;

/// Equivalent to `Math.round(len / 4)`. An empty string returns 0.
pub fn estimate(input: &str) -> usize {
    let n = input.chars().count();
    // Math.round is "round half to even" before banker's rounding; in JS it behaves as standard rounding.
    // Here (n + 2) / 4 is equivalent to round(n / 4) for positive integers.
    (n + CHARS_PER_TOKEN / 2) / CHARS_PER_TOKEN
}

/// Estimate after JSON serialization, aligned with opencode `compaction.ts:241`:
/// `Token.estimate(JSON.stringify(msgs))`
pub fn estimate_json<T: serde::Serialize>(value: &T) -> usize {
    serde_json::to_string(value)
        .map(|s| estimate(&s))
        .unwrap_or(0)
}
