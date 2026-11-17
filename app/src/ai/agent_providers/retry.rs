//! Retry policy for BYOP stream-open failures. Only used before any output was emitted,
//! so a retry never duplicates part of a reply.

use std::time::Duration;

/// Retries after the first attempt.
pub const MAX_RETRIES: u32 = 3;
const BASE_DELAY: Duration = Duration::from_secs(2);

const EXHAUSTED_QUOTA_MARKERS: [&str; 4] = [
    "insufficient_quota",
    "usage_limit_reached",
    "subscription_sharing_usage_limit_exceeded",
    "billing",
];

/// Whether an HTTP status (with its body) is worth sending again: 408, 429 (unless the
/// body says the quota is exhausted), and 5xx except 501/505. Overloaded (529) is a 5xx.
pub fn is_transient_status(status: u16, body: &str) -> bool {
    match status {
        408 => true,
        429 => {
            let body = body.to_ascii_lowercase();
            !EXHAUSTED_QUOTA_MARKERS.iter().any(|m| body.contains(m))
        }
        501 | 505 => false,
        500..=599 => true,
        _ => false,
    }
}

/// Wait before retry number `retry` (from 1): 2s, 4s, 8s.
pub fn delay(retry: u32) -> Duration {
    BASE_DELAY * 2u32.saturating_pow(retry.saturating_sub(1))
}

/// `Some(wait)` when the failure should be retried after `retries_done` earlier retries.
/// `status` is `None` for a dropped/failed connection.
pub fn next_delay(retries_done: u32, status: Option<(u16, &str)>) -> Option<Duration> {
    let transient = status.map_or(true, |(s, b)| is_transient_status(s, b));
    (transient && retries_done < MAX_RETRIES).then(|| delay(retries_done + 1))
}

#[cfg(test)]
#[path = "retry_tests.rs"]
mod tests;
