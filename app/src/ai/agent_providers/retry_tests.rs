use super::*;

#[test]
fn transient_statuses() {
    assert!(is_transient_status(429, "slow down"));
    assert!(is_transient_status(529, "overloaded"));
    assert!(is_transient_status(503, ""));
    assert!(!is_transient_status(429, "{\"error\":\"insufficient_quota\"}"));
    assert!(!is_transient_status(400, ""));
    assert!(!is_transient_status(401, ""));
    assert!(!is_transient_status(501, ""));
}

#[test]
fn backoff_is_2_4_8_then_gives_up() {
    assert_eq!(next_delay(0, None), Some(Duration::from_secs(2)));
    assert_eq!(next_delay(1, Some((429, ""))), Some(Duration::from_secs(4)));
    assert_eq!(next_delay(2, Some((500, ""))), Some(Duration::from_secs(8)));
    assert_eq!(next_delay(3, None), None);
    assert_eq!(next_delay(0, Some((400, ""))), None);
}
