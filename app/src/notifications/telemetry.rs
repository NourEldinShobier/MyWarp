//! Telemetry events for the in-app notification mailbox / toast stack.
//!
//! This is a minimal trimmed version of `AgentManagementTelemetryEvent`, which was deleted along with cloud removal in 002ce467,
//! keeping only the variants the notification center (`item_rendering.rs`) still uses — the artifact click event +
//! tombstone no longer exists, but the schema is kept for backward compatibility / future rebuild.

use serde::Serialize;

/// Notification artifact type (for telemetry).
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    Plan,
    Branch,
    PullRequest,
}

/// Notification-center related telemetry events.
#[derive(Serialize, Debug)]
pub enum NotificationsTelemetryEvent {
    /// The user clicked an artifact button (plan / branch / PR) in a notification item
    ArtifactClicked { artifact_type: ArtifactType },
}
