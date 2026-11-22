//! Notification center (mailbox + toast).
//!
//! Rebuilt after being wrongly deleted by 002ce467 cloud-removal, keeping only the local paths unrelated to the cloud:
//! - Completion/error notifications of the app's own BYOP agent (Oz)
//! - Status notifications of third-party CLI agents (Claude Code / Codex / DeepSeek, etc.)
//!
//! Module layout:
//! - `item`         data model (`NotificationItem` / `NotificationItems`, etc.)
//! - `item_rendering` single notification UI (shared by mailbox and toast)
//! - `model`        singleton `NotificationsModel` (subscribes to history / cli session models and produces notifications)
//! - `view`         `NotificationMailboxView` (main mailbox panel)
//! - `toast_stack`  `AgentNotificationToastStack` (bottom-right toast)
//! - `telemetry`    notification-center telemetry event (`NotificationsTelemetryEvent`)

pub(crate) mod item;
pub(crate) mod item_rendering;
pub mod model;
pub(crate) mod telemetry;
pub mod toast_stack;
pub mod view;

pub(crate) use item::{
    NotificationCategory, NotificationFilter, NotificationId, NotificationItem, NotificationItems,
    NotificationSourceAgent,
};
pub use toast_stack::AgentNotificationToastStack;
pub use view::{NotificationMailboxView, NotificationMailboxViewEvent};

pub fn init(app: &mut warpui::AppContext) {
    NotificationMailboxView::init(app);
}
