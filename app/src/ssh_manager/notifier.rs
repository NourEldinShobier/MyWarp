//! Global SSH tree change broadcast -- any view that changes the tree structure (add/delete/rename/change a server field)
//! calls `notify` afterwards, and subscribers such as SshManagerPanel refresh accordingly.
//!
//! Same pattern as `KeybindingChangedNotifier` (`app/src/settings_view/keybindings.rs:72`):
//! an empty struct + SingletonEntity + a single Event variant.

use warpui::{Entity, SingletonEntity};

#[derive(Default)]
pub struct SshTreeChangedNotifier {}

impl SshTreeChangedNotifier {
    pub fn new() -> Self {
        Default::default()
    }
}

#[derive(Clone, Debug)]
pub enum SshTreeChangedEvent {
    /// The node list / server details have changed; list_nodes needs to be called again.
    TreeChanged,
}

impl Entity for SshTreeChangedNotifier {
    type Event = SshTreeChangedEvent;
}

impl SingletonEntity for SshTreeChangedNotifier {}
