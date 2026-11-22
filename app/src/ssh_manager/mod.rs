//! SSH manager UI (left Tool Panel). Currently a skeleton; content is pending Commit 2b:
//! a tree folder/server list + a detail form on the right.
//!
//! The data layer is in the standalone crate `warp_ssh_manager` (`crates/warp_ssh_manager/`).

pub mod candidates;
pub mod notifier;
pub mod onekey;
pub mod panel;
pub mod password_prompt;
pub mod secret_injector;
pub mod server_view;
pub mod shell_prompt;
pub mod startup_command_injector;
pub mod su_password_injector;

// `CandidatesViewModel` is only referenced by `panel.rs` for now; `CandidateRow` is just an intermediate
// representation for the panel's internal layout and need not be exported. Add a re-export when external consumers need it.
#[allow(unused_imports)]
pub use candidates::CandidatesViewModel;
pub use notifier::{SshTreeChangedEvent, SshTreeChangedNotifier};
pub use panel::SshManagerPanel;
// Re-exports for downstream UI consumers (Commit 2b).
#[allow(unused_imports)]
pub use panel::{SshManagerPanelAction, SshManagerPanelEvent};
