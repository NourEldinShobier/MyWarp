//! Command palette data source: SSH servers (unique to openWarp).
//!
//! In Ctrl+Shift+P the user fuzzy-matches by server name / host; selecting one emits
//! `WorkspaceAction::OpenSshTerminal` to open a new tab and connect (via SecretInjector, automatically
//! injecting the password, fully equivalent to "Connect" in the SSH manager right-click menu).

pub mod data_source;
pub mod search_item;

pub use data_source::SshServersDataSource;
pub use search_item::SshServerSearchItem;
