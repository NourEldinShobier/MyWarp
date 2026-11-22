//! View model for the "Candidates" section -- flattens the result of `warp_ssh_manager::load_candidates()`
//! (plus the set of already imported aliases and the collapsed state) into a UI-friendly [`CandidateRow`]
//! list.
//!
//! Design notes (see `specs/gh-110-ssh-config-import/{PRODUCT,TECH}.md`):
//!
//! - `rows()` is a **pure function**: it depends only on the view-model's current fields and touches no IO / runtime,
//!   so unit tests can construct a `CandidatesViewModel` directly and assert on the output. This is exactly the point raised in the TDD
//!   discussion -- warpui tests for the PR 2 rendering layer are too costly, so extracting the "which rows to show"
//!   logic into a unit test is enough to cover the key decisions.
//! - `refresh()` synchronously calls `warp_ssh_manager::load_candidates()` (a <10KB file,
//!   see the trade-off in TECH.md §3.1) and stores the result in `state`.
//! - `on_tree_changed()` is called by the panel after it subscribes to `SshTreeChangedNotifier` -- it collects the `host` fields
//!   of all servers in the saved tree into a `HashSet`, which is the basis for the "Added" badge
//!   (PRODUCT.md decision E).
//! - "Already imported" is decided by `host == alias`. The import logic on the panel side sets `server.host`
//!   to the candidate alias (PRODUCT.md decision I), so the comparison semantics here match the import semantics.
//!
//! All fields are `pub(crate)`, visible only to `panel.rs`; `CandidatesViewModel` itself
//! is exposed via `pub` for the re-export in `mod.rs`.

use std::collections::HashSet;

use settings::Setting;
use warp_ssh_manager::{LoadOutcome, LoadResult, SshConfigCandidate, load_candidates};
use warpui::{Entity, ModelContext, SingletonEntity};

use crate::settings::SshSettings;

/// The source + state view of one candidate server line from `~/.ssh/config` in the UI.
pub struct CandidatesViewModel {
    /// Result of the most recent load. `None` means the model was just created and no refresh has been triggered yet.
    state: Option<LoadResult>,
    /// Set of `host` fields of all servers in the saved tree. `rows()` uses it to decide `added`.
    added_aliases: HashSet<String>,
    /// Section collapsed state (PRODUCT.md UX table "Many candidates"). Expanded by default.
    expanded: bool,
}

impl Default for CandidatesViewModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CandidatesViewModel {
    /// All-empty constructor -- used when the model has just been `add_model`ed into the App. `refresh()` must be triggered by the
    /// caller at a suitable time (calling it once right in the panel's `new` is enough).
    pub fn new() -> Self {
        Self {
            state: None,
            added_aliases: HashSet::new(),
            expanded: true,
        }
    }

    /// Test constructor: stuffs the internal state in explicitly, avoiding runtime / IO, to drive
    /// the various branches of `rows()` directly.
    #[cfg(test)]
    pub fn with_state(
        state: Option<LoadResult>,
        added_aliases: HashSet<String>,
        expanded: bool,
    ) -> Self {
        Self {
            state,
            added_aliases,
            expanded,
        }
    }

    /// Synchronously re-reads `~/.ssh/config` and stores the result in `state`.
    ///
    /// By design it returns no error -- `LoadOutcome::Error` already carries the error message string back,
    /// and the UI shows it as a red error row (see PRODUCT.md UX table "Parse / IO error").
    ///
    /// When the "auto-discover SSH hosts" setting is off, skips reading and clears the state.
    pub fn refresh(&mut self, ctx: &mut ModelContext<Self>) {
        let auto_discover = *SshSettings::as_ref(ctx).enable_ssh_auto_discovery.value();
        if !auto_discover {
            self.state = None;
            ctx.notify();
            return;
        }
        self.state = Some(load_candidates());
        ctx.notify();
    }

    /// Tree-change callback -- rebuilds `added_aliases` from the passed-in server hosts.
    ///
    /// Takes `impl IntoIterator<Item = String>` rather than `&SshRepository` so tests
    /// don't need to supply a real SQLite connection; the caller (panel) is responsible for collecting the host fields of `list_nodes` +
    /// `get_server` into an iterator before passing it in.
    pub fn on_tree_changed<I>(&mut self, hosts: I, ctx: &mut ModelContext<Self>)
    where
        I: IntoIterator<Item = String>,
    {
        self.added_aliases = hosts.into_iter().collect();
        ctx.notify();
    }

    /// Toggles the "section collapsed" state.
    pub fn toggle_expanded(&mut self, ctx: &mut ModelContext<Self>) {
        self.expanded = !self.expanded;
        ctx.notify();
    }

    /// Whether it is expanded (the panel decides at render time whether to show the body rows).
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Looks up a candidate by alias -- used when handling the `ImportCandidate { alias }` action,
    /// to call `SshRepository::create_server` after getting the full fields.
    pub fn find_candidate(&self, alias: &str) -> Option<&SshConfigCandidate> {
        let state = self.state.as_ref()?;
        match &state.outcome {
            LoadOutcome::Loaded(v) => v.iter().find(|c| c.alias == alias),
            LoadOutcome::NotFound | LoadOutcome::Error(_) => None,
        }
    }

    /// A readable string of the current `~/.ssh/config` path (used for `notes = "Imported from {}"`).
    /// `None` means it has not been loaded yet, or even home could not be determined.
    pub fn path_display(&self) -> Option<String> {
        self.state
            .as_ref()
            .and_then(|s| s.path.as_ref())
            .map(|p| p.display().to_string())
    }

    /// Flattens the current state into a row list -- see the "pure function" convention in the module docs.
    ///
    /// Output semantics (matching the PRODUCT.md §5 UX table):
    /// - Not yet refreshed: returns an empty Vec (the panel does not render the section when it sees `state == None`).
    /// - `NotFound`: Header + one `NotFound` row.
    /// - `Error`: Header + one `Error` row (can_refresh=true lets the user retry after fixing the config).
    /// - `Loaded(empty)`: Header + one `Empty` row.
    /// - `Loaded(non-empty)`: Header (count = N) + N `Candidate` rows, where each row's
    ///   `added` is decided by `added_aliases.contains(alias)`.
    pub fn rows(&self) -> Vec<CandidateRow> {
        let Some(state) = self.state.as_ref() else {
            return Vec::new();
        };

        let path_display = state
            .path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let mut out = Vec::new();
        let count = match &state.outcome {
            LoadOutcome::Loaded(v) => v.len(),
            LoadOutcome::NotFound | LoadOutcome::Error(_) => 0,
        };
        // The Header is always the first row -- even when the section is collapsed, the panel still draws the header (that is
        // the toggle entry). `can_refresh = true` always holds: in any state the user may click
        // Refresh to re-read.
        out.push(CandidateRow::Header {
            path_display: path_display.clone(),
            count,
            can_refresh: true,
        });

        // When the section is collapsed only the header is kept; the body is not rendered.
        if !self.expanded {
            return out;
        }

        match &state.outcome {
            LoadOutcome::NotFound => {
                out.push(CandidateRow::NotFound { path_display });
            }
            LoadOutcome::Error(msg) => {
                out.push(CandidateRow::Error {
                    path_display,
                    message: msg.clone(),
                });
            }
            LoadOutcome::Loaded(v) if v.is_empty() => {
                out.push(CandidateRow::Empty { path_display });
            }
            LoadOutcome::Loaded(v) => {
                for c in v {
                    out.push(CandidateRow::Candidate {
                        alias: c.alias.clone(),
                        hostname: c.hostname.clone(),
                        user: c.user.clone(),
                        port: c.port,
                        identity_file: c.identity_file.as_ref().map(|p| p.display().to_string()),
                        added: self.added_aliases.contains(&c.alias),
                    });
                }
            }
        }

        out
    }
}

/// One UI-friendly row. The Header is always first, followed by either a single status row (NotFound /
/// Empty / Error) or a series of Candidate rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateRow {
    Header {
        path_display: String,
        count: usize,
        can_refresh: bool,
    },
    NotFound {
        path_display: String,
    },
    Empty {
        path_display: String,
    },
    Error {
        path_display: String,
        message: String,
    },
    Candidate {
        alias: String,
        hostname: Option<String>,
        user: Option<String>,
        port: Option<u16>,
        identity_file: Option<String>,
        added: bool,
    },
}

impl Entity for CandidatesViewModel {
    type Event = ();
}

#[cfg(test)]
#[path = "candidates_tests.rs"]
mod tests;

// Lets test code ignore the concrete on-disk PathBuf -- the helper uses `LoadResult` to assemble a
// fixed display string. It is also used in the test module, so it lives in the outer scope to make #[cfg(test)] reuse easy.
#[cfg(test)]
pub(crate) fn fake_load_result_loaded(path: &str, cands: Vec<SshConfigCandidate>) -> LoadResult {
    LoadResult {
        path: Some(std::path::PathBuf::from(path)),
        outcome: LoadOutcome::Loaded(cands),
    }
}

#[cfg(test)]
pub(crate) fn fake_load_result_not_found(path: &str) -> LoadResult {
    LoadResult {
        path: Some(std::path::PathBuf::from(path)),
        outcome: LoadOutcome::NotFound,
    }
}

#[cfg(test)]
pub(crate) fn fake_load_result_error(path: &str, msg: &str) -> LoadResult {
    LoadResult {
        path: Some(std::path::PathBuf::from(path)),
        outcome: LoadOutcome::Error(msg.to_string()),
    }
}
