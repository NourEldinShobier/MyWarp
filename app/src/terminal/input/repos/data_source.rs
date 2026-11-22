//! Async data source for the inline repos menu.
//!
//! Historically this pulled the list of "previously opened git repos" from `PersistedWorkspace`.
//! After LSP + workspace history were taken offline, that candidate source no longer exists, so this data source
//! only keeps the trait and view wiring and always returns empty results -- that is, the menu can still be
//! brought up but never has candidates. This avoids large changes to the upper-level view / suggestions mode
//! wiring; if the "current pane group live cwd" is hooked up in the future, the data source can be restored.

use warpui::{AppContext, Entity};

use crate::search::data_source::{Query, QueryResult};
use crate::search::mixer::{AsyncDataSource, BoxFuture, DataSourceRunErrorWrapper};
use crate::terminal::input::repos::AcceptRepo;

pub struct RepoMenuDataSource;

impl RepoMenuDataSource {
    pub fn new() -> Self {
        Self
    }
}

impl AsyncDataSource for RepoMenuDataSource {
    type Action = AcceptRepo;

    fn run_query(
        &self,
        _query: &Query,
        _app: &AppContext,
    ) -> BoxFuture<'static, Result<Vec<QueryResult<Self::Action>>, DataSourceRunErrorWrapper>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
}

impl Entity for RepoMenuDataSource {
    type Event = ();
}
