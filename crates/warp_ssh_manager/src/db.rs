//! One process-wide `Mutex<SqliteConnection>` shared by the SSH manager.
//!
//! Current state: openWarp's main write connection lives in a dedicated write thread (see `app/src/persistence/sqlite.rs`)
//! and is handled asynchronously through the `ModelEvent` channel. Hooking the SSH manager into that event bus would need 6+ enum
//! variants plus cross-crate type exposure, which costs too much.
//!
//! Alternative: **SQLite WAL mode natively supports multiple write connections** (writes are mutually exclusive, with busy_timeout retries),
//! so this opens a separate write connection whose behavior stays entirely local to this crate. SSH manager writes are
//! user-driven (creating/deleting nodes) and very infrequent, so conflicts with the main write thread are negligible.
//!
//! The path is passed in by the caller at initialization (`set_database_path`), so this crate does not depend directly on the app
//! layer's `database_file_path()`. If no path was passed, `with_conn` returns `Err(NotInitialized)`.

use anyhow::{Result, anyhow};
use diesel::connection::SimpleConnection;
use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static DB_PATH: OnceLock<PathBuf> = OnceLock::new();
static CONN: OnceLock<Mutex<SqliteConnection>> = OnceLock::new();

/// Called once by the app at startup with the sqlite db file path. Repeated calls are ignored
/// (OnceLock semantics).
pub fn set_database_path(path: PathBuf) {
    let _ = DB_PATH.set(path);
}

fn open() -> Result<SqliteConnection> {
    let path = DB_PATH
        .get()
        .ok_or_else(|| anyhow!("warp_ssh_manager::db: database path not initialized"))?;
    let url = path.to_string_lossy();
    let mut conn = SqliteConnection::establish(&url)?;
    conn.batch_execute(
        "PRAGMA foreign_keys = ON; \
         PRAGMA busy_timeout = 2000; \
         PRAGMA journal_mode = WAL;",
    )?;
    Ok(conn)
}

/// Runs the closure under the lock. The connection is opened lazily on first call and reused afterwards.
pub fn with_conn<R>(f: impl FnOnce(&mut SqliteConnection) -> Result<R>) -> Result<R> {
    let mtx = CONN.get_or_init(|| Mutex::new(open().expect("warp_ssh_manager db open")));
    let mut guard = mtx
        .lock()
        .map_err(|_| anyhow!("warp_ssh_manager db mutex poisoned"))?;
    f(&mut guard)
}
