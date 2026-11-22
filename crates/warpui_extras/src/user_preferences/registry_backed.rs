use std::io;
use std::sync::Mutex;

/// Store user preferences in the Windows Registry.
/// Modeled after https://github.com/neovide/neovide/blob/main/src/windows_utils.rs .
use super::UserPreferences;
use windows_registry::{Key, CURRENT_USER};
use windows_result::HRESULT;

pub struct RegistryBackedPreferences {
    app_key_path: String,
    /// Caches the `HKCU\Software\Warp\<channel>` registry Key handle.
    ///
    /// At startup Warp sequentially calls `read_value` for ~100 settings;
    /// opening/creating the Key via `CURRENT_USER.create(...)` each time is a ~3ms synchronous syscall,
    /// adding up to 300ms+ (the bulk of the cold-start `READ_USER_DEFAULTS_AND_INITIALIZE_SETTINGS` phase).
    /// Here the Key from the first successful open is cached and reused by later reads, saving N-1 syscalls.
    ///
    /// Uses `Mutex<Option<Key>>` rather than `OnceLock`, because `windows_registry::Key`
    /// does not implement `Clone`, so a mutable lock is needed for `replace`/`take`; also the `read_value` interface is
    /// `&self`, so `RefCell` (which would need Sync) cannot be used.
    cached_key: Mutex<Option<Key>>,
}

static WARP_REGISTRY_BASE_PATH: &str = "Software\\Warp\\";
pub const KEY_NOT_FOUND_ERR: HRESULT = HRESULT::from_win32(0x80070002);

impl RegistryBackedPreferences {
    /// Construct a separate registry path for each channel (stable, dev, local, etc.)
    pub fn new(app_name: &str) -> Self {
        let app_key_path = WARP_REGISTRY_BASE_PATH.to_owned() + app_name;
        // Warm up the Key at startup so even the first setting read avoids the synchronous syscall.
        // A warm-up failure is not an error: `with_warp_registry` retries when needed.
        let initial_key = CURRENT_USER
            .create(app_key_path.clone())
            .inspect_err(|e| {
                log::warn!("warp registry key prewarm failed (will retry on first access): {e:#}");
            })
            .ok();
        Self {
            app_key_path,
            cached_key: Mutex::new(initial_key),
        }
    }

    /// Operates on the cached Warp registry Key through a callback. The first call does `CURRENT_USER.create(...)`,
    /// later calls reuse it directly. If the Key lock is poisoned (an earlier panic), it falls back to a one-off create
    /// without caching -- behavior degrades but it does not panic further.
    fn with_warp_registry<R>(
        &self,
        f: impl FnOnce(&Key) -> Result<R, super::Error>,
    ) -> Result<R, super::Error> {
        let mut guard = match self.cached_key.lock() {
            Ok(g) => g,
            // Mutex poisoned: take the one-off create path, no caching, behavior equivalent to the original.
            Err(_) => {
                let key = CURRENT_USER
                    .create(self.app_key_path.clone())
                    .map_err(|e| {
                        log::error!("unable to access Warp app key in Windows Registry: {e:#}");
                        super::Error::IoError(io::Error::from(e))
                    })?;
                return f(&key);
            }
        };

        if guard.is_none() {
            let key = CURRENT_USER
                .create(self.app_key_path.clone())
                .map_err(|e| {
                    log::error!("unable to access Warp app key in Windows Registry: {e:#}");
                    super::Error::IoError(io::Error::from(e))
                })?;
            *guard = Some(key);
        }

        // The guard is necessarily Some here; unwrap is safe.
        f(guard.as_ref().expect("cached_key must be Some after init"))
    }
}

impl UserPreferences for RegistryBackedPreferences {
    fn read_value(&self, name: &str) -> Result<Option<String>, super::Error> {
        self.with_warp_registry(|key| Ok(key.get_string(name).ok()))
    }

    fn write_value(&self, key: &str, value: String) -> Result<(), super::Error> {
        self.with_warp_registry(|reg| {
            reg.set_string(key, value.as_str())
                .map_err(|e| super::Error::from(io::Error::from(e)))
        })
    }

    fn remove_value(&self, key: &str) -> Result<(), super::Error> {
        self.with_warp_registry(|reg| match reg.remove_value(key) {
            Ok(_) => Ok(()),
            // If the key doesn't exist, then treat removal of that nonexistent key as a success.
            Err(e) if e.code() == KEY_NOT_FOUND_ERR => Ok(()),
            Err(e) => Err(super::Error::from(io::Error::from(e))),
        })
    }
}
