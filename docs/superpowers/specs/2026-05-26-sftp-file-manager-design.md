# SFTP File Manager Design Document

**Date**: 2026-05-26
**Status**: Approved

## Overview

Add a native SFTP file manager to the Warp terminal. It uses the `ssh2` crate (libssh2 bindings) to implement the SFTP protocol and provides full remote file browsing, transfer and management. It is implemented as a standalone Pane, coexists with the existing Server File Browser, and does not require installing a remote daemon.

## Technical Approach

Implement the SFTP protocol directly with the `ssh2` crate, which has proven stable in similar projects and covers the full SFTP feature set (directory traversal, streaming transfer, permission management).

Dependencies: `ssh2` (libssh2 bindings), `smol` (async runtime), `thiserror` (error handling). On Windows, enable the `openssl-on-win32` feature and vendored openssl-sys.

## Crate Structure and Module Organization

### Protocol layer — `crates/warp_sftp/` (new crate)

```
crates/warp_sftp/
  Cargo.toml
  build.rs                          # Link advapi32 on Windows
  src/
    lib.rs                          # Module root, exports the public API
    error.rs                        # SftpError / SftpChannelError
    types.rs                        # FileType / Metadata / DirEntry / OpenOptions, etc.
    session.rs                      # SftpSession (SSH connection management, authentication)
    sftp.rs                         # Sftp (SFTP channel, file/directory operations)
    dir.rs                          # Dir (directory reading and sorting)
    file.rs                         # File (file read/write)
```

### UI layer — `app/src/sftp_manager/` (new module)

```
app/src/sftp_manager/
  mod.rs                            # Module root
  types.rs                          # UI types: FileEntry / TransferTask / Dialog / ConnectionState
  sftp_ops.rs                       # High-level operation bridge
  browser.rs                        # SftpBrowserView main view
  file_list.rs                      # File list rendering
  breadcrumb.rs                     # Breadcrumb navigation
  context_menu.rs                   # Context menu
  dialogs.rs                        # Dialogs
  transfer_panel.rs                 # Transfer progress panel
```

### Pane integration

```
app/src/pane_group/pane/sftp_pane.rs (new)
```

## Core Protocol Layer Design

### session.rs — connection management

- `SftpSession`: internally holds `Arc<ssh2::Session>` + `TcpStream`
- `connect(host, port, username, auth_method) -> Result<SftpSession>`: establish the TCP connection → SSH handshake → authentication
- `sftp() -> Result<Sftp>`: open the SFTP subsystem on the existing session
- `disconnect()`: disconnect explicitly
- `Drop` disconnects automatically

`AuthMethod` enum: `Password(String)` | `PublicKey { path, passphrase }`

### sftp.rs — SFTP channel operations

- `Sftp`: wraps `Arc<Mutex<ssh2::Sftp>>`, Clone + thread-safe
- Operations: `open`, `create_dir`, `remove_dir`, `remove_file`, `rename`, `stat`, `lstat`, `read_dir`, `symlink`, `readlink`

### dir.rs — directory reading

- `Dir::read_dir() -> Result<Vec<DirEntry>>`
- Filter out `.` and `..`, convert to DirEntry
- Sorting: directories first, then alphabetical

### file.rs — file read/write

- `File`: wraps `ssh2::File`
- Operations: `read_to_end`, `write_all`, `read` (32KB chunks), `write` (32KB chunks), `flush`, `stat`

### types.rs — core types

- `FileType`: Dir | File | Symlink | Other
- `FilePermissions`: 9-bit Unix permissions (rwxrwxrwx)
- `Metadata`: type, perms, size, uid, gid, atime, mtime
- `DirEntry`: name, path, metadata
- `OpenOptions`: read, write, append, create, truncate
- `WriteMode`: Overwrite | Append | Resume

### error.rs — error types

- `SftpError`: IO | SSH2 | ConnectionFailed | AuthFailed | Timeout | NoSuchFile | PermissionDenied | General
- `SftpChannelError`: Sftp | SendFailed | RecvFailed

## UI Layer Design

### browser.rs — SftpBrowserView main view

Implements the `BackingView` + `TypedActionView` + `View` traits.

**State**:

| Field | Type | Description |
|------|------|------|
| connection | ConnectionState | Connecting/Connected/Disconnected/Failed |
| _session | Option\<SftpSession\> | Keeps the TCP connection alive |
| sftp | Option\<Sftp\> | SFTP channel |
| current_path | String | Current directory path |
| entries | Vec\<FileEntry\> | File list of the current directory |
| selection | Option\<usize\> | Index of the selected item |
| nav_history | NavHistory | Back/forward history |
| transfers | Vec\<TransferTask\> | Transfer queue |
| dialog | Option\<Dialog\> | Current dialog state |
| search_filter | Option\<String\> | Search filter |

**Action enum**:

- Connect(node_id), Disconnect
- NavigateTo(path), GoBack, GoForward, GoUp, Refresh
- Upload, Download, Delete, Rename, CreateFolder
- Select(index), Open(index)
- ShowContextMenu(index)
- CancelTransfer(task_id)
- Search(filter)

**Render structure** (top to bottom):

1. Toolbar: back/forward/up/refresh buttons + upload button + new folder button
2. Breadcrumb navigation: clickable path segments
3. File list: table style (name/size/modified date), click to select, double-click to open
4. Transfer panel: collapsible at the bottom, shows active transfer tasks and progress
5. Context menu: open/download/rename/delete/details
6. Dialogs: delete confirmation, rename input, new folder input, file details

### sftp_ops.rs — high-level operation bridge

- `connect_from_server(server_info, secret_store) -> Result<(SftpSession, Sftp)>`: read the config from the SSH manager → fetch credentials → establish the connection
- `list_dir(sftp, path) -> Result<Vec<FileEntry>>`
- `upload_file_streaming(sftp, local, remote, cancel_flag)`: 32KB chunks, AtomicBool supports cancellation
- `download_file_streaming(sftp, remote, local, cancel_flag)`: 32KB chunks
- `upload_dir_recursive`, `download_dir_recursive`
- `delete_file`, `delete_dir_recursive`, `create_dir`, `rename`
- Concurrency control: AtomicUsize CAS limits transfers to at most 2 in parallel

### Other UI modules

| Module | Responsibility |
|------|------|
| `file_list.rs` | File table header + row rendering, directory/file icons, hover effect, selection highlight |
| `breadcrumb.rs` | Clickable segments from the root to the current path, each segment triggers NavigateTo |
| `context_menu.rs` | Context menu items: open/download/rename/delete/details |
| `dialogs.rs` | Modal dialogs, EditorView text input, Enter to confirm / Escape to cancel |
| `transfer_panel.rs` | Transfer direction icon + file name + progress percentage + progress bar + status label |

### Keyboard shortcuts

| Key | Action |
|------|------|
| Backspace | Go to the parent directory |
| Delete | Delete the selected item |
| Ctrl+Shift+N | New folder |
| Escape | Cancel search / close dialog |

## Integration and Entry Points

### Integration with the SSH manager

The SFTP browser obtains connection information through `warp_ssh_manager`.

Entry points:

- `app/src/ssh_manager/panel.rs`: add an "SFTP Browse" option to the server context menu
- `app/src/ssh_manager/server_view.rs`: add an "SFTP Browse" button to the server detail action bar

Connection flow:

1. The user right-clicks a server in the SSH host list → menu item "SFTP Browse"
2. Get SshServerInfo (host, port, username, auth_type, key_path)
3. Get the password/key passphrase through KeychainSecretStore
4. Build AuthMethod
5. SftpOps::connect_from_server() establishes the connection
6. Open the SftpBrowserView Pane and show the root directory

### Pane system integration

- `app/src/pane_group/pane/sftp_pane.rs` (new): `SftpPane` wraps `SftpBrowserView` as `PaneContent`
  - Implements the PaneContent trait
  - Snapshot serializes to `LeafContents::Sftp { node_id }`
  - Reconnects automatically from node_id on restore

Registration changes:

- `app/src/pane_group/pane/mod.rs`: declare the sftp_pane module
- `app/src/lib.rs`: register SftpPane with the View system

### Feature Flag

No Feature Flag is used; the feature is always available globally.

## Data Flow and Error Handling

### Operation data flow

```
User action (click/right-click/shortcut)
  → SftpBrowserView receives an Action
  → dispatch_typed_action() matches the Action type
  → Async task submitted via ctx.spawn():
      ├── Get the SftpOps / Sftp instance
      ├── Run the SFTP operation (on the smol thread pool)
      └── Return the result to the main thread
  → Update SftpBrowserView state
  → Trigger re-render
```

### Connection lifecycle

```
Open Pane → Connect(node_id)
  → Connecting state (show loading animation)
  → Success → Connected (load the root directory)
  → Failure → Failed (show error message + retry button)

Close Pane → Drop
  → SftpSession disconnects automatically (Drop impl)
```

### Error handling strategy

| Scenario | Handling |
|------|----------|
| Connection failure (network/auth) | Show the error message and a retry button, no popup |
| Directory load failure | Show an error message + refresh button in the file list area |
| File operation failure (delete/rename) | Inline error message (Toast style), does not block the UI |
| Transfer failure | The transfer panel marks the task as Failed and shows the error reason |
| Connection lost | Switch automatically to the Disconnected state and prompt to reconnect |

All errors are mapped uniformly through `SftpError` to user-readable messages.
