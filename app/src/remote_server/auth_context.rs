use std::sync::Arc;

use remote_server::auth::RemoteServerAuthContext;
use warpui::r#async::BoxFuture;

use crate::auth::AuthState;

/// Builds the auth context used by the remote-server module.
///
/// Warp Wave 3-1: the `AuthClient` trait was physically deleted. The Bearer token source now reads directly
/// `AuthState::get_access_token_ignoring_validity()` (on the Warp path it returns `Some` only when the user has attached a
/// BYOP API key, and always `None` otherwise).
pub fn server_api_auth_context(auth_state: Arc<AuthState>) -> RemoteServerAuthContext {
    let token_auth_state = auth_state.clone();
    let identity_auth_state = auth_state;

    RemoteServerAuthContext::new(
        move || -> BoxFuture<'static, Option<String>> {
            let token = token_auth_state.get_access_token_ignoring_validity();
            Box::pin(async move { token })
        },
        move || remote_server_identity_key(&identity_auth_state),
    )
}

fn remote_server_identity_key(auth_state: &AuthState) -> String {
    // Warp no longer distinguishes anonymous / signed-in identities and uniformly uses `user_id()` (a local test UID).
    auth_state
        .user_id()
        .map(|uid| uid.as_string())
        .unwrap_or_else(|| auth_state.anonymous_id())
}
