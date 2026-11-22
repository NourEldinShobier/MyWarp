#!/usr/bin/env bash
# Zap website deploy script (runs on the server)
#
# Scenario: the site is hosted by 1Panel and the nginx root is fixed to the site directory (e.g.
#   /opt/1panel/www/sites/zap.zerx.dev/index), so the symlink approach (changing root) is not possible.
# So this script does an "atomic directory swap": CI has already rsynced the latest build to <site>/.incoming,
# and this script renames .incoming to the live index directory, keeping the previous version for rollback.
#
# Layout convention ($DEPLOY_PATH = site index directory, e.g. .../zap.zerx.dev/index):
#   <parent>/.incoming   ← latest dist contents pushed by CI rsync
#   <parent>/index       ← directory actually served by nginx (= $DEPLOY_PATH)
#   <parent>/.index.bak  ← previous version, for rollback
set -euo pipefail

# $DEPLOY_PATH is passed in by the caller (SSH command) as the first argument, i.e. the live index directory
INDEX="${1:?usage: deploy.sh <site-index-dir>}"
PARENT="$(dirname "$INDEX")"
INCOMING="$PARENT/.incoming"
BACKUP="$PARENT/.index.bak"

log() { printf '[deploy] %s\n' "$*"; }

if [ ! -d "$INCOMING" ] || [ -z "$(ls -A "$INCOMING" 2>/dev/null)" ]; then
  log "error: $INCOMING does not exist or is empty, nothing to publish (CI must rsync to .incoming first)."
  exit 1
fi

# Remove the previous backup and move the current index to backup (if it exists)
if [ -e "$INDEX" ]; then
  log "backing up current version -> $BACKUP"
  rm -rf "$BACKUP"
  mv -Tf "$INDEX" "$BACKUP"
fi

# Atomic go-live: rename .incoming -> index (same filesystem, mv is atomic)
log "publishing new version -> $INDEX"
mv -Tf "$INCOMING" "$INDEX"

log "done. Current serving directory: $INDEX"
log "to roll back: rm -rf '$INDEX' && mv -Tf '$BACKUP' '$INDEX'"
