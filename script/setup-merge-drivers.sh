#!/usr/bin/env bash
# Register the openWarp custom merge driver and enable rerere.
# Run once after the first clone; later upstream merges (merge / cherry-pick / rebase) will then:
# 1. keep the local version of paths marked merge=warp-ours in .gitattributes
# 2. let rerere record each conflict resolution and reuse it for the same conflict next time
set -euo pipefail

git config merge.warp-ours.name "Always keep openWarp version (custom driver)"
git config merge.warp-ours.driver true
git config rerere.enabled true
git config rerere.autoupdate true

echo "openWarp merge drivers + rerere configured."
echo "  rerere.enabled        = $(git config --get rerere.enabled)"
echo "  rerere.autoupdate     = $(git config --get rerere.autoupdate)"
echo "  merge.warp-ours   = $(git config --get merge.warp-ours.driver)"
