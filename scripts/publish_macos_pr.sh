#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

branch="macos-support"
base="master"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install with: brew install gh" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Log in first: gh auth login" >&2
  exit 1
fi

git push -u origin "$branch"

gh pr create --base "$base" --head "$branch" \
  --title "macOS support: LaunchAgent, paths, and doctor checks" \
  --body-file "$repo_root/scripts/macos_pr_body.md"
