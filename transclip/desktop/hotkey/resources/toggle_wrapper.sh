#!/bin/sh
set -u

LOG=@@LOG@@
STATE=@@STATE@@
BASE=@@BASE@@
MAX_SECONDS=@@MAX_SECONDS@@
STALE_LOCK_SECONDS=@@STALE_LOCK_SECONDS@@
LOCK=/tmp/transclip-toggle.lock

mkdir -p "$(dirname "$LOG")"
mkdir -p "$(dirname "$STATE")"

write_state() {
  state=$1
  detail=$2
  printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$state" "$detail" > "$STATE"
  printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') state=${state} ${detail}" >> "$LOG"
}

kill_process_tree() {
  pid=$1
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_process_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

clear_stale_lock_owner() {
  owner_pid=$(cat "$LOCK/pid" 2>/dev/null || true)
  case "$owner_pid" in
    ''|*[!0-9]*)
      return
      ;;
  esac
  owner_command=$(ps -p "$owner_pid" -o command= 2>/dev/null || true)
  case "$owner_command" in
    *transclip-toggle*)
      write_state recovering "Clearing stale action"
      printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') killing stale TransClip action pid=${owner_pid}" >> "$LOG"
      kill_process_tree "$owner_pid"
      ;;
  esac
}

printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') wrapper invoked" >> "$LOG"
write_state shortcut "Starting recording"

if ! mkdir "$LOCK" 2>/dev/null; then
  now=$(date +%s)
  lock_mtime=$(stat -f %m "$LOCK" 2>/dev/null || printf '0')
  lock_age=$((now - lock_mtime))
  if [ "$lock_age" -lt "$STALE_LOCK_SECONDS" ]; then
    write_state busy "Previous action still running"
    printf '%s\n'       "$(date '+%Y-%m-%dT%H:%M:%S%z') ignored: previous TransClip action still running (${lock_age}s)"       >> "$LOG"
    exit 0
  fi
  printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') clearing stale TransClip action lock (${lock_age}s)" >> "$LOG"
  clear_stale_lock_owner
  rm -rf "$LOCK"
  if ! mkdir "$LOCK" 2>/dev/null; then
    printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') ignored: could not acquire TransClip action lock" >> "$LOG"
    exit 0
  fi
fi
printf '%s\n' "$$" > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT HUP INT TERM

response=$(curl -sS --max-time 10 -X POST "$BASE/record/start"   -H 'content-type: application/json' --data '{}' 2>>"$LOG") || {
  write_state error "Start failed"
  printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') start failed; restarting service" >> "$LOG"
  @@RESTART_COMMAND@@
  exit 0
}
printf '%s\n' "$response" >> "$LOG"
case "$response" in
  *'"already_recording": true'*|*'"already_recording":true'*)
    already_recording=1
    ;;
  *)
    already_recording=0
    ;;
esac
printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') start already_recording=${already_recording}" >> "$LOG"

if [ "$already_recording" = "1" ]; then
  write_state transcribing "Transcribing"
  response=$(curl -sS --max-time "$MAX_SECONDS" -X POST "$BASE/record/stop"     -H 'content-type: application/json' --data '{}' 2>>"$LOG") || {
    write_state error "Stop timed out; restarted"
    printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') stop failed; restarting service" >> "$LOG"
    @@RESTART_COMMAND@@
    exit 0
  }
  printf '%s\n' "$response" >> "$LOG"
  text=$(printf '%s' "$response" | @@PYTHON@@ -c     'import json,sys; print(json.load(sys.stdin).get("text", ""), end="")' 2>>"$LOG")
  if [ -n "$text" ]; then
    write_state paste_requested "Paste transcript"
    printf '%s' "$text" | pbcopy
    printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') copied transcript chars=${#text}" >> "$LOG"
  else
    write_state finished "No transcript"
    printf '%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') stop returned no text" >> "$LOG"
  fi
else
  write_state listening "Recording"
fi

exit 0
