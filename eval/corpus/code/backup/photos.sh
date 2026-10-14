#!/usr/bin/env bash
# Nightly one-way copy of ~/Pictures to the NAS in the hall closet.
# Runs from cron at 02:30. Logs go to ~/.local/state/photos-backup.log
set -euo pipefail

SRC="$HOME/Pictures/"
NAS_USER="backup"
NAS_HOST="nas-closet.local"
DEST="/volume1/photos-mirror/"
LOG_DIR="$HOME/.local/state"
LOG="$LOG_DIR/photos-backup.log"
LOCK="/tmp/photos-backup.lock"

# skip junk that the phone and the photo editor leave behind
EXCLUDES=(
  --exclude '*.tmp'
  --exclude '.thumbnails/'
  --exclude '.DS_Store'
  --exclude 'Screenshots/'
)

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

# only one copy at a time, cron sometimes overlaps after a slow night
if [ -e "$LOCK" ]; then
  log "lock file exists, another run is still going, exiting"
  exit 0
fi
trap 'rm -f "$LOCK"' EXIT
touch "$LOCK"

mkdir -p "$LOG_DIR"

if ! ping -c 1 -W 2 "$NAS_HOST" >/dev/null 2>&1; then
  log "NAS not reachable, skipping tonight"
  exit 0
fi

log "starting copy of $SRC"
# -a keeps times and permissions, -z compresses over the wifi link
# no --delete on purpose: photos deleted on the laptop should stay on the NAS
rsync -az --partial --info=stats1 "${EXCLUDES[@]}" \
  "$SRC" "$NAS_USER@$NAS_HOST:$DEST" >> "$LOG" 2>&1
status=$?

if [ "$status" -eq 0 ]; then
  log "done, $(du -sh "$SRC" | cut -f1) in source"
else
  log "rsync failed with status $status"
fi
exit "$status"
