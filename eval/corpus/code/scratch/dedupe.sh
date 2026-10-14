#!/usr/bin/env bash
# find identical files under a folder by content hash and list the extras
# usage: ./dedupe.sh ~/Downloads [--delete]
set -eu

root="${1:?give a folder}"
mode="${2:-list}"
tmp="$(mktemp)"
trap 'rm -f "$tmp" "$tmp.dupes"' EXIT

# hash every regular file, skip tiny ones, they're not worth it
find "$root" -type f -size +4k -print0 \
  | xargs -0 md5sum 2>/dev/null \
  | sort > "$tmp"

# group by hash; the first file in each group is the one we keep
awk '
  {
    hash = $1
    path = substr($0, index($0, "  ") + 2)
    if (hash in seen) {
      print path
    } else {
      seen[hash] = path
    }
  }
' "$tmp" > "$tmp.dupes"

count=$(wc -l < "$tmp.dupes")
echo "found $count duplicate files under $root"

if [ "$mode" = "--delete" ]; then
  # move to a dated trash folder instead of rm, in case I'm wrong
  trash="$HOME/.dedupe-trash/$(date +%Y%m%d)"
  mkdir -p "$trash"
  while IFS= read -r f; do
    mv -n -- "$f" "$trash/"
  done < "$tmp.dupes"
  echo "moved to $trash"
else
  cat "$tmp.dupes"
fi
