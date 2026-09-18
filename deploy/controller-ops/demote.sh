#!/bin/sh
# Demote one approved plate: archive it, remove it from the catalog, refresh the
# active catalog so it is no longer served. Does NOT regenerate. Reversible via
# the archived copy. Usage: demote.sh <taxon_id>
set -e
T="$1"
[ -n "$T" ] || { echo "usage: demote.sh <taxon_id>"; exit 1; }
CFG=/data/config.toml
ARCH=/data/var/controller/archive
mkdir -p "$ARCH"
D=$(ls -d /data/catalog/species/${T}-* 2>/dev/null | head -1)
[ -n "$D" ] || { echo "[demote $T] not in catalog"; exit 1; }
STAMP=$(date +%s)
cp -r "$D" "$ARCH/$(basename "$D")-demoted-$STAMP"
rm -rf "$D"
echo "[demote $T] archived to $ARCH/$(basename "$D")-demoted-$STAMP and removed from catalog"
inky-bird-frame refresh --config "$CFG" >/dev/null 2>&1 && echo "[demote $T] refreshed active catalog"
