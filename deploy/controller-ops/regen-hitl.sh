#!/bin/sh
# Usage: regen-hitl.sh <taxon_id>...  -- regenerate approved plates THROUGH the human gate.
# For each taxon: archive+remove the approved entry, generate a new candidate (stays pending),
# send the pending notification, and deliver it now. The notification carries the same inline
# image + Approve/Reject buttons as the scheduler's own pending pushes (via
# cli._pending_notification_links), falling back to CLI instructions when no notify_base_url.
# Nothing is approved here; the human runs `approve <taxon>` / `reject <taxon> --reason`.
CFG=/data/config.toml; ARCH=/data/var/controller/archive; mkdir -p "$ARCH"
for T in "$@"; do
  D=$(ls -d /data/catalog/species/${T}-* 2>/dev/null | head -1)
  if [ -n "$D" ]; then
    echo "[hitl $T] archiving $(basename "$D")"; cp -r "$D" "$ARCH/$(basename "$D")-pre-fixedfont-$(date +%s)"; rm -rf "$D"
  else
    echo "[hitl $T] no approved entry (fresh generation)"
  fi
  python - "$T" <<'PY' || { echo "[hitl $T] FAILED"; continue; }
import sys
from pathlib import Path
from inky_bird_frame.config import NotificationEvent, load_config
from inky_bird_frame.controller import _read_discovery_snapshot, generate_candidate
from inky_bird_frame.notifications import safe_notify
from inky_bird_frame.cli import _pending_notification_links
t = int(sys.argv[1]); cfg = load_config(Path("/data/config.toml"))
sp = next((s for s in _read_discovery_snapshot(cfg).species if s.taxon_id == t), None)
if sp is None:
    raise SystemExit(f"[hitl {t}] taxon not in discovery snapshot")
cand = generate_candidate(cfg, sp, cfg.controller.workspace_dir)
print(f"[hitl {t}] pending at {cand}")
image_url, actions = _pending_notification_links(cfg.controller.notify_base_url, t)
if image_url is not None:
    body = (f"A new {sp.common_name} plate passed automated review and is waiting for you. "
            "Tap Approve or Reject below, or open the plate to see it full size.")
    attachment = None
else:
    body = (f"A new {sp.common_name} plate passed automated review and is waiting for you. "
            f"Approve: inky-bird-frame approve {t}. Reject: inky-bird-frame reject {t} --reason \"...\".")
    attachment = str(cand / "portrait.png")
r = safe_notify(cfg, NotificationEvent.GENERATION_PENDING,
    dedupe_key=f"pending:{t}:{int(cand.stat().st_mtime)}",
    title=f"{sp.common_name} plate awaiting your approval",
    body=body, attachment=attachment, image_url=image_url, actions=actions)
print(f"[hitl {t}] notification queued: {r.get('queued')} (buttons: {actions is not None})")
PY
  inky-bird-frame notifications dispatch --config "$CFG" | head -c 200; echo
  echo "[hitl $T] DONE (pending, awaiting approval)"
done
echo "[hitl-all] DONE"
