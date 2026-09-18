"""Usage: regen-fix.py <taxon_id> <correction finding>  -- regenerate through the human gate.

Generates a fresh candidate whose first attempt already carries the human's
correction, leaves it in pending, and queues the pending notification.
"""
import sys
from pathlib import Path

from inky_bird_frame.config import NotificationEvent, load_config
from inky_bird_frame.controller import _read_discovery_snapshot, generate_candidate
from inky_bird_frame.notifications import safe_notify
from inky_bird_frame.cli import _pending_notification_links

taxon = int(sys.argv[1])
finding = " ".join(sys.argv[2:]).strip()
if not finding:
    raise SystemExit("a correction finding is required")
cfg = load_config(Path("/data/config.toml"))
species = next((s for s in _read_discovery_snapshot(cfg).species if s.taxon_id == taxon), None)
if species is None:
    raise SystemExit(f"[fix {taxon}] taxon not in discovery snapshot")
print(f"[fix {taxon}] correction: {finding}", flush=True)
candidate = generate_candidate(
    cfg, species, cfg.controller.workspace_dir, initial_correction_findings=(finding,)
)
print(f"[fix {taxon}] pending at {candidate}", flush=True)
if "--no-notify" not in sys.argv:
    image_url, actions = _pending_notification_links(cfg.controller.notify_base_url, taxon)
    if image_url is not None:
        body = (
            f"A corrected {species.common_name} plate passed automated review. "
            "Tap Approve or Reject below, or open the plate to see it full size."
        )
        attachment = None
    else:
        body = (
            f"A corrected {species.common_name} plate passed automated review. "
            f"Approve: inky-bird-frame approve {taxon}. "
            f'Reject: inky-bird-frame reject {taxon} --reason "...".'
        )
        attachment = str(candidate / "portrait.png")
    result = safe_notify(
        cfg,
        NotificationEvent.GENERATION_PENDING,
        dedupe_key=f"pending:{taxon}:{int(candidate.stat().st_mtime)}",
        title=f"{species.common_name} plate awaiting your approval",
        body=body,
        attachment=attachment,
        image_url=image_url,
        actions=actions,
    )
    print(f"[fix {taxon}] notification queued: {result.get('queued')}", flush=True)
