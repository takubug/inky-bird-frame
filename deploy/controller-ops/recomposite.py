"""Re-composite a candidate's labels from its retained text-free illustration (no re-render).

Usage: recomposite.py <taxon_id> [--no-notify] [--from-failed [--attempt N]]
  --from-failed  salvage: take the newest FAILED run's best attempt raw illustration,
                 composite it with the current rules, re-run ONLY the review (no render),
                 and create a pending candidate if it passes.
Rebuilds a fresh PENDING candidate from the most recent rejected/pending candidate's
manifest and the run's raw illustration, using the current compositor, then notifies.
"""
import json, shutil, sys, time
from pathlib import Path
from tempfile import TemporaryDirectory
import inky_bird_frame.claude_runner as cr
from inky_bird_frame.catalog import candidate_directory, find_taxon_directory, write_candidate_manifest
from inky_bird_frame.config import NotificationEvent, load_config
from inky_bird_frame.controller import _read_discovery_snapshot, load_or_fetch_references
from inky_bird_frame.images import prepare_generated_plate
from inky_bird_frame.models import QualityReview
from inky_bird_frame.notifications import safe_notify
from inky_bird_frame.cli import _pending_notification_links
from inky_bird_frame.prompts import PROMPT_VERSION

t = int(sys.argv[1]); notify = "--no-notify" not in sys.argv[2:]; salvage = "--from-failed" in sys.argv[2:]
cfg = load_config(Path("/data/config.toml")); state = cfg.controller.state_dir
sp = next(s for s in _read_discovery_snapshot(cfg).species if s.taxon_id == t)
if salvage:
    from inky_bird_frame.controller import create_generation_runner, load_or_create_profile
    from inky_bird_frame.claude_runner import spectra_panel_preview
    failed = sorted(p for p in (state / "failed").glob(f"{t}-*") if p.is_dir())
    if not failed: raise SystemExit(f"[salvage {t}] no failed run")
    fdir = failed[-1]
    runs = sorted((state / "runs").glob(f"{t}-*")); run = runs[-1]
    # best attempt = highest composition among attempts (all failed on layout, so any is fine)
    best_att, best_c = 1, -1
    for q in sorted(fdir.glob("attempt-*/quality-review.json")):
        c = json.loads(q.read_text()).get("composition_quality", 0); n = int(q.parent.name.split("-")[1])
        if c > best_c: best_att, best_c = n, c
    if "--attempt" in sys.argv: best_att = int(sys.argv[sys.argv.index("--attempt") + 1])  # explicit pick
    raw = run / f"02-generation-attempt-{best_att:02d}-raw.png"
    if not raw.is_file(): raise SystemExit(f"[salvage {t}] no raw for attempt {best_att}: {raw}")
    refs = load_or_fetch_references(cfg, sp); root = state / "references" / str(t); ref_paths = [root / r.filename for r in refs]
    runner = create_generation_runner(cfg, cfg.controller.workspace_dir)
    with TemporaryDirectory(prefix=f"{t}-salvage-", dir=state / "work") as tmp:
        work = Path(tmp)
        profile, _ = load_or_create_profile(cfg, sp, refs, ref_paths, runner, work / "profile.json", work / "01-profile.log")
        from PIL import Image
        plate = Image.open(raw).convert("RGB"); cr.composite_plate_labels(plate, profile)
        composited = work / "generated.png"; plate.save(composited, format="PNG")
        attempt_dir = work / f"attempt-{best_att:02d}"; attempt_dir.mkdir()
        prepare_generated_plate(composited, attempt_dir / "portrait.png", attempt_dir / "display.png")
        logs = run  # keep review artefacts with the run
        review = runner.review_plate(sp, profile, refs, attempt_dir / "portrait.png", ref_paths,
                                     attempt_dir / "quality-review.json", logs / f"03-quality-review-salvage-{int(time.time())}.log",
                                     allowed_domains=cfg.research.allowed_domains)
        print(f"[salvage {t}] review passed={review.passed} comp={review.composition_quality} findings={len(review.findings)}")
        for f in review.findings[:4]: print("   -", f[:150])
        if not review.passed: raise SystemExit(f"[salvage {t}] review failed; not creating a candidate")
        (attempt_dir / "quality-review.json").write_text(json.dumps(review.as_dict(), indent=2))
        write_candidate_manifest(attempt_dir, sp, profile, refs, review, generator=cr.GENERATOR_LABEL,
                                 prompt_version=PROMPT_VERSION, attempt=best_att, max_attempts=cfg.controller.max_generation_attempts)
        dest = candidate_directory(state, sp)
        if dest.exists(): shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(attempt_dir, dest)
    print(f"[salvage {t}] pending at {dest} (from {fdir.name} attempt {best_att})")
    if notify:
        image_url, actions = _pending_notification_links(cfg.controller.notify_base_url, t)
        body = ("Re-composited from a retained illustration and re-reviewed. Tap Approve or Reject below."
                if image_url is not None
                else f"Re-composited from a retained illustration and re-reviewed. Approve: inky-bird-frame approve {t}. Reject: inky-bird-frame reject {t} --reason \"...\".")
        r = safe_notify(cfg, NotificationEvent.GENERATION_PENDING, dedupe_key=f"pending:{t}:{int(time.time())}",
            title=f"{sp.common_name} plate awaiting your approval (salvaged)",
            body=body, attachment=(None if image_url is not None else str(dest / "portrait.png")),
            image_url=image_url, actions=actions)
        print(f"[salvage {t}] notification queued: {r.get('queued')} (buttons: {actions is not None})")
    raise SystemExit(0)

# Source candidate: pending first, else the newest rejected copy.
src = find_taxon_directory(state / "pending", t)
if src is None:
    # Newest rejection by its manifest's rejected_at, not by directory name.
    cands = [p for p in (state / "rejected").glob(f"{t}-*") if (p / "manifest.json").is_file()]
    src = max(cands, key=lambda p: json.loads((p / "manifest.json").read_text()).get("rejected_at", ""), default=None)
if src is None:
    raise SystemExit(f"[recomp {t}] no pending or rejected candidate to rebuild from")
manifest = json.loads((src / "manifest.json").read_text())
gen = manifest.get("generation", {}); attempt = int(gen.get("attempt", 1))
# The run that produced this candidate: the newest run dir at or before its generated_at.
generated_at = str(gen.get("generated_at", "")).replace("-", "").replace(":", "")[:15]  # YYYYMMDDTHHMMSS
runs = sorted((state / "runs").glob(f"{t}-*"))
eligible = [r for r in runs if r.name.split("-", 1)[1][:15] <= generated_at] if generated_at else runs
run = (eligible or runs)[-1]
raw = run / f"02-generation-attempt-{attempt:02d}-raw.png"
if not raw.is_file():
    raise SystemExit(f"[recomp {t}] no retained raw illustration at {raw}")
profile = manifest["profile"]
refs = load_or_fetch_references(cfg, sp)
rv = manifest["quality_review"]
review = QualityReview(rv["passed"], rv["species_accuracy"], rv["anatomy_accuracy"], rv["text_accuracy"],
                       rv["composition_quality"], rv["location_free"], tuple(rv.get("findings", [])),
                       tuple(rv.get("verification_sources", [])))
with TemporaryDirectory(prefix=f"{t}-recomp-", dir=state / "work") as tmp:
    work = Path(tmp)
    from PIL import Image
    plate = Image.open(raw).convert("RGB")
    cr.composite_plate_labels(plate, profile)
    composited = work / "generated.png"; plate.save(composited, format="PNG")
    attempt_dir = work / f"attempt-{attempt:02d}"; attempt_dir.mkdir()
    prepare_generated_plate(composited, attempt_dir / "portrait.png", attempt_dir / "display.png")
    (attempt_dir / "quality-review.json").write_text(json.dumps(rv, indent=2))
    write_candidate_manifest(attempt_dir, sp, profile, refs, review, generator=gen.get("generator", cr.GENERATOR_LABEL),
                             prompt_version=gen.get("prompt_version", PROMPT_VERSION), attempt=attempt,
                             max_attempts=int(gen.get("max_attempts", cfg.controller.max_generation_attempts)))
    dest = candidate_directory(state, sp)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(attempt_dir, dest)
print(f"[recomp {t}] pending at {dest} (from {src.name}, raw {raw.name})")
if not notify:
    raise SystemExit(0)
image_url, actions = _pending_notification_links(cfg.controller.notify_base_url, t)
body = ("Labels re-composited on the same illustration. Tap Approve or Reject below."
        if image_url is not None
        else f"Labels re-composited on the same illustration. Approve: inky-bird-frame approve {t}. Reject: inky-bird-frame reject {t} --reason \"...\".")
r = safe_notify(cfg, NotificationEvent.GENERATION_PENDING, dedupe_key=f"pending:{t}:{int(time.time())}",
    title=f"{sp.common_name} plate awaiting your approval (re-laid-out)",
    body=body, attachment=(None if image_url is not None else str(dest / "portrait.png")),
    image_url=image_url, actions=actions)
print(f"[recomp {t}] notification queued: {r.get('queued')} (buttons: {actions is not None})")
