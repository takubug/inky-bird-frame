# Controller ops scripts

Helpers that live in the controller's `/data` volume and run in the scheduler's
env (which holds the API keys). Run them with:

```sh
docker compose run --rm --no-deps -T --entrypoint python  scheduler /data/<script>.py ...
docker compose run --rm --no-deps -T --entrypoint sh      controller /data/<script>.sh ...
```

Use `--no-deps` so the `bootstrap` service does not re-run.

| Script | Purpose |
|---|---|
| `regen-hitl.sh <taxon>...` | Archive an approved plate, generate a fresh candidate through the human gate (stays pending), and send the pending ntfy (inline image + Approve/Reject buttons via `cli._pending_notification_links`). |
| `recomposite.py <taxon> [--no-notify] [--from-failed [--attempt N]]` | Rebuild a pending candidate's labels from the retained text-free raw illustration (no re-render). `--from-failed` salvages a failed run's best/chosen attempt and re-runs only the review. |
| `regen-fix.py <taxon> "<finding>" [--no-notify]` | Regenerate through the gate with a human correction fed in as the first-attempt `initial_correction_findings`. |
| `demote.sh <taxon>` | Archive an approved plate and remove it from the catalog, then refresh the active catalog so it stops being served. Reversible via the archive copy. |

All pending-notification paths use the same `cli._pending_notification_links`
builder as the scheduler, so manual regenerations get the inline image and
Approve/Reject buttons when `[controller] notify_base_url` is set.

These are copies of the deployed volume scripts, kept here for version control;
the running copies live at `/data/*.sh` and `/data/*.py` on the controller.
