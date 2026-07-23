# Design note: battery-powered display node (periodic wake)

Status: design/pending hardware. Captured before the Raspberry Pi Zero 2 W and
PiSugar 2 arrive so the plan is not lost. No core code change is required yet.

## Target deployment

- **Control node:** `oasurgerydev` (always-on) runs discovery, generation
  (Claude/Gemini backend), review, and the HTTP catalog service.
- **Display node:** Raspberry Pi Zero 2 W behind the frame, **battery-powered
  via a PiSugar 2**. It is normally powered off. The PiSugar's RTC wakes it on a
  schedule (target: ~3x/day). On each wake it performs a single image rotation,
  then powers down again.

## Why the application already fits this model

The display role was built as a stateless one-shot, which is exactly what a
timed wake wants to invoke:

- **`display-cycle` is `Type=oneshot`** (`installation.py`): fetch the approved
  catalog, select the next plate, verify checksum, render to the panel, report
  success (`GET /v1/display-success`), then exit. It is not a long-running loop.
- **Rotation state persists to `state.json`** (`display_node.py`): `next_index`,
  `shuffle_bag_remaining`, `shuffle_bag_seen`, etc. are read and rewritten every
  cycle, so `sequential` / `shuffle_bag` / `weighted` rotation advances correctly
  across power cycles. Rebooting between rotations is expected and safe.
- **E-paper is bistable.** The Spectra 6 panel holds the last plate at zero power
  while the Pi is asleep. A few updates per day with the Pi off in between is the
  intended use case for this hardware, not a workaround.

So the core generation/display logic needs **no change** for battery operation.

## What actually changes

`schedule.rotation_minutes` currently drives two things: the display's systemd
timer interval, and the controller's `display_stale` alarm. Only the items below
need attention.

1. **Trigger: interval timer -> boot-fired oneshot (the one code-adjacent change).**
   The installer generates a `.timer` with
   `OnUnitActiveSec = rotation_minutes * 60` (`installation.py`), i.e. an
   always-on interval. A powered-off Pi cannot run that. Battery mode reuses the
   `oneshot` **service unchanged** but replaces the trigger so the cycle runs once
   shortly after each boot — e.g. a `.timer` with `OnBootSec=<delay>`, or a
   service `WantedBy=multi-user.target`. This belongs in a **"battery" display
   install mode** in `installation.py` / `deploy/install-display-node.sh`. It is
   generally useful (battery frames are a common want), so it is upstreamable —
   implement it as a clean feature branch off `main`, like `intl-postcodes`.

2. **`display_stale` calibration (config only, no code).** The controller alarms
   if no completed update is reported within
   `max(3 * rotation_minutes, 60 min)` (`notifications.py`). At the default
   30 min that is a 90-minute window, which would false-fire constantly against
   an 8-hourly display. Set `rotation_minutes` to the wake cadence
   (~480 for 3x/day) so the window becomes ~24 h and the alarm only fires after
   roughly three missed wakes.

3. **PiSugar wake + shutdown (Pi-side scripts, not app code).** Two ops steps:
   `shutdown` after a successful cycle, and setting the wake schedule. PiSugar 2's
   scheduled-wake firmware is oriented around a single RTC alarm, so three
   distinct times per day is most reliably done by **setting the next alarm on
   each boot** (via `pisugar-server`) before shutting down, rather than a native
   three-slot schedule. Confirm against the installed PiSugar firmware.

## Failure behavior (already handled)

- If `oasurgerydev` is unreachable when the Pi wakes, the cycle fails, the panel
  keeps its current plate (bistable), and the next wake retries. No state is
  corrupted — `display-cycle` validates local state before fetching.
- A checksum mismatch causes the display to refuse the asset and keep current
  state.

## Decision

- **Core code:** no change.
- **Now:** nothing urgent; `rotation_minutes` is a one-line config set at deploy
  time.
- **When the hardware arrives:** implement the "battery" display install mode
  (feature branch off `main`, testable on the real Pi + PiSugar, optional upstream
  PR), or hand-roll the boot service + PiSugar next-alarm/shutdown scripts on the
  Pi directly.
