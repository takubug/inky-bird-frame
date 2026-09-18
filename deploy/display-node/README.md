# Display-node runtime scripts (Raspberry Pi Zero 2 W + Inky Impression 13.3")

Operational glue that runs on the battery display node, alongside the systemd
`inky-frame-battery-cycle.service` (which renders one plate on boot then powers
off after a ~120s grace, unless held).

## PiSugar button scripts

Installed to `/usr/local/bin/` and wired in `/etc/pisugar-server/config.json`
(pisugar-server runs as root):

| Gesture | `*_tap_shell` | Script | Effect |
|---|---|---|---|
| single | `single_tap_shell` | `inky-frame-tap-cycle` | render the next bird, hold this boot |
| double | `double_tap_shell` | `inky-frame-maintenance` | enter maintenance: set `STAY_AWAKE` (persists across boots) + `/run/inky-frame-hold` |
| long   | `long_tap_shell`   | `inky-frame-shutdown`   | clear `STAY_AWAKE` and power off -> normal daily cycle resumes |

`battery-cycle.sh` checks `STAY_AWAKE` at start (skip render+poweroff) and
re-checks both `STAY_AWAKE` and `/run/inky-frame-hold` right before poweroff, so
a double-tap during the grace window is honoured.

**Gotcha:** pisugar-server takes a few seconds after boot to start listening, so
a double-tap in the first moments of boot is missed. On a fresh wake, wait for
the bird to render before double-tapping. Single and long taps work anytime.

## Maintenance flag (manual)

- Enter:  `touch /home/inkybird/inky-bird-frame/STAY_AWAKE`
- Exit:   `rm -f /home/inkybird/inky-bird-frame/STAY_AWAKE`
- `auto_power_on` is null in the PiSugar config, so a poweroff sticks while
  USB-charging; `auto_wake_time` 05:00 (repeat 127) wakes it daily.

## SPI slowdown patch (signal-integrity)

The 13.3" panel's refresh on the impedance-sensitive Zero 2 W SPI bus can hang
(BUSY never asserts) when a battery board loads the lines. Cap the SPI clock:
in the installed `inky/inky_el133uf1.py`, change
`self._spi_bus.max_speed_hz = 10000000` to `= 2000000`. No reboot needed (a
fresh `SpiDev` is opened per render); render time barely changes because the
panel's refresh physics dominate. NB: this edit is in the pip-installed `inky`
package (pin 2.5.0) and must be re-applied if `inky` is reinstalled/upgraded.
