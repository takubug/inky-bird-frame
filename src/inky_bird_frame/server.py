"""Read-only HTTP service for approved catalog metadata and assets."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Callable
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .catalog import (
    approve_candidate,
    find_taxon_directory,
    read_json,
    rebuild_catalog_index,
    reject_candidate,
    utc_now,
)
from .config import ControllerConfig
from .errors import CatalogError
from .http import write_json_atomic


def _species_count(path: Path) -> int:
    try:
        value = read_json(path)
    except CatalogError:
        return 0
    if isinstance(value, dict) and isinstance(value.get("species"), list):
        return len(value["species"])
    return 0


class CatalogRequestHandler(BaseHTTPRequestHandler):
    catalog_dir: Path
    active_catalog_path: Path
    state_dir: Path
    # Rebuilds the active (on-frame) catalog after an approval so a phone approve
    # reaches the frame on the next fetch, not just the next generation cycle.
    refresh_active_catalog: Callable[[], None] | None = None
    # Pushes a confirmation back to the notifier after a phone Approve/Reject
    # (taxon_id, outcome, message), so a silent action button gets visible feedback.
    on_decision: Callable[[int, str, str], None] | None = None

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400, immutable")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        split = urlsplit(self.path)
        request_path = split.path
        if request_path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "approved_species": _species_count(self.catalog_dir / "index.json"),
                    "active_species": _species_count(self.active_catalog_path),
                    "schema_version": 1,
                },
            )
            return
        if request_path == "/v1/catalog":
            try:
                payload = read_json(self.active_catalog_path)
            except CatalogError:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "error": "active catalog unavailable", "schema_version": 1},
                )
                return
            self._send_json(HTTPStatus.OK, payload)
            # Only display nodes send the marker; a curl or monitor fetch must
            # not refresh the liveness signal.
            if parse_qs(split.query).get("reports_success") == ["1"]:
                with suppress(OSError):
                    write_json_atomic(
                        self.state_dir / "display-last-fetch.json",
                        {"schema_version": 1, "fetched_at": utc_now()},
                    )
            return
        if request_path == "/v1/display-success":
            with suppress(OSError):
                write_json_atomic(
                    self.state_dir / "display-last-success.json",
                    {"schema_version": 1, "succeeded_at": utc_now()},
                )
            self._send_json(HTTPStatus.OK, {"ok": True, "schema_version": 1})
            return
        if request_path == "/v1/pending":
            self._send_json(HTTPStatus.OK, {"pending": self._pending_list(), "schema_version": 1})
            return
        pending_taxon = self._pending_taxon(request_path, "/portrait.png")
        if pending_taxon is not None:
            source = find_taxon_directory(self.state_dir / "pending", pending_taxon)
            portrait = source / "portrait.png" if source is not None else None
            if portrait is None or not portrait.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            self._send_file(portrait)
            return
        prefix = "/v1/assets/"
        if request_path.startswith(prefix):
            relative = Path(unquote(request_path.removeprefix(prefix)))
            root = self.catalog_dir.resolve()
            candidate = (root / relative).resolve()
            if (
                any(part.startswith(".") for part in relative.parts)
                or not candidate.is_relative_to(root)
                or not candidate.is_file()
            ):
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            self._send_file(candidate)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        split = urlsplit(self.path)
        request_path = split.path
        # Drain any request body so the socket can be reused cleanly.
        with suppress(ValueError):
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(length)

        approve_taxon = self._pending_taxon(request_path, "/approve")
        if approve_taxon is not None:
            self._approve(approve_taxon)
            return
        reject_taxon = self._pending_taxon(request_path, "/reject")
        if reject_taxon is not None:
            reason = parse_qs(split.query).get("reason", ["Rejected from the notification"])[0]
            self._reject(reject_taxon, reason)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def _approve(self, taxon_id: int) -> None:
        name = self._pending_common_name(taxon_id) or f"Plate {taxon_id}"
        try:
            entry = approve_candidate(self.state_dir, self.catalog_dir, taxon_id)
        except CatalogError as exc:
            self._decision(taxon_id, "error", f"Could not approve {name}: {exc}")
            self._send_json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
            return
        if self.refresh_active_catalog is not None:
            with suppress(Exception):
                self.refresh_active_catalog()
        self._decision(
            taxon_id, "approved", f"{entry.common_name} approved; it will appear on the frame."
        )
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "status": "approved",
                "taxon_id": entry.taxon_id,
                "common_name": entry.common_name,
                "schema_version": 1,
            },
        )

    def _reject(self, taxon_id: int, reason: str) -> None:
        name = self._pending_common_name(taxon_id) or f"Plate {taxon_id}"
        try:
            reject_candidate(self.state_dir, taxon_id, reason)
        except CatalogError as exc:
            self._decision(taxon_id, "error", f"Could not reject {name}: {exc}")
            self._send_json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
            return
        self._decision(taxon_id, "rejected", f"{name} rejected; it will not be published.")
        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "status": "rejected", "taxon_id": taxon_id, "schema_version": 1},
        )

    def _decision(self, taxon_id: int, outcome: str, message: str) -> None:
        if self.on_decision is not None:
            with suppress(Exception):
                self.on_decision(taxon_id, outcome, message)

    def _pending_common_name(self, taxon_id: int) -> str | None:
        source = find_taxon_directory(self.state_dir / "pending", taxon_id)
        if source is None:
            return None
        try:
            manifest = read_json(source / "manifest.json")
        except CatalogError:
            return None
        name = manifest.get("common_name") if isinstance(manifest, dict) else None
        return name if isinstance(name, str) else None

    def _pending_taxon(self, request_path: str, suffix: str) -> int | None:
        """Parse the taxon id from /v1/pending/<taxon><suffix>, or None if it does not match."""
        prefix = "/v1/pending/"
        if not request_path.startswith(prefix) or not request_path.endswith(suffix):
            return None
        middle = request_path[len(prefix) : -len(suffix)]
        if not middle.isdigit():
            return None
        return int(middle)

    def _pending_list(self) -> list[dict[str, object]]:
        pending_root = self.state_dir / "pending"
        entries: list[dict[str, object]] = []
        if not pending_root.is_dir():
            return entries
        for manifest_path in sorted(pending_root.glob("*/manifest.json")):
            try:
                manifest = read_json(manifest_path)
            except CatalogError:
                continue
            if not isinstance(manifest, dict):
                continue
            taxon_id = manifest.get("taxon_id")
            common_name = manifest.get("common_name")
            if isinstance(taxon_id, int) and isinstance(common_name, str):
                entries.append({"taxon_id": taxon_id, "common_name": common_name})
        return entries

    def log_message(self, message_format: str, *args: object) -> None:
        message = message_format % args
        print(
            json.dumps(
                {"event": "http_request", "client": self.client_address[0], "message": message}
            )
        )


def rebuild_index_logging_failures(catalog_dir: Path) -> None:
    try:
        rebuild_catalog_index(catalog_dir)
    except (CatalogError, OSError) as exc:
        print(json.dumps({"event": "catalog_index_rebuild_failed", "error": str(exc)}))


def serve_catalog(
    config: ControllerConfig,
    *,
    refresh_active_catalog: Callable[[], None] | None = None,
    on_decision: Callable[[int, str, str], None] | None = None,
) -> None:
    config.catalog_dir.mkdir(parents=True, exist_ok=True)
    rebuild_index_logging_failures(config.catalog_dir)
    handler = type(
        "ConfiguredCatalogRequestHandler",
        (CatalogRequestHandler,),
        {
            "catalog_dir": config.catalog_dir,
            "active_catalog_path": config.state_dir / "active-catalog.json",
            "state_dir": config.state_dir,
            "refresh_active_catalog": staticmethod(refresh_active_catalog)
            if refresh_active_catalog is not None
            else None,
            "on_decision": staticmethod(on_decision) if on_decision is not None else None,
        },
    )
    server = ThreadingHTTPServer((config.bind_host, config.port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
