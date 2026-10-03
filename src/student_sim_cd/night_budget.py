"""Persist conservative wall-time reservations across bounded overnight retries.

The caller must stop and wait for its child processes before finish(). A lease
that is killed or never finished retains its full reservation; process IDs and
connection state are never used to infer that a run has stopped.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from time import monotonic, time
import uuid


SCHEMA_VERSION = "student-sim-cd.night-budget.v1"
LEDGER_FIELDS = {"schema_version", "stop_by", "max_seconds", "created_at", "updated_at", "entries"}
ENTRY_FIELDS = {"entry_id", "pid", "requested_seconds", "reserved_seconds", "charged_seconds",
                "status", "acquired_at", "finished_at", "elapsed_seconds"}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a nonnegative finite number") from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return converted


def _aware(value):
    if not isinstance(value, str):
        raise ValueError("stop_by must be an aware ISO timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("stop_by must be an aware ISO timestamp string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stop_by must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _canonical(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("budget paths must be canonical and contain no symlinks")
    if not path.parent.is_dir():
        raise ValueError("budget parent directory must already exist")
    return path


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate budget JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite budget JSON number: " + value)


class NightBudget:
    """One nonblocking exclusive lease, held until finish or process exit.

    Atomic ledger replacement requires a separate persistent lock file. Locking
    the ledger inode itself would allow another process to bypass the lock after
    replacement. Neither the ledger nor the lock is deleted during recovery.
    """

    def __init__(self, path, *, stop_by: str, max_seconds=14400):
        self.path = _canonical(path)
        self.lock_path = _canonical(self.path.with_name(self.path.name + ".lock"))
        self.stop_by = _aware(stop_by).isoformat()
        self.deadline = _aware(stop_by).timestamp()
        self.max_seconds = _positive(max_seconds, "max_seconds")
        self._lock_fd = None
        self._entry_id = None
        self._reservation = None
        self._started = None

    def _release(self):
        if self._lock_fd is not None:
            descriptor, self._lock_fd = self._lock_fd, None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read(self, now):
        _canonical(self.path)
        try:
            descriptor = os.open(str(self.path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
            return {"schema_version": SCHEMA_VERSION, "stop_by": self.stop_by,
                    "max_seconds": self.max_seconds, "created_at": stamp,
                    "updated_at": stamp, "entries": []}
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("budget ledger must be a regular file owned by the current account")
            try:
                ledger = json.load(handle, object_pairs_hook=_unique, parse_constant=_reject_constant)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("damaged budget ledger; existing bytes were preserved") from exc
        if (not isinstance(ledger, dict) or set(ledger) != LEDGER_FIELDS
                or ledger["schema_version"] != SCHEMA_VERSION
                or ledger["stop_by"] != self.stop_by
                or type(ledger["max_seconds"]) not in (int, float)
                or ledger["max_seconds"] != self.max_seconds
                or not isinstance(ledger["entries"], list)):
            raise ValueError("budget ledger metadata/schema mismatch; existing bytes were preserved")
        _aware(ledger["created_at"])
        _aware(ledger["updated_at"])
        identifiers = set()
        for entry in ledger["entries"]:
            if (not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS
                    or not isinstance(entry["entry_id"], str) or not entry["entry_id"]
                    or entry["entry_id"] in identifiers or type(entry["pid"]) is not int or entry["pid"] < 1):
                raise ValueError("damaged budget entry; existing bytes were preserved")
            identifiers.add(entry["entry_id"])
            requested = _positive(entry["requested_seconds"], "entry requested_seconds")
            reserved = _positive(entry["reserved_seconds"], "entry reserved_seconds")
            charged = _nonnegative(entry["charged_seconds"], "entry charged_seconds")
            if not charged <= reserved <= requested:
                raise ValueError("invalid budget reservation/charge; existing bytes were preserved")
            _aware(entry["acquired_at"])
            if entry["status"] == "reserved":
                if charged != reserved or entry["finished_at"] is not None or entry["elapsed_seconds"] is not None:
                    raise ValueError("unfinished budget entry must retain its full reservation")
            elif entry["status"] == "finished":
                elapsed = _nonnegative(entry["elapsed_seconds"], "entry elapsed_seconds")
                if charged != min(reserved, math.ceil(elapsed)):
                    raise ValueError("invalid settled budget entry; existing bytes were preserved")
                _aware(entry["finished_at"])
            else:
                raise ValueError("unsupported budget entry status; existing bytes were preserved")
        if math.fsum(entry["charged_seconds"] for entry in ledger["entries"]) > self.max_seconds:
            raise ValueError("budget ledger exceeds its maximum; existing bytes were preserved")
        return ledger

    def _write(self, ledger):
        _canonical(self.path)
        descriptor, temporary = tempfile.mkstemp(prefix=".night-budget-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            # Only the exclusively created temporary file is eligible for cleanup.
            if os.path.exists(temporary):
                os.unlink(temporary)

    def acquire(self, requested_seconds) -> float:
        """Reserve min(requested, remaining total budget, deadline remaining)."""
        requested = _positive(requested_seconds, "requested_seconds")
        if self._lock_fd is not None:
            raise ValueError("this budget object already has an active lease")
        _canonical(self.path)
        _canonical(self.lock_path)
        descriptor = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        self._lock_fd = descriptor
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("budget lock must be a regular file owned by the current account")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("night budget has an active lease; concurrent runs are forbidden") from exc
            now = time()
            ledger = self._read(now)
            remaining = self.max_seconds - math.fsum(entry["charged_seconds"] for entry in ledger["entries"])
            allowed = min(requested, remaining, self.deadline - now)
            if allowed <= 0:
                raise ValueError("night budget exhausted or absolute stop_by deadline reached")
            started = monotonic()
            if not math.isfinite(started):
                raise ValueError("invalid monotonic clock value")
            entry_id = uuid.uuid4().hex
            stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
            ledger["entries"].append({
                "entry_id": entry_id, "pid": os.getpid(), "requested_seconds": requested,
                "reserved_seconds": allowed, "charged_seconds": allowed, "status": "reserved",
                "acquired_at": stamp, "finished_at": None, "elapsed_seconds": None,
            })
            ledger["updated_at"] = stamp
            self._write(ledger)  # Persist the full charge before permitting the run.
            self._entry_id, self._reservation, self._started = entry_id, allowed, started
            return float(allowed)
        except BaseException:
            self._release()
            raise

    def finish(self) -> None:
        """After all child processes stop, settle elapsed time and release lease.

        Calling finish again is harmless. A damaged ledger or clock leaves the
        full precharge intact and releases the file lock while reporting failure.
        """
        if self._lock_fd is None:
            return
        try:
            now = time()
            ledger = self._read(now)
            entries = [entry for entry in ledger["entries"] if entry["entry_id"] == self._entry_id]
            if (len(entries) != 1 or entries[0]["status"] != "reserved"
                    or entries[0]["reserved_seconds"] != self._reservation):
                raise ValueError("active budget reservation changed; existing ledger was preserved")
            elapsed = monotonic() - self._started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid elapsed monotonic time; full reservation was preserved")
            stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
            entries[0].update(status="finished", finished_at=stamp, elapsed_seconds=elapsed,
                              charged_seconds=min(self._reservation, math.ceil(elapsed)))
            ledger["updated_at"] = stamp
            self._write(ledger)
        finally:
            self._release()
