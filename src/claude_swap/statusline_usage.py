"""Usage readings for setup-token slots, from the Claude Code statusline cache.

A setup-token (``claude setup-token``) carries only the ``user:inference``
scope, so the usage endpoint refuses it. Claude Code still hands the
account's rate limits to its statusline command on every render; a
statusline script records them per token, and this module reads that record
back. cswap only ever reads the cache: the directory, its permissions and
its pruning belong to the writer.

The contract, shared with the writer:

- directory: ``$CLAUDE_USAGE_CACHE_DIR`` when set and non-empty, else
  ``${XDG_STATE_HOME:-$HOME/.local/state}/claude-usage``. That is the
  writer's rule verbatim, on every OS — not ``paths``' Linux-only,
  absolute-only XDG rule, which would read somewhere the writer never
  writes;
- file: ``<first 16 hex digits of sha256(accessToken)>.json``;
- body: ``{"schema": 1, "source": "statusline", "captured_at": <epoch s>,
  "five_hour": {"used_percentage": <num>, "resets_at": <epoch s>} | null,
  "seven_day": {...} | null}``, ``resets_at`` optional.

A reading is only as fresh as the last Claude Code reply on that token, so
its staleness is bounded by the windows themselves instead of an age: a
window whose reset has passed reads as 0% (usage starts over at a reset),
and the reading stays decision-grade at any age.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from claude_swap import oauth
from claude_swap.json_output import USAGE_STATUSLINE_PENDING
from claude_swap.usage_store import STATUSLINE_SOURCE, UsageEntry

_SCHEMA = 1
_BODY_SOURCE = "statusline"
# A reset further past the capture than one full window (plus slack for clock
# drift between Claude Code's servers and this machine) cannot describe that
# window. Trusting it would keep a stale percentage alive long after the real
# reset, so such a window is dropped as malformed.
_WINDOWS = (("five_hour", 5 * 3600.0), ("seven_day", 7 * 86400.0))
_RESET_SLACK_S = 3600.0


def cache_dir() -> Path:
    """The statusline cache directory, resolved now (never at import)."""
    override = os.environ.get("CLAUDE_USAGE_CACHE_DIR")
    if override:
        return Path(override)
    state = os.environ.get("XDG_STATE_HOME")
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "claude-usage"


def token_fingerprint(access_token: str) -> str:
    """The cache file stem for a token: 16 hex digits of its SHA-256."""
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]


def cache_path(access_token: str) -> Path:
    return cache_dir() / f"{token_fingerprint(access_token)}.json"


class StatuslineUsage(dict):
    """A usage dict converted from a statusline reading.

    The normal internal shape (``oauth.build_usage_result``) and equal to its
    plain-dict copy; the type only lets value-level consumers tell a reading
    that structurally cannot carry per-model windows from an API reading
    that happens to have none (the model-name typo guards).
    """


def is_statusline_value(value: object) -> bool:
    """Whether a decision value came from the statusline cache: a reading,
    or the pending sentinel standing in for one."""
    return isinstance(value, StatuslineUsage) or value == USAGE_STATUSLINE_PENDING


@dataclass(frozen=True)
class StatuslineReading:
    """What the cache holds for one token.

    ``usage`` is the converted dict, or ``None`` when there is no usable
    window (no file, an unreadable or invalid one, or every window null or
    malformed). ``captured_at`` is set whenever the file itself was valid.
    """

    usage: StatuslineUsage | None
    captured_at: float | None


_NO_READING = StatuslineReading(None, None)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _api_window(
    raw: object, captured_at: float, length_s: float, now: float
) -> dict | None:
    """One window in the usage API's raw shape, or None if absent/malformed."""
    if not isinstance(raw, dict):
        return None
    pct = raw.get("used_percentage")
    if not _finite(pct):
        return None
    window: dict = {"utilization": min(max(float(pct), 0.0), 100.0)}
    resets_at = raw.get("resets_at")
    if resets_at is None:
        return window
    if not _finite(resets_at) or resets_at > captured_at + length_s + _RESET_SLACK_S:
        return None
    if resets_at <= now:
        # Rolled over since the capture: the window starts empty, and its
        # past reset must go too, or every reset-bounded consumer would read
        # the whole snapshot as obsolete.
        return {"utilization": 0.0}
    try:
        window["resets_at"] = datetime.fromtimestamp(
            resets_at, tz=timezone.utc
        ).isoformat()
    except (OverflowError, OSError, ValueError):
        return None
    return window


def parse_reading(body: object, now: float) -> StatuslineReading:
    """Validate a cache body and convert it, rolling windows reset by ``now``."""
    if not isinstance(body, dict):
        return _NO_READING
    schema = body.get("schema")
    if isinstance(schema, bool) or schema != _SCHEMA:
        return _NO_READING
    if body.get("source") != _BODY_SOURCE:
        return _NO_READING
    captured_at = body.get("captured_at")
    # Beyond the clock-drift slack in the future, or before the epoch, the
    # capture time is corrupt: it would read as "just now" at any real age,
    # or fail every timestamp projection downstream.
    if not _finite(captured_at) or not 0 < captured_at <= now + _RESET_SLACK_S:
        return _NO_READING
    captured_at = float(captured_at)
    raw: dict = {}
    for key, length_s in _WINDOWS:
        window = _api_window(body.get(key), captured_at, length_s, now)
        if window is not None:
            raw[key] = window
    usage = oauth.build_usage_result(raw) if raw else None
    return StatuslineReading(
        StatuslineUsage(usage) if usage else None, captured_at
    )


def read_reading(credentials: str, now: float) -> StatuslineReading:
    """The cached reading for a credential's access token. Never raises."""
    token = oauth.extract_access_token(credentials) if credentials else None
    if not isinstance(token, str) or not token:
        return _NO_READING
    try:
        body = json.loads(cache_path(token).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return _NO_READING
    return parse_reading(body, now)


def usage_entry(credentials: str, now: float) -> UsageEntry:
    """A setup-token slot's usage entry, built fresh from the cache.

    Never derived from the usage store's row: fields left there from when
    these slots were still fetched (an ``http-403`` backoff, a strike from a
    401, a poll plan, an adopted reading) must not leak into display or the
    engine. A reading is decision-grade at any age (``trust_extended``): its
    windows already rolled to 0% at their resets.
    """
    reading = read_reading(credentials, now)
    if reading.usage is None:
        return UsageEntry(
            sentinel=USAGE_STATUSLINE_PENDING,
            fetched_at=reading.captured_at,
            source=STATUSLINE_SOURCE,
        )
    return UsageEntry(
        last_good=reading.usage,
        fetched_at=reading.captured_at,
        age_s=max(0.0, now - reading.captured_at),
        trust_extended=True,
        source=STATUSLINE_SOURCE,
    )
