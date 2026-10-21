"""Setup-token usage read from the Claude Code statusline cache.

The reader (``claude_swap.statusline_usage``), and the collector contract it
feeds: a setup-token slot is never fetched, refreshed or struck, whatever is
or is not in the cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth, statusline_usage
from claude_swap.json_output import USAGE_STATUSLINE_PENDING
from claude_swap.models import Platform
from claude_swap.paths import get_credentials_path, get_global_config_path
from claude_swap.statusline_usage import StatuslineUsage, parse_reading, read_reading
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import STATUSLINE_SOURCE

TOKEN = "sk-ant-oat01-" + "s3t4u5p6t7" * 4
OTHER_TOKEN = "sk-ant-oat01-" + "o1t2h3e4r5" * 4
SETUP_EMAIL = "setup-token-2@token.local"
OAUTH_EMAIL = "alice@example.com"
HOUR = 3600.0
DAY = 86400.0


def _setup_creds(token: str = TOKEN) -> str:
    return json.dumps(
        {"claudeAiOauth": {"accessToken": token, "scopes": ["user:inference"]}}
    )


def _oauth_creds(name: str = "alice") -> str:
    return json.dumps({
        "claudeAiOauth": {
            "accessToken": f"at-{name}",
            "refreshToken": f"rt-{name}",
            "expiresAt": int((time.time() + DAY) * 1000),
            "scopes": ["user:inference", "user:profile"],
        }
    })


def _body(
    captured_at: float,
    five: tuple[float, float | None] | None = (12.0, None),
    seven: tuple[float, float | None] | None = (40.0, None),
    **extra,
) -> dict:
    def window(spec):
        if spec is None:
            return None
        pct, resets_at = spec
        out: dict = {"used_percentage": pct}
        if resets_at is not None:
            out["resets_at"] = resets_at
        return out

    body = {
        "schema": 1,
        "source": "statusline",
        "captured_at": captured_at,
        "five_hour": window(five),
        "seven_day": window(seven),
    }
    body.update(extra)
    return body


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@pytest.fixture
def cache(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "claude-usage"
    monkeypatch.setenv("CLAUDE_USAGE_CACHE_DIR", str(directory))
    return directory


def _write_reading(cache: Path, body: dict, token: str = TOKEN) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{hashlib.sha256(token.encode()).hexdigest()[:16]}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------


class TestReader:
    NOW = 1_800_000_000.0

    def test_converts_to_the_internal_usage_shape(self):
        five_reset = self.NOW + 2 * HOUR
        seven_reset = self.NOW + 3 * DAY
        reading = parse_reading(
            _body(self.NOW - 60, five=(12, five_reset), seven=(40.5, seven_reset)),
            self.NOW,
        )
        assert reading.captured_at == self.NOW - 60
        usage = reading.usage
        assert isinstance(usage, StatuslineUsage)
        assert usage["five_hour"]["pct"] == 12.0
        assert usage["five_hour"]["resets_at"] == _iso(five_reset)
        assert usage["five_hour"]["resets_at"].endswith("+00:00")
        assert {"countdown", "clock"} <= usage["five_hour"].keys()
        assert usage["seven_day"]["pct"] == 40.5
        assert usage["seven_day"]["resets_at"] == _iso(seven_reset)
        assert "scoped" not in usage and "spend" not in usage
        # Same shape the API path builds, so every consumer reads it as-is.
        assert oauth.account_headroom(usage) == pytest.approx(59.5)

    def test_reads_the_tokens_file(self, cache):
        now = time.time()
        _write_reading(cache, _body(now - 10, five=(33, now + HOUR)))
        _write_reading(cache, _body(now - 10, five=(99, now + HOUR)), OTHER_TOKEN)
        reading = read_reading(_setup_creds(), now)
        assert reading.usage["five_hour"]["pct"] == 33.0

    @pytest.mark.parametrize("content", ["", "{not json", "\xff\xfe", "[]"])
    def test_corrupt_file(self, cache, content):
        cache.mkdir()
        path = cache / f"{statusline_usage.token_fingerprint(TOKEN)}.json"
        path.write_bytes(content.encode("latin-1"))
        reading = read_reading(_setup_creds(), time.time())
        assert reading.usage is None and reading.captured_at is None


# ---------------------------------------------------------------------------


def _linux_switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _seed_oauth(s: ClaudeAccountSwitcher, num: int, email: str, creds: str) -> None:
    s._write_account_credentials(str(num), email, creds)
    s._write_account_config(str(num), email, json.dumps({
        "oauthAccount": {"emailAddress": email, "accountUuid": f"uuid-{num}"},
    }))
    data = s._get_sequence_data()
    data["accounts"][str(num)] = {
        "email": email,
        "uuid": f"uuid-{num}",
        "organizationUuid": "",
        "organizationName": "",
        "added": "2024-01-01T00:00:00Z",
    }
    data["sequence"] = sorted({*data["sequence"], num})
    s._write_json(s.sequence_file, data)


def _make_live(email: str, creds: str, account_uuid: str = "") -> None:
    path = get_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds, encoding="utf-8")
    get_global_config_path().write_text(json.dumps({
        "oauthAccount": {"emailAddress": email, "accountUuid": account_uuid},
    }), encoding="utf-8")


@pytest.fixture
def fleet(temp_home: Path, cache: Path):
    """Slot 1: a refreshable OAuth login (active). Slot 2: a setup-token."""
    s = _linux_switcher()
    _seed_oauth(s, 1, OAUTH_EMAIL, _oauth_creds())
    s.add_account_from_token(TOKEN, slot=2)
    _make_live(OAUTH_EMAIL, _oauth_creds(), "uuid-1")
    return s


class _NetworkSpy:
    """Every way a usage pass could reach the network or a refresh."""

    def __init__(self, s: ClaudeAccountSwitcher):
        self.s = s
        self.fetched: list[str] = []

    def __enter__(self):
        def fake_fetch(num, email, creds, **kwargs):
            self.fetched.append(email)
            return oauth.UsageOutcome({"five_hour": {"pct": 5.0}})

        self._patches = [
            patch(
                "claude_swap.oauth.try_fetch_usage_for_account",
                side_effect=fake_fetch,
            ),
            patch(
                "claude_swap.oauth.request_usage_data",
                side_effect=AssertionError("usage endpoint requested"),
            ),
            patch(
                "claude_swap.oauth.try_refresh_oauth_credentials",
                side_effect=AssertionError("refresh attempted"),
            ),
            patch.object(
                self.s, "consume_backup_grant",
                side_effect=AssertionError("backup grant consumed"),
            ),
            patch.object(
                self.s._usage_store, "clear_dead_token",
                wraps=self.s._usage_store.clear_dead_token,
            ),
        ]
        mocks = [p.start() for p in self._patches]
        self.clear_dead_token = mocks[-1]
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False



class TestCollectorNeverFetches:
    def test_inactive_slot_with_a_reading(self, fleet, cache):
        now = time.time()
        _write_reading(cache, _body(now - 2 * HOUR, five=(20, now + HOUR), seven=(30, now + DAY)))
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
        assert spy.fetched == [OAUTH_EMAIL]
        entry = entries["2"]
        assert entry.source == STATUSLINE_SOURCE
        assert entry.decision_value()["seven_day"]["pct"] == 30.0
        assert entries["1"].source is None

    @pytest.mark.parametrize("with_reading", [True, False])
    def test_active_slot(self, fleet, cache, with_reading):
        now = time.time()
        if with_reading:
            _write_reading(cache, _body(now - 60, five=(20, now + HOUR)))
        _make_live(SETUP_EMAIL, _setup_creds())
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
            single = fleet._active_account_usage("2", SETUP_EMAIL, "")
        assert spy.fetched == [OAUTH_EMAIL]
        for entry in (entries["2"], single):
            assert entry.source == STATUSLINE_SOURCE
            if with_reading:
                assert entry.decision_value()["five_hour"]["pct"] == 20.0
            else:
                assert entry.decision_value() == USAGE_STATUSLINE_PENDING


class TestListAndSwitch:
    def test_list_json(self, fleet, cache):
        now = time.time()
        captured = now - 3 * HOUR
        _write_reading(cache, _body(captured, five=(20, now + HOUR), seven=(30, now + DAY)))
        with _NetworkSpy(fleet):
            payload = fleet.list_accounts(json_output=True)
        oauth_row, setup_row = payload["accounts"]
        assert "usageSource" not in oauth_row
        assert setup_row["usageSource"] == "statusline"
        assert setup_row["usageStatus"] == "ok"
        assert setup_row["usage"]["fiveHour"]["pct"] == 20.0
        assert setup_row["usage"]["sevenDay"]["pct"] == 30.0
        assert setup_row["usageAgeSeconds"] == pytest.approx(3 * HOUR, abs=5)
        assert setup_row["usageFetchedAt"] == (
            datetime.fromtimestamp(captured, tz=timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        assert "lastGoodUsage" not in setup_row

    def test_best_chooses_an_hours_old_reading(self, fleet, cache):
        now = time.time()
        _write_reading(cache, _body(now - 5 * HOUR, five=(10, now + HOUR), seven=(15, now + DAY)))
        with _NetworkSpy(fleet), patch(
            "claude_swap.oauth.try_fetch_usage_for_account",
            return_value=oauth.UsageOutcome({
                "five_hour": {"pct": 80.0}, "seven_day": {"pct": 50.0},
            }),
        ):
            target, note = fleet._select_best_switchable("1")
        assert (target, note) == ("2", "")
