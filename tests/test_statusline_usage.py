"""Setup-token usage read from the Claude Code statusline cache.

The reader (``claude_swap.statusline_usage``), and the collector contract it
feeds: a setup-token slot is never fetched, refreshed or struck, whatever is
or is not in the cache and whatever its old usage-store row says.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth, statusline_usage
from claude_swap.json_output import USAGE_STATUSLINE_PENDING, usage_fields
from claude_swap.models import Platform
from claude_swap.paths import get_credentials_path, get_global_config_path
from claude_swap.statusline_usage import (
    StatuslineUsage,
    is_statusline_value,
    parse_reading,
    read_reading,
    usage_entry,
)
from claude_swap.switcher import SENTINEL_NOTES, ClaudeAccountSwitcher
from claude_swap.usage_store import (
    STALE_OK_S,
    STATUSLINE_SOURCE,
    TRUST_MAX_AGE_S,
    FetchRecord,
)

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


class TestCacheDir:
    def test_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_USAGE_CACHE_DIR", str(tmp_path / "x"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        assert statusline_usage.cache_dir() == tmp_path / "x"

    def test_empty_override_falls_through_to_xdg_state_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_USAGE_CACHE_DIR", "")
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        assert statusline_usage.cache_dir() == tmp_path / "state" / "claude-usage"

    def test_default_is_under_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("XDG_STATE_HOME", "")
        assert statusline_usage.cache_dir() == (
            tmp_path / ".local" / "state" / "claude-usage"
        )

    @pytest.mark.parametrize("platform", [Platform.LINUX, Platform.MACOS])
    def test_xdg_state_home_is_the_writers_rule_on_every_os(
        self, monkeypatch, platform
    ):
        """Not ``paths``' rule (Linux only, absolute only): the reader must
        land wherever the statusline writer wrote, which takes the variable
        verbatim on every OS."""
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: platform))
        monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
        assert statusline_usage.cache_dir() == Path("relative-state") / "claude-usage"

    def test_resolved_at_read_time(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_USAGE_CACHE_DIR", str(tmp_path / "first"))
        first = statusline_usage.cache_path(TOKEN)
        monkeypatch.setenv("CLAUDE_USAGE_CACHE_DIR", str(tmp_path / "second"))
        assert statusline_usage.cache_path(TOKEN).parent == tmp_path / "second"
        assert first.parent == tmp_path / "first"

    def test_file_name_is_the_tokens_sha256_prefix(self, cache):
        expected = hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()[:16]
        assert statusline_usage.token_fingerprint(TOKEN) == expected
        assert statusline_usage.cache_path(TOKEN) == cache / f"{expected}.json"
        # Unlike oauth.access_token_fingerprint, no prefix and truncated.
        assert len(expected) == 16


# ---------------------------------------------------------------------------
# Parsing and conversion
# ---------------------------------------------------------------------------


class TestParseReading:
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

    def test_null_window_is_omitted(self):
        reading = parse_reading(_body(self.NOW, five=(12, None), seven=None), self.NOW)
        assert set(reading.usage) == {"five_hour"}

    def test_no_window_at_all_is_no_usage(self):
        reading = parse_reading(_body(self.NOW, five=None, seven=None), self.NOW)
        assert reading.usage is None
        assert reading.captured_at == self.NOW

    @pytest.mark.parametrize(
        ("raw", "pct"), [(-5, 0.0), (150, 100.0), (99.5, 99.5), (0, 0.0)]
    )
    def test_percentage_is_clamped(self, raw, pct):
        reading = parse_reading(_body(self.NOW, five=(raw, None), seven=None), self.NOW)
        assert reading.usage["five_hour"]["pct"] == pct

    @pytest.mark.parametrize("bad", [math.nan, math.inf, "12", None, True])
    def test_unusable_percentage_drops_the_window(self, bad):
        body = _body(self.NOW, seven=(40, None))
        body["five_hour"] = {"used_percentage": bad}
        reading = parse_reading(body, self.NOW)
        assert set(reading.usage) == {"seven_day"}

    @pytest.mark.parametrize(
        "body",
        [
            None,
            [],
            "x",
            _body(1.0, schema=2),
            _body(1.0, schema=True),
            _body(1.0, schema="1"),
            _body(1.0, source="api"),
            {"schema": 1, "source": "statusline", "five_hour": {"used_percentage": 1}},
            _body(math.nan),
            _body(math.inf),
            _body("1700000000"),  # type: ignore[arg-type]
            _body(0.0),
            _body(-1e12),
            _body(1e20),
            _body(NOW + 2 * HOUR),
        ],
    )
    def test_invalid_body_is_no_reading(self, body):
        reading = parse_reading(body, self.NOW)
        assert reading.usage is None
        assert reading.captured_at is None

    @pytest.mark.parametrize(
        ("key", "length"), [("five_hour", 5 * HOUR), ("seven_day", 7 * DAY)]
    )
    def test_reset_beyond_one_window_is_malformed(self, key, length):
        captured = self.NOW - 60
        spec_ok = (50, captured + length + HOUR)
        spec_bad = (50, captured + length + HOUR + 1)
        other = "seven_day" if key == "five_hour" else "five_hour"

        def body(spec):
            b = _body(captured, five=None, seven=None)
            b[key] = {"used_percentage": spec[0], "resets_at": spec[1]}
            b[other] = {"used_percentage": 1}
            return b

        assert key in parse_reading(body(spec_ok), self.NOW).usage
        assert key not in parse_reading(body(spec_bad), self.NOW).usage

    @pytest.mark.parametrize("bad", ["soon", math.nan, True, [1]])
    def test_unparseable_reset_drops_the_window(self, bad):
        body = _body(self.NOW, seven=(40, None))
        body["five_hour"] = {"used_percentage": 12, "resets_at": bad}
        assert set(parse_reading(body, self.NOW).usage) == {"seven_day"}

    def test_rolled_window_reads_zero_without_a_reset(self):
        """A window whose reset passed since the capture starts over: 0%,
        with the obsolete reset (and its countdown/clock) gone."""
        reading = parse_reading(
            _body(
                self.NOW - 6 * HOUR,
                five=(97, self.NOW - HOUR),
                seven=(40, self.NOW + 2 * DAY),
            ),
            self.NOW,
        )
        assert reading.usage["five_hour"] == {"pct": 0.0}
        assert reading.usage["seven_day"]["pct"] == 40.0
        assert reading.usage["seven_day"]["resets_at"] == _iso(self.NOW + 2 * DAY)

    def test_reset_exactly_now_has_rolled(self):
        reading = parse_reading(
            _body(self.NOW - HOUR, five=(50, self.NOW), seven=None), self.NOW
        )
        assert reading.usage["five_hour"] == {"pct": 0.0}

    def test_every_window_rolled_is_zero_zero(self):
        reading = parse_reading(
            _body(
                self.NOW - 8 * DAY,
                five=(97, self.NOW - 8 * DAY + HOUR),
                seven=(88, self.NOW - DAY),
            ),
            self.NOW,
        )
        assert reading.usage == {"five_hour": {"pct": 0.0}, "seven_day": {"pct": 0.0}}
        assert oauth.account_headroom(reading.usage) == 100.0

    def test_window_without_reset_keeps_its_percentage(self):
        reading = parse_reading(_body(self.NOW - 30 * DAY, five=(70, None)), self.NOW)
        assert reading.usage["five_hour"] == {"pct": 70.0}


class TestReadReading:
    def test_reads_the_tokens_file(self, cache):
        now = time.time()
        _write_reading(cache, _body(now - 10, five=(33, now + HOUR)))
        _write_reading(cache, _body(now - 10, five=(99, now + HOUR)), OTHER_TOKEN)
        reading = read_reading(_setup_creds(), now)
        assert reading.usage["five_hour"]["pct"] == 33.0

    def test_missing_file(self, cache):
        assert read_reading(_setup_creds(), time.time()).usage is None
        # Read-only: the reader never creates the writer's directory.
        assert not cache.exists()

    @pytest.mark.parametrize("content", ["", "{not json", "\xff\xfe", "[]"])
    def test_corrupt_file(self, cache, content):
        cache.mkdir()
        path = cache / f"{statusline_usage.token_fingerprint(TOKEN)}.json"
        path.write_bytes(content.encode("latin-1"))
        reading = read_reading(_setup_creds(), time.time())
        assert reading.usage is None and reading.captured_at is None

    @pytest.mark.parametrize("creds", ["", "not json", '{"claudeAiOauth": {}}'])
    def test_no_access_token(self, cache, creds):
        assert read_reading(creds, time.time()).usage is None


class TestUsageEntry:
    def test_reading_is_decision_grade_at_any_age(self, cache):
        now = time.time()
        captured = now - 3 * TRUST_MAX_AGE_S
        _write_reading(cache, _body(captured, five=(20, now + HOUR), seven=(30, now + DAY)))
        entry = usage_entry(_setup_creds(), now)
        assert entry.source == STATUSLINE_SOURCE
        assert entry.sentinel is None
        assert entry.fetched_at == captured
        assert entry.age_s == pytest.approx(3 * TRUST_MAX_AGE_S)
        assert entry.age_s > STALE_OK_S
        assert entry.trust_extended is True
        assert entry.decision_value() == entry.last_good
        assert entry.last_good["five_hour"]["pct"] == 20.0
        # Nothing of the store's fetch state.
        assert entry.consecutive_failures == 0
        assert entry.last_error is None
        assert entry.backoff_until is None
        assert entry.next_poll_at is None
        assert entry.auth_dead_strikes == 0
        assert not entry.token_dead()

    def test_future_capture_has_zero_age(self, cache):
        now = time.time()
        _write_reading(cache, _body(now + 120))
        assert usage_entry(_setup_creds(), now).age_s == 0.0

    @pytest.mark.parametrize("captured_at", [1e20, -1e12])
    def test_out_of_range_capture_is_pending_and_projects(self, cache, captured_at):
        """A corrupt capture time must not reach the JSON timestamp
        projection, which cannot represent it."""
        from claude_swap.json_output import usage_freshness_fields

        _write_reading(cache, _body(captured_at, five=(50, None), seven=None))
        entry = usage_entry(_setup_creds(), time.time())
        assert entry.sentinel == USAGE_STATUSLINE_PENDING
        assert entry.fetched_at is None
        assert usage_fields(entry.decision_value(), entry.fetched_at)[0] == (
            "statusline_pending"
        )
        assert usage_freshness_fields(entry.fetched_at, entry.age_s) == {}

    def test_no_reading_is_pending(self, cache):
        entry = usage_entry(_setup_creds(), time.time())
        assert entry.sentinel == USAGE_STATUSLINE_PENDING
        assert entry.source == STATUSLINE_SOURCE
        assert entry.fetched_at is None
        assert entry.last_good is None
        assert entry.decision_value() == USAGE_STATUSLINE_PENDING

    def test_valid_file_without_windows_is_pending_at_its_capture(self, cache):
        now = time.time()
        _write_reading(cache, _body(now - 5, five=None, seven=None))
        entry = usage_entry(_setup_creds(), now)
        assert entry.sentinel == USAGE_STATUSLINE_PENDING
        assert entry.fetched_at == now - 5

    def test_is_statusline_value(self, cache):
        now = time.time()
        _write_reading(cache, _body(now))
        value = usage_entry(_setup_creds(), now).decision_value()
        assert is_statusline_value(value)
        assert value == dict(value)  # a plain dict to every other consumer
        assert is_statusline_value(USAGE_STATUSLINE_PENDING)
        assert not is_statusline_value(dict(value))
        assert not is_statusline_value(None)
        assert not is_statusline_value("api key")


def test_pending_maps_to_its_own_status():
    """Never the ``no_credentials`` fallthrough: the slot has a credential,
    and re-adding it would change nothing."""
    assert usage_fields(USAGE_STATUSLINE_PENDING) == ("statusline_pending", None)
    assert "statusline" in SENTINEL_NOTES[USAGE_STATUSLINE_PENDING]


# ---------------------------------------------------------------------------
# Through the switcher
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


def _identity(s: ClaudeAccountSwitcher, num: str) -> dict:
    data = s._get_sequence_data()
    acct = data["accounts"][num]
    return {num: (acct["email"], acct.get("organizationUuid", "") or "")}


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

    def test_inactive_slot_without_a_reading(self, fleet):
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
        assert spy.fetched == [OAUTH_EMAIL]
        assert entries["2"].decision_value() == USAGE_STATUSLINE_PENDING
        assert not entries["2"].token_dead()

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

    def test_explicit_fetch_set_does_not_reach_it(self, fleet):
        with _NetworkSpy(fleet) as spy:
            entries = fleet.usage_entries_by_account(fetch={"1", "2"})
        assert spy.fetched == [OAUTH_EMAIL]
        assert entries["2"].decision_value() == USAGE_STATUSLINE_PENDING

    def test_no_claim_is_stamped(self, fleet):
        with _NetworkSpy(fleet):
            fleet._collect_usage_entries(fleet._build_accounts_info())
        stored = fleet._usage_store.entries(_identity(fleet, "2"))["2"]
        assert stored.last_attempt_at is None
        assert not stored.claimed(time.time())

    def test_live_session_profile_is_not_read_only_fetched(self, fleet, monkeypatch):
        monkeypatch.setattr(fleet, "_live_session_pids", lambda *a: [4242])
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
        assert spy.fetched == [OAUTH_EMAIL]
        assert entries["2"].source == STATUSLINE_SOURCE

    @pytest.mark.parametrize("legacy", ["strike", "other-strike", "http-403"])
    def test_legacy_store_row_does_not_leak(self, fleet, cache, legacy):
        """Rows left from when these slots were fetched: a 403 backoff, or a
        no_refresh_token strike from an active-slot 401 (on these bytes, or
        on an earlier token the slot held). None may surface ("re-login
        needed", "usage unavailable (http-403)") or be cleared behind the
        user's back; the reading is the whole story."""
        identity = _identity(fleet, "2")
        record = {
            "strike": FetchRecord(
                error="no_refresh_token",
                struck_fp=oauth.credential_fingerprint(_setup_creds()),
            ),
            "other-strike": FetchRecord(
                error="no_refresh_token",
                struck_fp=oauth.credential_fingerprint(_setup_creds(OTHER_TOKEN)),
            ),
            "http-403": FetchRecord(error="http-403", retry_after_s=600.0),
        }[legacy]
        fleet._usage_store.record({"2": record}, identity)
        stored = fleet._usage_store.entries(identity)["2"]
        assert stored.last_error == record.error  # PREMISE: the row is there

        now = time.time()
        _write_reading(cache, _body(now - 60, five=(20, now + HOUR)))
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
            payload = fleet._build_list_payload(fleet._build_accounts_info(), entries)
        entry = entries["2"]
        assert entry.sentinel is None
        assert entry.decision_value()["five_hour"]["pct"] == 20.0
        assert entry.last_error is None and entry.backoff_until is None
        assert entry.auth_dead_strikes == 0
        spy.clear_dead_token.assert_not_called()
        row = next(r for r in payload["accounts"] if r["number"] == 2)
        assert row["usageStatus"] == "ok"
        assert "usageError" not in row
        # Pending ignores it too.
        _write_reading(cache, {"schema": 0})
        with _NetworkSpy(fleet):
            entry = fleet._collect_usage_entries(fleet._build_accounts_info())["2"]
        assert entry.sentinel == USAGE_STATUSLINE_PENDING
        assert entry.last_error is None and entry.last_good is None

    def test_slot_token_dead_ignores_a_legacy_strike(self, fleet):
        """`cswap import` auto-heal must not replace a working setup-token
        slot without --force over a strike the collectors never honor."""
        identity = _identity(fleet, "2")
        fleet._usage_store.record(
            {"2": FetchRecord(
                error="no_refresh_token",
                struck_fp=oauth.credential_fingerprint(_setup_creds()),
            )},
            identity,
        )
        assert fleet._usage_store.entries(identity)["2"].token_dead()  # PREMISE
        assert fleet._slot_token_dead("2", SETUP_EMAIL) is False

    @pytest.mark.parametrize("with_reading", [True, False])
    def test_oauth_slot_with_drifted_live_setup_token_keeps_oauth_path(
        self, fleet, cache, with_reading
    ):
        """The config names OAuth slot 1 but the live store holds slot 2's
        setup-token (a switch killed between its two writes, or an external
        tool). Slot 1 is not reclassified: it keeps the fetch path, whose
        refusal fails over and so repairs the drift, instead of holding on
        pending forever or serving slot 2's reading as its own."""
        now = time.time()
        if with_reading:
            _write_reading(cache, _body(now - 60, five=(20, now + HOUR)))
        _make_live(OAUTH_EMAIL, _setup_creds(), "uuid-1")
        info = fleet._build_accounts_info()
        assert info[0][4] and info[0][5] == _setup_creds()  # PREMISE: drifted
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(info)
        assert spy.fetched == [OAUTH_EMAIL]
        assert entries["1"].source is None
        assert entries["2"].source == STATUSLINE_SOURCE

    def test_slot_token_dead_honors_a_strike_on_a_drifted_oauth_slot(self, fleet):
        _make_live(OAUTH_EMAIL, _setup_creds(), "uuid-1")
        identity = _identity(fleet, "1")
        fleet._usage_store.record(
            {"1": FetchRecord(
                error="no_refresh_token",
                struck_fp=oauth.credential_fingerprint(_setup_creds()),
            )},
            identity,
        )
        assert fleet._usage_store.entries(identity)["1"].token_dead()  # PREMISE
        assert fleet._slot_token_dead("1", OAUTH_EMAIL) is True

    def test_active_setup_token_with_unreadable_backup_stays_statusline(
        self, fleet, cache
    ):
        now = time.time()
        _write_reading(cache, _body(now - 60, five=(20, now + HOUR)))
        _make_live(SETUP_EMAIL, _setup_creds())
        with patch.object(
            fleet, "_read_account_credentials_ex", return_value=("", True)
        ), _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
        assert spy.fetched == [OAUTH_EMAIL]
        assert entries["2"].decision_value()["five_hour"]["pct"] == 20.0

    def test_accesstoken_only_oauth_blob_is_still_fetched(self, temp_home, cache):
        """The stand-in shape many fixtures use for an OAuth login (no
        scopes) keeps the usage-endpoint path."""
        s = _linux_switcher()
        blob = json.dumps({"claudeAiOauth": {"accessToken": "at-x"}})
        _seed_oauth(s, 1, OAUTH_EMAIL, blob)
        with _NetworkSpy(s) as spy:
            entries = s._collect_usage_entries(
                [(1, OAUTH_EMAIL, "", "", False, blob, "")]
            )
        assert spy.fetched == [OAUTH_EMAIL]
        assert entries["1"].source is None


class TestListAndStatus:
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

    def test_list_json_pending(self, fleet):
        with _NetworkSpy(fleet):
            payload = fleet.list_accounts(json_output=True)
        row = payload["accounts"][1]
        assert row["usageStatus"] == "statusline_pending"
        assert row["usage"] is None
        assert row["usageSource"] == "statusline"
        assert "usageError" not in row and "lastGoodUsage" not in row

    def test_list_human(self, fleet, cache, capsys):
        now = time.time()
        _write_reading(cache, _body(now - 3 * HOUR - 30, five=(20, now + HOUR), seven=(30, now + DAY)))
        with _NetworkSpy(fleet):
            fleet.list_accounts()
        out = capsys.readouterr().out
        lines = out.splitlines()
        start = next(i for i, l in enumerate(lines) if SETUP_EMAIL in l)
        block = lines[start + 1:start + 3]
        assert "5h:" in block[0] and "via statusline" not in block[0]
        assert block[1].rstrip().endswith("· via statusline · 3h ago")

    def test_list_human_recent_reading_still_names_its_source(self, fleet, cache, capsys):
        now = time.time()
        _write_reading(cache, _body(now - 5, five=(20, now + HOUR), seven=None))
        with _NetworkSpy(fleet):
            fleet.list_accounts()
        assert "· via statusline · just now" in capsys.readouterr().out

    def test_list_human_pending(self, fleet, capsys):
        with _NetworkSpy(fleet):
            fleet.list_accounts()
        out = capsys.readouterr().out
        assert SENTINEL_NOTES[USAGE_STATUSLINE_PENDING] in out
        assert "no credentials" not in out

    def test_status_human_and_json(self, fleet, cache, capsys):
        now = time.time()
        _write_reading(cache, _body(now - 2 * HOUR, five=(20, now + HOUR), seven=None))
        _make_live(SETUP_EMAIL, _setup_creds())
        with _NetworkSpy(fleet) as spy:
            fleet.status()
            payload = fleet.status(json_output=True)
        assert spy.fetched == []
        assert "· via statusline · 2h ago" in capsys.readouterr().out
        active = payload["active"]
        assert active["number"] == 2
        assert active["usageStatus"] == "ok"
        assert active["usageSource"] == "statusline"
        assert active["usageAgeSeconds"] == pytest.approx(2 * HOUR, abs=5)

    def test_status_json_pending(self, fleet):
        _make_live(SETUP_EMAIL, _setup_creds())
        with _NetworkSpy(fleet):
            active = fleet.status(json_output=True)["active"]
        assert active["usageStatus"] == "statusline_pending"
        assert active["usageSource"] == "statusline"

    def test_status_json_oauth_row_has_no_source(self, fleet):
        with _NetworkSpy(fleet):
            active = fleet.status(json_output=True)["active"]
        assert active["number"] == 1
        assert "usageSource" not in active


class TestSwitchStrategies:
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

    def test_model_warning_ignores_statusline_readings(self, fleet, cache, capsys):
        """A configured model is no typo just because a setup-token slot,
        whose readings never carry per-model windows, shows none."""
        now = time.time()
        _write_reading(cache, _body(now, five=(10, now + HOUR)))
        value = usage_entry(_setup_creds(), now).decision_value()
        warnings: list[str] = []
        fleet._warn_inert_models({"2": value}, ("Fable",), True, warnings)
        fleet._warn_inert_models(
            {"2": USAGE_STATUSLINE_PENDING}, ("Fable",), True, warnings
        )
        assert warnings == []
        # An API reading beside it is still evidence.
        fleet._warn_inert_models(
            {"1": {"five_hour": {"pct": 1.0}}, "2": value}, ("Fable",), True, warnings
        )
        assert len(warnings) == 1 and "Fable" in warnings[0]

    def test_lockstep_warning_skips_statusline_rows(self, fleet, cache):
        """Two setup-tokens minted from one claude.ai account report the
        same windows by design; the issue #117 remedy does not apply."""
        fleet.add_account_from_token(OTHER_TOKEN, slot=3)
        now = time.time()
        body = _body(now - 60, five=(20, now + HOUR), seven=(30, now + DAY))
        _write_reading(cache, body)
        _write_reading(cache, body, OTHER_TOKEN)
        with _NetworkSpy(fleet):
            payload = fleet.list_accounts(json_output=True)
        assert [r["usageSource"] for r in payload["accounts"][1:]] == [
            "statusline", "statusline",
        ]
        assert "lockstepUsageWarnings" not in payload


class TestImportUsage:
    def test_setup_token_rows_adopt_without_effect(self, fleet, temp_home, capsys):
        """No credential is read to tell the slots apart (a Keychain read on
        macOS); the adopted row is simply never served for a setup-token."""
        from claude_swap.transfer import import_usage

        resets = _iso(time.time() + HOUR).replace("+00:00", "Z")
        rows = [
            {
                "email": email, "organizationUuid": "", "usageStatus": "ok",
                "usage": {"fiveHour": {"pct": pct, "resetsAt": resets}},
                "usageAgeSeconds": 30.0,
            }
            for email, pct in ((OAUTH_EMAIL, 11.0), (SETUP_EMAIL, 22.0))
        ]
        path = temp_home / "usage.json"
        path.write_text(json.dumps({
            "schemaVersion": 1, "activeAccountNumber": None, "accounts": rows,
        }), encoding="utf-8")
        with patch.object(
            fleet, "_read_account_credentials",
            side_effect=AssertionError("credential read"),
        ), patch.object(
            fleet, "_read_account_credentials_ex",
            side_effect=AssertionError("credential read"),
        ):
            import_usage(fleet, str(path), hold_s=600)

        err = capsys.readouterr().err
        assert "Done: 2 adopted, 0 kept, 0 skipped" in err
        with _NetworkSpy(fleet) as spy:
            entries = fleet._collect_usage_entries(fleet._build_accounts_info())
        assert spy.fetched == []  # slot 1 held for the adopted reading
        assert entries["1"].decision_value()["five_hour"]["pct"] == 11.0
        assert entries["2"].decision_value() == USAGE_STATUSLINE_PENDING
