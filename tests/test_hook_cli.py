"""`cswap hook stop-failure`: a turn that died because the account's org
blocked Claude Code access disables that account and rotates off it; one a
setup-token account's 5h/7d limit refused records that window as full and
lets one auto-switch tick move off it.

Synthetic fixtures only (see `tests/test_real_store_guard.py`).
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli, statusline_usage
from claude_swap.switcher import ClaudeAccountSwitcher

from tests.test_account_policy_store import PolicyStoreBase

BLOCKED = {"hook_event_name": "StopFailure", "error": "oauth_org_not_allowed"}


class TestStopFailureHook(PolicyStoreBase):
    def _login(self, temp_home: Path, s: ClaudeAccountSwitcher, num: int) -> None:
        email = f"account-{num}@example.test"
        (temp_home / ".claude.json").write_text(json.dumps({"oauthAccount": {
            "emailAddress": email, "accountUuid": f"uuid-{num}"}}))
        s._write_credentials(s._read_account_credentials(str(num), email))

    @staticmethod
    def _hook(payload: dict, env: dict | None = None, live_blocked: bool = True) -> None:
        stdin = io.StringIO(json.dumps(payload))
        with patch.object(sys, "stdin", stdin), \
             patch.dict(os.environ, env or {}), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "hook", "stop-failure"]), \
             patch.object(cli, "_default_login_blocked", return_value=live_blocked), \
             patch.object(ClaudeAccountSwitcher, "_usage_by_account", return_value={}):
            cli.main()

    def test_blocked_active_account_is_disabled_and_switched_off(self, temp_home: Path):
        s = self._fleet(temp_home)
        self._login(temp_home, s, 1)
        assert s.current_account_number() == "1"

        self._hook(BLOCKED)

        assert s.is_account_disabled("1")
        assert s.current_account_number() == "2"

    def test_a_second_failure_does_not_blame_the_healthy_replacement(self, temp_home: Path):
        s = self._fleet(temp_home)
        self._login(temp_home, s, 1)
        self._hook(BLOCKED)

        self._hook(BLOCKED, live_blocked=False)

        assert s.disabled_account_numbers() == ["1"]
        assert s.current_account_number() == "2"

    def test_blocked_run_session_disables_only_its_account(self, temp_home: Path):
        s = self._fleet(temp_home)
        self._login(temp_home, s, 1)
        session = s._session_dir("2", "account-2@example.test")
        session.mkdir(parents=True)

        self._hook(BLOCKED, env={"CLAUDE_CONFIG_DIR": str(session)}, live_blocked=False)

        assert s.disabled_account_numbers() == ["2"]
        assert s.current_account_number() == "1"

    def test_other_errors_change_nothing(self, temp_home: Path):
        s = self._fleet(temp_home)
        self._login(temp_home, s, 1)
        before = Path(s.sequence_file).read_bytes()

        self._hook({**BLOCKED, "error": "server_error"})

        assert Path(s.sequence_file).read_bytes() == before


TOKEN_1 = "sk-ant-oat01-" + "a1b2c3d4e5" * 4
TOKEN_2 = "sk-ant-oat01-" + "f6g7h8i9j0" * 4
HOUR = 3600.0


class TestRateLimitHook(PolicyStoreBase):
    """Slot 1 is live and refused; slot 2 has room. Both are setup-tokens."""

    @pytest.fixture
    def fleet(self, temp_home: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("CLAUDE_USAGE_CACHE_DIR", str(tmp_path / "usage"))
        s = self._setup(temp_home)
        s.add_account_from_token(TOKEN_2, slot=2)
        s.add_account_from_token(TOKEN_1, slot=1)
        s.switch_to("1")
        assert s.current_account_number() == "1"
        return s

    @staticmethod
    def _reading(token: str, five: float, resets_at: float) -> None:
        path = statusline_usage.cache_path(token)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": 1, "source": "statusline", "captured_at": time.time() - HOUR,
            "five_hour": {"used_percentage": five, "resets_at": resets_at},
            "seven_day": {"used_percentage": 20},
        }))

    @staticmethod
    def _switched_ago(s: ClaudeAccountSwitcher, seconds: float) -> None:
        data = s._get_sequence_data()
        data["lastUpdated"] = datetime.fromtimestamp(
            time.time() - seconds, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        s._write_json(s.sequence_file, data)

    @staticmethod
    def _refused(tmp_path: Path, resets_at: float) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_text("\n".join(json.dumps(r) for r in [
            {"type": "user", "message": {"role": "user", "content": "go"}},
            {
                "type": "assistant", "error": "rate_limit", "isApiErrorMessage": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "quotaLimits": {
                    "status": "rejected", "rateLimitType": "five_hour", "resetsAt": resets_at,
                },
            },
        ]) + "\n")
        stdin = io.StringIO(json.dumps({
            "hook_event_name": "StopFailure", "error": "rate_limit",
            "transcript_path": str(transcript),
        }))
        with patch.object(sys, "stdin", stdin), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch.object(sys, "argv", ["cswap", "hook", "stop-failure"]):
            cli.main()

    def test_refused_account_is_recorded_full_and_switched_off(self, fleet, tmp_path):
        resets_at = float(int(time.time() + 2 * HOUR))
        self._reading(TOKEN_1, 76, resets_at)
        self._reading(TOKEN_2, 10, resets_at + HOUR)
        # Just switched onto slot 1: only its reading's reset can name it.
        self._switched_ago(fleet, 30)

        self._refused(tmp_path, resets_at)

        body = json.loads(statusline_usage.cache_path(TOKEN_1).read_text())
        assert body["five_hour"] == {"used_percentage": 100, "resets_at": resets_at}
        assert body["seven_day"] == {"used_percentage": 20}
        assert fleet.current_account_number() == "2"

    @pytest.mark.parametrize("switched_ago, marked", [(30, False), (2 * HOUR, True)])
    def test_unmatched_reset_blames_the_live_login_only_after_a_switch_settles(
        self, fleet, tmp_path, switched_ago, marked
    ):
        self._reading(TOKEN_1, 76, time.time() - HOUR)
        self._switched_ago(fleet, switched_ago)
        before = statusline_usage.cache_path(TOKEN_1).read_bytes()

        self._refused(tmp_path, float(int(time.time() + 2 * HOUR)))

        after = statusline_usage.cache_path(TOKEN_1).read_bytes()
        assert (after != before) is marked
