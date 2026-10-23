"""`cswap hook stop-failure`: a turn that died because the account's org
blocked Claude Code access disables that account and rotates off it.

Synthetic fixtures only (see `tests/test_real_store_guard.py`).
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from claude_swap import cli
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

        self._hook({**BLOCKED, "error": "rate_limit"})

        assert Path(s.sequence_file).read_bytes() == before
