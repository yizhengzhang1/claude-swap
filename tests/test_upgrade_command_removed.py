"""`cswap upgrade` and the PyPI update check are gone from this fork.

The fork is installed from git, so a self-upgrade against PyPI would replace it
with upstream, and the passive "newer version" hint would advertise exactly
that. Both were removed rather than disabled.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from claude_swap import cli


class TestTheVerbIsGone:
    def test_the_verbs_are_not_in_the_subcommand_map(self):
        assert "upgrade" not in cli._SUBCOMMAND_FLAGS
        assert "update" not in cli._SUBCOMMAND_FLAGS

    @pytest.mark.parametrize(
        "argv", [["upgrade"], ["update"], ["--upgrade"]], ids=" ".join
    )
    def test_driving_it_through_main_fails(self, temp_home, argv):
        with patch("claude_swap.cli.ClaudeAccountSwitcher") as switcher_cls, \
             patch.object(sys, "argv", ["claude-swap", *argv]), \
             patch("os.geteuid", return_value=1000, create=True), \
             pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code not in (0, None)
        switcher_cls.assert_not_called()

    def test_help_does_not_mention_it(self, temp_home, capsys):
        with patch.object(sys, "argv", ["claude-swap", "--help"]), \
             pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "upgrade" not in out
        assert "Aliases: ls=list  rm=remove" in out

    def test_a_normal_command_prints_no_update_hint(self, temp_home, capsys):
        with patch("claude_swap.cli.ClaudeAccountSwitcher"), \
             patch.object(sys, "argv", ["claude-swap", "--list"]), \
             patch("os.geteuid", return_value=1000, create=True):
            cli.main()
        assert "newer version" not in capsys.readouterr().err

    @pytest.mark.parametrize("module", ["claude_swap.update_check", "claude_swap.cache"])
    def test_the_modules_are_gone(self, module):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
