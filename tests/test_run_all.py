"""Runner exit-status and stop-on-failure tests without executing any lab step."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_all


class RunAllTests(unittest.TestCase):
    """Check CLI outcomes with mocked steps; no model or provider calls."""

    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_successful_step(self):
        module = Mock()
        with patch.object(run_all.importlib, "import_module", return_value=module):
            self.assertTrue(run_all.run_step(1))
        module.main.assert_called_once_with()

    def test_nonzero_step_exit_is_failure(self):
        module = Mock()
        module.main.side_effect = SystemExit(1)
        with patch.object(run_all.importlib, "import_module", return_value=module):
            self.assertFalse(run_all.run_step(1))

    def test_normal_system_exit_is_success(self):
        module = Mock()
        module.main.side_effect = SystemExit()
        with patch.object(run_all.importlib, "import_module", return_value=module):
            self.assertTrue(run_all.run_step(1))

    def test_import_failure_is_reported(self):
        with patch.object(run_all.importlib, "import_module", side_effect=ImportError):
            self.assertFalse(run_all.run_step(1))

    def test_full_run_stops_after_first_failure(self):
        with patch.object(run_all, "run_step", side_effect=[True, False]) as step:
            self.assertEqual(run_all.main([]), 1)
        self.assertEqual(step.call_args_list, [call(1), call(2)])

    def test_all_successful_steps_return_zero(self):
        with patch.object(run_all, "run_step", return_value=True) as step:
            self.assertEqual(run_all.main([]), 0)
        self.assertEqual(step.call_args_list, [call(1), call(2), call(3), call(4)])

    def test_single_step_success(self):
        with patch.object(run_all, "run_step", return_value=True) as step:
            self.assertEqual(run_all.main(["--step", "4"]), 0)
        step.assert_called_once_with(4)

    def test_single_step_failure_returns_nonzero(self):
        with patch.object(run_all, "run_step", return_value=False):
            self.assertEqual(run_all.main(["--step", "3"]), 1)


if __name__ == "__main__":
    unittest.main()
