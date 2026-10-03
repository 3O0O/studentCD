"""CPU-only real-file budget/lock tests; no models, network or credentials."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from student_sim_cd import night_budget
from student_sim_cd.night_budget import NightBudget


STOP_BY = "2030-01-01T01:00:00+00:00"
NOW = datetime.fromisoformat("2030-01-01T00:00:00+00:00").timestamp()


class NightBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="night-budget-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "budget.json"
        patcher = mock.patch.object(night_budget, "time", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)

    def budget(self, **kwargs):
        options = {"stop_by": STOP_BY, "max_seconds": 10}
        options.update(kwargs)
        result = NightBudget(self.path, **options)
        self.addCleanup(result.finish)
        return result

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def child(self, *, seconds=6, finish=False, expected_busy=False):
        code = (
            "import sys\n"
            "from student_sim_cd.night_budget import NightBudget\n"
            "b = NightBudget(sys.argv[1], stop_by=sys.argv[2], max_seconds=10)\n"
            "try:\n"
            "    allowed = b.acquire(float(sys.argv[3]))\n"
            "except ValueError as exc:\n"
            "    print(str(exc))\n"
            "    raise SystemExit(9)\n"
            "print(allowed)\n"
            + ("b.finish()\n" if finish else "")
        )
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.path), STOP_BY, str(seconds)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 9 if expected_busy else 0, result.stderr)
        return result

    def test_construction_has_zero_usage_and_does_not_create_a_ledger(self):
        self.budget()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_name("budget.json.lock").exists())

    def test_cross_call_total_reservations_and_ceil_settlement(self):
        first = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[100, 102.1]):
            self.assertEqual(first.acquire(6), 6)
            self.assertEqual(self.read()["entries"][0]["charged_seconds"], 6)
            first.finish()
        self.assertEqual(self.read()["entries"][0]["charged_seconds"], 3)
        second = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[200, 209.2]):
            self.assertEqual(second.acquire(10), 7)
            second.finish()
        entries = self.read()["entries"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(sum(entry["charged_seconds"] for entry in entries), 10)
        self.assertLessEqual(entries[1]["charged_seconds"], entries[1]["reserved_seconds"])
        with self.assertRaisesRegex(ValueError, "exhausted"):
            self.budget().acquire(1)
        self.assertEqual(len(self.read()["entries"]), 2)

    def test_real_subprocess_exit_without_finish_retains_full_precharge(self):
        result = self.child(seconds=6)
        self.assertEqual(float(result.stdout), 6)
        self.assertEqual(self.read()["entries"][0]["status"], "reserved")
        self.assertEqual(self.read()["entries"][0]["charged_seconds"], 6)
        next_budget = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[100, 100.2]):
            self.assertEqual(next_budget.acquire(10), 4)
            next_budget.finish()
        self.assertEqual(self.read()["entries"][0]["charged_seconds"], 6)
        self.assertEqual(self.read()["entries"][1]["charged_seconds"], 1)

    def test_nonblocking_lock_rejects_same_process_and_actual_subprocess(self):
        budget = self.budget()
        budget.acquire(6)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "active lease"):
            self.budget().acquire(1)
        self.assertIn("active lease", self.child(seconds=1, expected_busy=True).stdout)
        self.assertEqual(before, self.path.read_bytes())
        budget.finish()
        self.assertTrue(self.path.with_name("budget.json.lock").exists())

    def test_deadline_limits_allowed_time_and_expiry_does_not_create_entry(self):
        near_deadline = self.budget(stop_by="2030-01-01T00:00:02.500000+00:00")
        with mock.patch.object(night_budget, "monotonic", side_effect=[1, 1.1]):
            self.assertEqual(near_deadline.acquire(10), 2.5)
            near_deadline.finish()
        new_path = self.root / "expired.json"
        expired = NightBudget(new_path, stop_by="2030-01-01T00:00:00+00:00", max_seconds=10)
        with self.assertRaisesRegex(ValueError, "deadline"):
            expired.acquire(1)
        self.assertFalse(new_path.exists())

    def test_damaged_json_duplicate_fields_and_invalid_entries_are_preserved(self):
        for content in (b"broken json", b'{"schema_version":1,"schema_version":2}', b'{"value":NaN}'):
            with self.subTest(content=content):
                self.path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "damaged"):
                    self.budget().acquire(1)
                self.assertEqual(self.path.read_bytes(), content)
        self.path.unlink()
        budget = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[1, 2]):
            budget.acquire(2)
            budget.finish()
        ledger = self.read()
        ledger["entries"][0]["charged_seconds"] = -1
        self.path.write_text(json.dumps(ledger), encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.budget().acquire(1)
        self.assertEqual(before, self.path.read_bytes())

    def test_metadata_mismatch_preserves_original_ledger(self):
        budget = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[1, 2]):
            budget.acquire(2)
            budget.finish()
        before = self.path.read_bytes()
        for options in ({"max_seconds": 20}, {"stop_by": "2030-01-01T02:00:00+00:00"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "metadata"):
                self.budget(**options).acquire(1)
            self.assertEqual(before, self.path.read_bytes())

    def test_invalid_configuration_and_requested_seconds_fail_without_a_ledger(self):
        for stop in ("2030-01-01T01:00:00", "not-a-date", 42):
            with self.subTest(stop=stop), self.assertRaises(ValueError):
                self.budget(stop_by=stop)
        for value in (0, -1, True, "10", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.budget(max_seconds=value)
                with self.assertRaises(ValueError):
                    self.budget().acquire(value)
        self.assertFalse(self.path.exists())

    def test_ledger_parent_and_lock_symlinks_are_rejected(self):
        target = self.root / "target.json"
        target.write_bytes(b"unchanged")
        self.path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.budget()
        self.path.unlink()
        link = self.root / "linked-parent"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            NightBudget(link / "ledger.json", stop_by=STOP_BY)
        self.path.with_name("budget.json.lock").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.budget()
        self.assertEqual(target.read_bytes(), b"unchanged")

    def test_finish_is_idempotent_and_invalid_clock_keeps_full_reservation(self):
        budget = self.budget()
        budget.finish()
        with mock.patch.object(night_budget, "monotonic", side_effect=[10, 9]):
            budget.acquire(5)
            with self.assertRaisesRegex(ValueError, "monotonic"):
                budget.finish()
        self.assertEqual(self.read()["entries"][0]["charged_seconds"], 5)
        self.assertEqual(self.read()["entries"][0]["status"], "reserved")
        budget.finish()
        another = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[20, 20]):
            self.assertEqual(another.acquire(10), 5)
            another.finish()

    def test_fractional_reservation_never_charges_more_than_reserved(self):
        budget = self.budget()
        with mock.patch.object(night_budget, "monotonic", side_effect=[0, 0.05]):
            self.assertEqual(budget.acquire(0.2), 0.2)
            budget.finish()
        self.assertEqual(self.read()["entries"][0]["charged_seconds"], 0.2)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_corruption_during_lease_is_preserved_on_finish_and_lock_is_released(self):
        budget = self.budget()
        budget.acquire(5)
        damaged = b"damaged during active lease"
        self.path.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, "damaged"):
            budget.finish()
        self.assertEqual(self.path.read_bytes(), damaged)
        with self.assertRaisesRegex(ValueError, "damaged"):
            self.budget().acquire(1)
        self.assertEqual(self.path.read_bytes(), damaged)

    def test_nonregular_fifo_ledger_fails_without_blocking_or_replacing_it(self):
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.budget().acquire(1)
        self.assertTrue(self.path.exists())


if __name__ == "__main__":
    unittest.main()
