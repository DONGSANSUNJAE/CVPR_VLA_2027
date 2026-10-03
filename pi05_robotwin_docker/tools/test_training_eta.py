import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from training_eta import make_snapshot, read_records, read_status, recent_rate


class EtaTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 3, 9, 0, tzinfo=timezone.utc)
        self.status = dict(phase='training', completed_updates=120, target_updates=60000,
                           time_utc=self.now.isoformat())
        self.records = [dict(completed_updates=s, seconds_per_update=900 if s == 1 else 6)
                        for s in [1, 2, 3, 10, *range(20, 121, 10)]]

    def test_excludes_compilation_and_uses_100_updates(self):
        self.assertEqual(recent_rate(self.records, 120), (6, 100))
        r = make_snapshot(self.status, self.records, True, self.now)
        self.assertEqual(r['estimated_remaining_seconds'], 359280)
        self.assertEqual(r['estimated_finish_kst'], '2026-10-07T21:48:00+09:00')

    def test_weights_intervals_not_log_rows(self):
        rows = [dict(completed_updates=s, seconds_per_update=r)
                for s, r in [(3, 100), (10, 2), (20, 10), (30, 4)]]
        rate, n = recent_rate(rows, 30)
        self.assertEqual(n, 27)
        self.assertAlmostEqual(rate, 154 / 27)

    def test_rollback_cannot_mix_previous_session(self):
        records = self.records + [dict(completed_updates=110, seconds_per_update=800),
                                  dict(completed_updates=120, seconds_per_update=7)]
        self.assertEqual(recent_rate(records, 120), (None, 10))

    def test_not_running_suppresses_finish(self):
        r = make_snapshot(self.status, self.records, False, self.now)
        self.assertEqual(r['eta_status'], 'not_running')
        self.assertIsNone(r['estimated_finish_kst'])

    def test_stale_progress_suppresses_finish(self):
        r = make_snapshot(self.status, self.records, True, self.now + timedelta(hours=1))
        self.assertEqual(r['eta_status'], 'stale_progress')
        self.assertIsNone(r['estimated_finish_utc'])

    def test_failed_keeps_last_step_without_eta(self):
        r = make_snapshot(dict(phase='failed', last_status=self.status), self.records, False, self.now)
        self.assertEqual(r['completed_updates'], 120)
        self.assertEqual(r['eta_status'], 'stopped')
        self.assertIsNone(r['estimated_remaining_seconds'])

    def test_final_update_is_not_checkpoint_completion(self):
        r = make_snapshot(dict(self.status, completed_updates=60000), [], True, self.now)
        self.assertEqual(r['eta_status'], 'final_checkpoint_pending')
        r = make_snapshot(dict(self.status, phase='complete', completed_updates=60000), [], False, self.now)
        self.assertEqual(r['eta_status'], 'complete')
        self.assertEqual(r['estimated_remaining_seconds'], 0)

    def test_partial_log_and_stale_status_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            (logs / 'train_status.json').write_text(json.dumps(self.status))
            later = dict(phase='failed', time_utc=(self.now + timedelta(minutes=1)).isoformat(),
                         last_status=self.status)
            (logs / 'launch.log').write_text('unstructured text\n' + json.dumps(later) + '\n' + '{"partial":')
            self.assertEqual(len(read_records(logs / 'launch.log')), 1)
            r = read_status(logs)
            self.assertEqual(r['phase'], 'failed')
            self.assertEqual(r['source_status_file'], 'launch.log')


if __name__ == '__main__':
    unittest.main()
