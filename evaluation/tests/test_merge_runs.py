import json
from pathlib import Path
import tempfile
import unittest

from evaluation.client import ROOT, episode_specs
from evaluation.merge_runs import merge_runs
from evaluation.protocol import benchmark_defaults


class MergeRunTests(unittest.TestCase):
    def source(self, root, name, seeds):
        path = root / name
        path.mkdir()
        tasks = ['CloseDrawer']
        manifest = dict(backend='pi05', checkpoint='/same/checkpoint', config={'contract': 'same'},
                        revisions={'sim': 'same'}, tasks=tasks, trials_per_task_per_seed=1,
                        replan_steps=32, benchmark_protocol=benchmark_defaults(), evaluation_seeds=seeds,
                        replicas=[{}], workers_per_gpu=8, max_workers_per_task=8)
        registry = json.loads((ROOT / 'evaluation/configs/tasks.json').read_text())
        rows = [dict(spec, success=spec['evaluation_seed'] != 1, steps=10)
                for spec in episode_specs(tasks, 1, seeds, registry)]
        (path / 'CloseDrawer').mkdir()
        (path / 'CloseDrawer/episodes.json').write_text(json.dumps(rows))
        (path / 'manifest.json').write_text(json.dumps(manifest))
        (path / 'summary.json').write_text(json.dumps(dict(complete=True, completed_episodes=len(rows),
                                                          run_wall_seconds=10, total_wall_seconds=12,
                                                          startup_and_warmup_seconds=2)))
        return path, rows

    def test_supplement_preserves_old_records_and_computes_three_seed_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, old_rows = self.source(root, 'old', [0, 1])
            new, _ = self.source(root, 'new', [2])
            summary = merge_runs([old, new], root / 'merged', [0, 1, 2])
            self.assertEqual(summary['completed_episodes'], 3)
            self.assertEqual(summary['success_count'], 2)
            self.assertAlmostEqual(summary['macro_success_rate'], 2 / 3)
            self.assertEqual(summary['evaluation_seeds'], [0, 1, 2])
            merged = json.loads((root / 'merged/CloseDrawer/episodes.json').read_text())
            self.assertEqual(merged[:2], old_rows)

    def test_missing_or_duplicate_seed_runs_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, _ = self.source(root, 'old', [0, 1])
            for sources in ([old], [old, old]):
                with self.assertRaises(ValueError):
                    merge_runs(sources, root / 'merged', [0, 1, 2])
                self.assertFalse((root / 'merged').exists())


if __name__ == '__main__':
    unittest.main()
