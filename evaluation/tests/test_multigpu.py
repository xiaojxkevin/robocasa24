import json
from pathlib import Path
import tempfile
import unittest
from evaluation.multigpu import merge_results


class MergeTests(unittest.TestCase):
    def prepare(self, root, rows):
        replicas = []
        for gpu, tasks in enumerate(rows):
            replica = root / 'replicas' / str(gpu)
            replicas.append(dict(directory=str(replica), gpu=dict(index=gpu)))
            for task, records in tasks.items():
                path = replica / 'rollouts' / task
                path.mkdir(parents=True)
                (path / 'episodes.json').write_text(json.dumps(records))
        return replicas

    def test_global_merge_counts_each_episode_once_and_averages_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replicas = self.prepare(root, [
                {'a': [dict(seed=0, success=True, steps=10)], 'b': [dict(seed=100000, success=True, steps=20)]},
                {'a': [dict(seed=1, success=False, steps=500)]},
            ])
            summary = merge_results(root, replicas, {('a', 0), ('a', 1), ('b', 100000)}, 5., 10.)
            self.assertEqual(summary['completed_episodes'], 3)
            self.assertEqual(summary['macro_success_rate'], .75)
            self.assertEqual(summary['steps'], 530)
            self.assertEqual(summary['run_wall_seconds'], 5.)
            self.assertEqual(summary['startup_and_warmup_seconds'], 5.)
            merged = json.loads((root / 'a/episodes.json').read_text())
            self.assertEqual([r['seed'] for r in merged], [0, 1])
            self.assertEqual([r['replica_gpu'] for r in merged], [0, 1])

    def test_duplicate_and_missing_shards_are_rejected(self):
        for duplicate in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = [{'a': [dict(seed=0, success=True, steps=1)]}]
                if duplicate:
                    rows.append({'a': [dict(seed=0, success=True, steps=1)]})
                replicas = self.prepare(root, rows)
                with self.assertRaises(ValueError):
                    merge_results(root, replicas, {('a', 0), ('a', 1)}, 1., 2.)


if __name__ == '__main__':
    unittest.main()
