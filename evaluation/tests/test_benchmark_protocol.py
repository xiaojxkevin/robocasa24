import argparse
import json
from pathlib import Path
import tempfile
import unittest
from evaluation.client import ROOT, episode_specs, success_metrics
from evaluation.multigpu import merge_results
from evaluation.protocol import add_benchmark_arguments, resolve_evaluation_seeds


class BenchmarkProtocolTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads((ROOT / 'evaluation/configs/tasks.json').read_text())
        self.tasks = list(self.registry)

    def test_defaults_are_24_tasks_50_trials_on_three_seeds(self):
        parser = argparse.ArgumentParser()
        add_benchmark_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.tasks, ['all'])
        self.assertEqual(args.episodes, 50)
        self.assertEqual(resolve_evaluation_seeds(args), [0, 1, 2])
        self.assertEqual(args.replan_steps, 32)
        specs = episode_specs(self.tasks, args.episodes, resolve_evaluation_seeds(args), self.registry)
        self.assertEqual(len(specs), 3600)
        self.assertEqual(len({row['seed'] for row in specs}), 3600)
        self.assertEqual(sum(row['evaluation_seed'] == 0 for row in specs), 1200)
        self.assertEqual(sum(row['evaluation_seed'] == 1 for row in specs), 1200)
        self.assertEqual(sum(row['evaluation_seed'] == 2 for row in specs), 1200)
        a = {row['seed'] for row in specs if row['evaluation_seed'] == 0}
        b = {row['seed'] for row in specs if row['evaluation_seed'] == 1}
        self.assertFalse(a & b)

    def test_appending_third_seed_preserves_existing_test_cases(self):
        original = episode_specs(self.tasks, 50, [0, 1], self.registry)
        expanded = episode_specs(self.tasks, 50, [0, 1, 2], self.registry)
        self.assertEqual(len(expanded), 3600)
        self.assertEqual(expanded[:2400], original)
        self.assertEqual(len({row['seed'] for row in expanded}), 3600)

    def test_multiseed_gpu_shards_have_complete_nonoverlapping_coverage(self):
        whole = episode_specs(self.tasks, 50, [0, 1], self.registry)
        expected = {(r['task'], r['evaluation_seed'], r['trial_index'], r['seed']) for r in whole}
        for gpu_count in [1, 2, 8]:
            merged = [r for gpu in range(gpu_count) for r in
                      episode_specs(self.tasks, 50, [0, 1], self.registry, gpu, gpu_count)]
            self.assertEqual(len(merged), 2400)
            self.assertEqual({(r['task'], r['evaluation_seed'], r['trial_index'], r['seed']) for r in merged}, expected)
        for seeds in [[0, 0], [-1, 1], [1000]]:
            with self.assertRaises(ValueError):
                episode_specs(self.tasks, 50, seeds, self.registry)

    def test_seed_metrics_and_global_merge_preserve_two_seed_groups(self):
        rows = {
            'a': [dict(seed=0, evaluation_seed=0, trial_index=0, success=True, steps=10),
                  dict(seed=10000000, evaluation_seed=1, trial_index=0, success=True, steps=10)],
            'b': [dict(seed=100000, evaluation_seed=0, trial_index=0, success=True, steps=10),
                  dict(seed=10100000, evaluation_seed=1, trial_index=0, success=False, steps=500)],
        }
        metrics = success_metrics(rows)
        self.assertEqual(metrics['per_evaluation_seed']['0']['macro_success_rate'], 1.)
        self.assertEqual(metrics['per_evaluation_seed']['1']['macro_success_rate'], .5)
        self.assertEqual(metrics['mean_success_rate_across_seeds'], .75)
        self.assertAlmostEqual(metrics['std_success_rate_across_seeds'], 2**-.5 / 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            replicas = []
            for i in range(2):
                replica = output / f'replica_{i}'
                replicas.append(dict(directory=str(replica), gpu=dict(index=i)))
                for task, records in rows.items():
                    target = replica / 'rollouts' / task
                    target.mkdir(parents=True)
                    (target / 'episodes.json').write_text(json.dumps([records[i]]))
            expected = {(task, r['seed']) for task, records in rows.items() for r in records}
            summary = merge_results(output, replicas, expected, 2., 5.)
            self.assertEqual(summary['completed_episodes'], 4)
            self.assertEqual(summary['per_evaluation_seed'], metrics['per_evaluation_seed'])


if __name__ == '__main__':
    unittest.main()
