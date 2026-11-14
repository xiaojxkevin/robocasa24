import unittest
from unittest.mock import patch

from evaluation.client import episode_specs
from evaluation.pi05_multigpu import assign_tasks, cpu_partitions


class TaskAssignmentTests(unittest.TestCase):
    def test_cpu_groups_do_not_overlap(self):
        with patch('evaluation.pi05_multigpu.os.sched_getaffinity', return_value=set(range(224))):
            groups = cpu_partitions(8)
        all_cpus = [cpu for model, simulators in groups for cpu in model + simulators]
        self.assertEqual(sorted(all_cpus), list(range(224)))
        self.assertTrue(all(len(model) == 9 and len(simulators) == 19 for model, simulators in groups))

    def test_every_task_has_one_owner_and_every_episode_is_preserved(self):
        registry = {f'task_{i}': 300 + i * 20 for i in range(24)}
        groups = assign_tasks(list(registry), registry, 8)
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(groups))
        self.assertEqual(sorted(task for group in groups for task in group), sorted(registry))
        episodes = [spec for group in groups for spec in episode_specs(group, 50, [0, 1, 2], registry)]
        expected = episode_specs(list(registry), 50, [0, 1, 2], registry)
        self.assertEqual(len(episodes), 3600)
        self.assertEqual({(e['task'], e['seed']) for e in episodes},
                         {(e['task'], e['seed']) for e in expected})


if __name__ == '__main__':
    unittest.main()
