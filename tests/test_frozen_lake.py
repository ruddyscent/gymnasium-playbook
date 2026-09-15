"""Regression tests for the learning semantics, not a performance benchmark."""

import unittest
from unittest.mock import Mock, patch
from typing import Any

import numpy as np

from environments.toy_text.frozen_lake.q_learning import (
    QTable,
    TrainingConfig,
    TrainingEpisode,
    evaluate,
    make_env,
    train,
)


def train_from_table(
    table: QTable,
    config: TrainingConfig,
    actions: list[int] | None = None,
) -> tuple[QTable, list[TrainingEpisode]]:
    """Start the real training loop with known values and optional forced actions."""
    # Replace NumPy only in the learning module; Gymnasium keeps its own RNG.
    with patch("environments.toy_text.frozen_lake.q_learning.np", wraps=np) as numpy:
        numpy.float64 = np.float64
        numpy.zeros.return_value = table.copy()
        if actions is not None:
            rng = Mock(wraps=np.random.default_rng(config.seed))
            rng.random.return_value = 0.0
            rng.integers.side_effect = actions
            numpy.random.default_rng.return_value = rng
        return train(config)


class LearningTests(unittest.TestCase):
    def test_nonterminal_update_includes_discounted_future_value(self) -> None:
        table = np.zeros((16, 4))
        table[0, 2] = 2.0
        table[1, 0] = 4.0
        expected = table.copy()
        expected[0, 2] = 2.4  # 2 + 0.25 * ((0 + 0.9 * 4) - 2)
        actual, history = train_from_table(
            table,
            TrainingConfig(episodes=1, max_episode_steps=1, learning_rate=0.25, discount=0.9),
            [2],
        )
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(history[0]["terminated"])
        self.assertTrue(history[0]["truncated"])

    def test_real_time_limit_still_bootstraps(self) -> None:
        table = np.zeros((16, 4))
        table[1, 0] = 4.0
        actual, history = train_from_table(
            table,
            TrainingConfig(episodes=1, max_episode_steps=1, learning_rate=1.0, discount=0.9),
            [2],
        )
        self.assertAlmostEqual(actual[0, 2], 3.6)
        self.assertFalse(history[0]["terminated"])
        self.assertTrue(history[0]["truncated"])

    def test_terminal_target_ignores_even_a_large_next_value(self) -> None:
        for limit in (6, 100):
            with self.subTest(max_episode_steps=limit):
                table = np.zeros((16, 4))
                table[15] = 100.0
                actual, history = train_from_table(
                    table,
                    TrainingConfig(episodes=1, max_episode_steps=limit, learning_rate=1.0),
                    [1, 1, 2, 1, 2, 2],
                )
                self.assertEqual(actual[14, 2], 1.0)
                self.assertTrue(history[0]["terminated"])
                self.assertEqual(history[0]["truncated"], limit == 6)

    def test_hole_terminates_with_zero_target(self) -> None:
        table = np.zeros((16, 4))
        table[5] = 100.0
        table[1, 1] = 2.0
        actual, history = train_from_table(
            table, TrainingConfig(episodes=1, learning_rate=1.0), [2, 1],
        )
        self.assertEqual(actual[1, 1], 0.0)
        self.assertTrue(history[0]["terminated"])
        self.assertEqual(history[0]["success"], 0)

    def test_exploitation_chooses_the_first_best_action(self) -> None:
        table = np.zeros((16, 4))
        table[0] = [0.0, 2.0, 2.0, 0.0]
        # Keep the best values tied after updates to check the fixed tie rule.
        table[1, 0] = table[4, 0] = 2.0
        with make_env(max_episode_steps=1) as env, patch(
            "environments.toy_text.frozen_lake.q_learning.make_env", return_value=env,
        ), patch.object(env, "step", wraps=env.step) as step:
            train_from_table(
                table,
                TrainingConfig(seed=42, episodes=100, max_episode_steps=1,
                               epsilon_start=0.0, epsilon_end=0.0, discount=1.0),
            )
        self.assertEqual({call.args[0] for call in step.call_args_list}, {1})

    def test_full_exploration_can_choose_every_action(self) -> None:
        table = np.zeros((16, 4))
        table[0, 0] = 10.0
        with make_env(max_episode_steps=1) as env, patch(
            "environments.toy_text.frozen_lake.q_learning.make_env", return_value=env,
        ), patch.object(env, "step", wraps=env.step) as step:
            train_from_table(
                table,
                TrainingConfig(seed=42, episodes=100, max_episode_steps=1,
                               epsilon_start=1.0, epsilon_end=1.0),
            )
        self.assertEqual({call.args[0] for call in step.call_args_list}, {0, 1, 2, 3})

    def test_training_is_reproducible_and_honors_time_limit(self) -> None:
        config = TrainingConfig(seed=7, episodes=20, max_episode_steps=1)
        first, first_history = train(config)
        second, second_history = train(config)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_history, second_history)
        self.assertTrue(all(row["length"] == 1 for row in first_history))
        self.assertTrue(all(row["truncated"] for row in first_history))

    def test_training_reproduces_learned_values_not_just_initial_zeros(self) -> None:
        config = TrainingConfig(seed=0, episodes=1_000)
        first, first_history = train(config)
        second, second_history = train(config)
        self.assertGreater(first.max(), 0.0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_history, second_history)

    def test_episode_sink_receives_copies_without_changing_learning(self) -> None:
        config = TrainingConfig(seed=0, episodes=1_000)
        expected, expected_history = train(config)
        observed: list[TrainingEpisode] = []

        def mutate_copy(row: TrainingEpisode) -> None:
            observed.append(row.copy())
            row["return"] = -100.0
            row["episode"] = -1

        actual, history = train(config, episode_sink=mutate_copy)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(history, expected_history)
        self.assertEqual(observed, history)

    def test_episode_sink_preserves_goal_and_time_limit_flags(self) -> None:
        observed: list[TrainingEpisode] = []
        config = TrainingConfig(episodes=1, max_episode_steps=6)
        rng = Mock(wraps=np.random.default_rng(config.seed))
        rng.random.return_value = 0.0
        rng.integers.side_effect = [1, 1, 2, 1, 2, 2]
        with patch("environments.toy_text.frozen_lake.q_learning.np", wraps=np) as numpy:
            numpy.float64 = np.float64
            numpy.random.default_rng.return_value = rng
            _, history = train(config, episode_sink=observed.append)
        self.assertEqual(observed, history)
        self.assertEqual(observed[0]["episode"], 1)
        self.assertTrue(observed[0]["terminated"])
        self.assertTrue(observed[0]["truncated"])
        self.assertEqual(observed[0]["success"], 1)

    def test_evaluation_is_read_only_and_reports_success_and_length(self) -> None:
        # A known six-step policy tests evaluation independently of learning.
        table = np.zeros((16, 4))
        for state, action in [(0, 1), (4, 1), (8, 2), (9, 1), (13, 2), (14, 2)]:
            table[state, action] = 1.0
        table.setflags(write=False)
        result = evaluate(table, episodes=5)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["mean_episode_length"], 6.0)
        self.assertEqual(result["mean_successful_episode_length"], 6.0)
        self.assertEqual(result["truncations"], 0)

    def test_zero_table_times_out_with_fixed_greedy_tie_rule(self) -> None:
        result = evaluate(np.zeros((16, 4)), episodes=3, max_episode_steps=4)
        self.assertEqual(result["success_rate"], 0.0)
        self.assertEqual(result["mean_episode_length"], 4.0)
        self.assertEqual(result["truncations"], 3)
        self.assertIsNone(result["mean_successful_episode_length"])

    def test_random_evaluation_is_reproducible(self) -> None:
        self.assertEqual(evaluate(None, 50, 7), evaluate(None, 50, 7))

    def test_invalid_inputs_fail_before_training_or_evaluation(self) -> None:
        invalid_options: list[dict[str, Any]] = [
            {"episodes": 0}, {"learning_rate": 0}, {"discount": float("nan")},
            {"epsilon_end": 1, "epsilon_start": 0}, {"seed": -1},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingConfig(**options)
        with self.assertRaises(ValueError):
            evaluate(np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            evaluate(None, episodes=0)


if __name__ == "__main__":
    unittest.main()
