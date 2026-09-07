"""Regression tests for the learning semantics, not a performance benchmark."""

import unittest
from typing import Any

import numpy as np

from environments.toy_text.frozen_lake.q_learning import (
    TrainingConfig,
    epsilon_greedy,
    evaluate,
    make_env,
    q_learning_target,
    train,
    update_q_value,
)


class LearningTests(unittest.TestCase):
    def test_nonterminal_target_includes_discounted_future_value(self) -> None:
        self.assertAlmostEqual(q_learning_target(1.0, 4.0, False, 0.9), 4.6)

    def test_terminal_target_ignores_even_a_large_next_value(self) -> None:
        self.assertEqual(q_learning_target(1.0, 100.0, True, 0.9), 1.0)

    def test_update_interpolates_one_entry(self) -> None:
        table = np.zeros((16, 4))
        table[0, 2] = 2.0
        table[1, 0] = 4.0
        expected = table.copy()
        expected[0, 2] = 2.65  # 2 + 0.25 * ((1 + 0.9 * 4) - 2)
        update_q_value(table, 0, 2, 1.0, 1, False, 0.25, 0.9)
        np.testing.assert_allclose(table, expected)

    def test_real_time_limit_still_bootstraps(self) -> None:
        with make_env(max_episode_steps=1) as env:
            state, _ = env.reset(seed=0)
            next_state, reward, terminated, truncated, _ = env.step(2)
        self.assertEqual(next_state, 1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        table = np.zeros((16, 4))
        table[next_state, 0] = 4.0
        update_q_value(table, state, 2, reward, next_state, terminated, 1.0, 0.9)
        self.assertAlmostEqual(table[state, 2], 3.6)

    def test_goal_on_time_limit_does_not_bootstrap(self) -> None:
        with make_env(max_episode_steps=6) as env:
            env.reset(seed=0)
            for action in [1, 1, 2, 1, 2, 2]:
                _, reward, terminated, truncated, _ = env.step(action)
        self.assertTrue(terminated)
        self.assertTrue(truncated)
        self.assertEqual(q_learning_target(reward, 100.0, terminated, 0.99), 1.0)

    def test_hole_terminates_with_zero_target(self) -> None:
        with make_env() as env:
            env.reset(seed=0)
            env.step(2)
            _, reward, terminated, _, _ = env.step(1)
        self.assertTrue(terminated)
        self.assertEqual(q_learning_target(reward, 100.0, terminated, 0.99), 0.0)

    def test_exploitation_breaks_ties_without_choosing_worse_actions(self) -> None:
        rng = np.random.default_rng(42)
        actions = {epsilon_greedy(np.array([0.0, 2.0, 2.0, 0.0]), 0.0, rng)
                   for _ in range(100)}
        self.assertEqual(actions, {1, 2})

    def test_full_exploration_can_choose_every_action(self) -> None:
        rng = np.random.default_rng(42)
        actions = {epsilon_greedy(np.array([10.0, 0.0, 0.0, 0.0]), 1.0, rng)
                   for _ in range(100)}
        self.assertEqual(actions, {0, 1, 2, 3})

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
