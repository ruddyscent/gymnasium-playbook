"""The learning algorithm, separate from command-line and file handling."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    episodes: int = 10_000
    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999
    max_episode_steps: int = 100

    def __post_init__(self):
        if self.seed < 0 or self.episodes < 1 or self.max_episode_steps < 1:
            raise ValueError("Seed must be nonnegative; episode counts and limits positive")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.discount <= 1:
            raise ValueError("Require 0 < learning_rate <= 1 and 0 <= discount <= 1")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("Require 0 < epsilon_decay <= 1")


def make_env(max_episode_steps=100):
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        max_episode_steps=max_episode_steps,
    )


def epsilon_greedy(q_values, epsilon, rng):
    """Explore uniformly; otherwise break ties uniformly among the best actions."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best_actions = np.flatnonzero(q_values == q_values.max())
    # Initially every value is zero. Always choosing argmax would favor LEFT.
    return int(rng.choice(best_actions))


def q_learning_target(reward, next_value, terminated, discount):
    """Bootstrap unless the underlying task ended, even at a time limit."""
    if terminated:
        return float(reward)
    return float(reward + discount * next_value)


def update_q_value(
    q_table, state, action, reward, next_state, terminated, learning_rate, discount
):
    target = q_learning_target(
        reward, q_table[next_state].max(), terminated, discount
    )
    prediction = q_table[state, action]
    q_table[state, action] += learning_rate * (target - prediction)


def train(config):
    """Return the learned Q-table and one record per training episode."""
    rng = np.random.default_rng(config.seed)
    history = []

    with make_env(config.max_episode_steps) as env:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
        for episode in range(config.episodes):
            state, _ = env.reset(seed=config.seed if episode == 0 else None)
            epsilon = max(
                config.epsilon_end,
                config.epsilon_start * config.epsilon_decay**episode,
            )
            episode_return = 0.0
            length = 0
            while True:
                action = epsilon_greedy(q_table[state], epsilon, rng)
                next_state, reward, terminated, truncated, _ = env.step(action)
                update_q_value(
                    q_table, state, action, reward, next_state, terminated,
                    config.learning_rate, config.discount,
                )
                episode_return += float(reward)
                length += 1
                state = next_state
                # Both flags end the rollout; only terminated disables bootstrapping.
                if terminated or truncated:
                    break
            history.append({
                "episode": episode + 1,
                "epsilon": epsilon,
                "return": episode_return,
                "length": length,
                "success": int(episode_return > 0),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    return q_table, history


def evaluate(q_table, episodes=1_000, seed=10_000, max_episode_steps=100):
    """Evaluate a fixed greedy table, or a uniform random policy when None."""
    if episodes < 1 or seed < 0 or max_episode_steps < 1:
        raise ValueError("Seed must be nonnegative; episode counts and limits positive")
    if q_table is not None:
        if q_table.shape != (16, 4) or not np.isfinite(q_table).all():
            raise ValueError("Expected a finite 16-by-4 Q-table")

    rng = np.random.default_rng(seed)
    successes = 0
    total_steps = 0
    successful_steps = 0
    truncations = 0
    with make_env(max_episode_steps) as env:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            length = 0
            while True:
                if q_table is None:
                    action = int(rng.integers(env.action_space.n))
                else:
                    # A fixed tie rule makes greedy evaluation deterministic.
                    action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                length += 1
                if terminated or truncated:
                    success = reward > 0
                    successes += int(success)
                    successful_steps += length if success else 0
                    truncations += int(truncated)
                    total_steps += length
                    break

    return {
        "episodes": episodes,
        "seed": seed,
        "max_episode_steps": max_episode_steps,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_episode_length": total_steps / episodes,
        "mean_successful_episode_length": (
            successful_steps / successes if successes else None
        ),
        "truncations": truncations,
    }
