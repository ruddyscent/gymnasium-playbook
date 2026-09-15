"""The learning algorithm, separate from command-line and file handling."""

from dataclasses import dataclass
from collections.abc import Callable
from typing import TypedDict

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


type QTable = NDArray[np.float64]

TrainingEpisode = TypedDict("TrainingEpisode", {
    "episode": int,
    "epsilon": float,
    "return": float,
    "length": int,
    "success": int,
    "terminated": bool,
    "truncated": bool,
})

type EpisodeSink = Callable[[TrainingEpisode], None]


class EvaluationResult(TypedDict):
    episodes: int
    seed: int
    max_episode_steps: int
    successes: int
    success_rate: float
    mean_episode_length: float
    mean_successful_episode_length: float | None
    truncations: int


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

    def __post_init__(self) -> None:
        if self.seed < 0 or self.episodes < 1 or self.max_episode_steps < 1:
            raise ValueError("Seed must be nonnegative; episode counts and limits positive")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.discount <= 1:
            raise ValueError("Require 0 < learning_rate <= 1 and 0 <= discount <= 1")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("Require 0 < epsilon_decay <= 1")


def make_env(max_episode_steps: int = 100) -> gym.Env[int, int]:
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        max_episode_steps=max_episode_steps,
    )


def validate_q_table(q_table: object) -> QTable:
    """Return a finite real-valued table with one value per state-action pair."""
    if not isinstance(q_table, np.ndarray) or q_table.shape != (16, 4):
        raise ValueError("Expected a 16-by-4 Q-table")
    if (
        not np.issubdtype(q_table.dtype, np.number)
        or np.issubdtype(q_table.dtype, np.complexfloating)
    ):
        raise ValueError("Expected a real numeric 16-by-4 Q-table")
    table = q_table.astype(np.float64, copy=False)
    if not np.isfinite(table).all():
        raise ValueError("Expected a finite 16-by-4 Q-table")
    return table


def train(
    config: TrainingConfig, episode_sink: EpisodeSink | None = None,
) -> tuple[QTable, list[TrainingEpisode]]:
    """Return the learned Q-table and one record per training episode."""
    rng = np.random.default_rng(config.seed)
    # The fixed 4x4 map has 16 states and 4 actions: LEFT, DOWN, RIGHT, UP.
    q_table: QTable = np.zeros((16, 4), dtype=np.float64)
    history: list[TrainingEpisode] = []

    with make_env(config.max_episode_steps) as env:
        for episode in range(config.episodes):
            state, _ = env.reset(seed=config.seed if episode == 0 else None)
            epsilon = max(
                config.epsilon_end,
                config.epsilon_start * config.epsilon_decay**episode,
            )
            episode_return = 0.0
            for length in range(1, config.max_episode_steps + 1):
                if rng.random() < epsilon:
                    action = int(rng.integers(4))
                else:
                    action = int(np.argmax(q_table[state]))

                next_state, reward, terminated, truncated, _ = env.step(action)
                target = float(reward)
                if not terminated:
                    target += config.discount * float(q_table[next_state].max())
                q_table[state, action] += config.learning_rate * (
                    target - q_table[state, action]
                )
                episode_return += float(reward)
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
            if episode_sink is not None:
                episode_sink(history[-1].copy())
    return q_table, history


def evaluate(
    q_table: QTable | None,
    episodes: int = 1_000,
    seed: int = 10_000,
    max_episode_steps: int = 100,
) -> EvaluationResult:
    """Evaluate a fixed greedy table, or a uniform random policy when None."""
    if episodes < 1 or seed < 0 or max_episode_steps < 1:
        raise ValueError("Seed must be nonnegative; episode counts and limits positive")
    if q_table is not None:
        q_table = validate_q_table(q_table)

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
                    action = int(rng.integers(4))
                else:
                    # A fixed tie rule makes greedy evaluation deterministic.
                    action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                length += 1
                if terminated or truncated:
                    success = float(reward) > 0
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
