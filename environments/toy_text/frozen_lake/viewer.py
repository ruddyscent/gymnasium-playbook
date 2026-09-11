"""Replay a fixed policy in a window; close the window or press Escape to exit."""

import gymnasium as gym
import numpy as np
import pygame

from .q_learning import QTable, validate_q_table


def watch(
    q_table: QTable | None,
    seed: int = 10_000,
    fps: int = 2,
    max_episode_steps: int = 100,
) -> None:
    if seed < 0 or not 1 <= fps <= 30 or max_episode_steps < 1:
        raise ValueError(
            "Require a nonnegative seed, 1 <= fps <= 30, and a positive episode limit"
        )
    if q_table is not None:
        q_table = validate_q_table(q_table)
    rng = np.random.default_rng(seed)
    pygame.init()
    try:
        with gym.make(
            "FrozenLake-v1", map_name="4x4", is_slippery=False,
            max_episode_steps=max_episode_steps, render_mode="rgb_array",
        ) as env:
            state, _ = env.reset(seed=seed)
            frame = env.render()
            screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            clock = pygame.time.Clock()
            steps = 0
            finished = False
            status = "Ready"
            next_step = pygame.time.get_ticks() + 1000
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        return
                now = pygame.time.get_ticks()
                if now >= next_step:
                    if finished:
                        state, _ = env.reset()
                        steps, finished, status = 0, False, "Ready"
                    else:
                        action = int(rng.integers(4)) if q_table is None else int(
                            np.argmax(q_table[state])
                        )
                        state, reward, terminated, truncated, _ = env.step(action)
                        steps += 1
                        finished = terminated or truncated
                        status = "Goal!" if float(reward) > 0 else (
                            "Hole" if terminated else "Time limit" if truncated else "Moving"
                        )
                    next_step = now + (2000 if finished else 1000 // fps)
                    frame = env.render()
                name = "Random" if q_table is None else "Q-learning"
                pygame.display.set_caption(
                    f"FrozenLake | {name} | Step {steps} | {status} | Esc: close"
                )
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                clock.tick(30)
    finally:
        pygame.quit()
