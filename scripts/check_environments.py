"""Check installed environment families without opening a display window."""

import argparse
import math
from typing import Any

import gymnasium as gym


def check_environment(env_id: str, **kwargs: Any) -> None:
    with gym.make(env_id, **kwargs) as env:
        observation, _ = env.reset(seed=42)
        env.action_space.seed(42)
        if not env.observation_space.contains(observation):
            raise AssertionError(f"{env_id}: invalid reset observation")
        for _ in range(20):
            observation, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if not env.observation_space.contains(observation):
                raise AssertionError(f"{env_id}: invalid step observation")
            if not math.isfinite(float(reward)):
                raise AssertionError(f"{env_id}: non-finite reward")
            if terminated or truncated:
                env.reset()
        if kwargs.get("render_mode") == "rgb_array":
            frame = env.render()
            if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
                raise AssertionError(f"{env_id}: invalid RGB render")
    print(f"PASS {env_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Check all optional families")
    for family in ("box2d", "mujoco", "atari"):
        parser.add_argument(f"--{family}", action="store_true")
    args = parser.parse_args()

    check_environment("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    check_environment("CartPole-v1", render_mode="rgb_array")
    if args.all or args.box2d:
        check_environment("LunarLander-v3", render_mode="rgb_array")
    if args.all or args.mujoco:
        # Physics does not need an OpenGL context; rendering is a separate check.
        check_environment("InvertedPendulum-v5")
    if args.all or args.atari:
        import ale_py

        gym.register_envs(ale_py)
        check_environment("ALE/Pong-v5", render_mode="rgb_array")


if __name__ == "__main__":
    main()
