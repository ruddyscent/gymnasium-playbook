"""Train, evaluate, or compare seeds: python -m environments.toy_text.frozen_lake."""

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
import json
from pathlib import Path
import platform

import gymnasium as gym
import numpy as np

from .q_learning import (
    EvaluationResult, QTable, TrainingConfig, TrainingEpisode, evaluate, train,
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }


def save_training(
    output_dir: Path,
    config: TrainingConfig,
    q_table: QTable,
    history: list[TrainingEpisode],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "q_table.npy", q_table, allow_pickle=False)
    write_csv(output_dir / "training.csv", history)
    write_json(output_dir / "config.json", {
        "environment": {
            "id": "FrozenLake-v1", "map_name": "4x4", "is_slippery": False,
        },
        "training": asdict(config),
        "versions": versions(),
    })


def training_config(args: argparse.Namespace, seed: int) -> TrainingConfig:
    options = {field.name: getattr(args, field.name) for field in fields(TrainingConfig)
               if field.name != "seed"}
    return TrainingConfig(seed=seed, **options)


def add_training_options(parser: argparse.ArgumentParser) -> None:
    defaults = TrainingConfig()
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--discount", type=float, default=defaults.discount)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=defaults.epsilon_decay)
    parser.add_argument("--max-episode-steps", type=int, default=defaults.max_episode_steps)


def benchmark(args: argparse.Namespace) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Benchmark seeds must be distinct")
    if args.eval_episodes < 1 or args.eval_seed < 0:
        raise ValueError("Evaluation episodes must be positive and seed nonnegative")
    # Validate all configurations before starting a potentially long experiment.
    configs = [training_config(args, seed) for seed in args.seeds]
    rows: list[dict[str, object]] = []
    for config in configs:
        q_table, history = train(config)
        output_dir = args.output_dir / f"seed-{config.seed}"
        save_training(output_dir, config, q_table, history)
        evaluation_seed = args.eval_seed + config.seed
        results: dict[str, EvaluationResult] = {}
        for policy, table in (("q_learning", q_table), ("random", None)):
            result = evaluate(
                table, args.eval_episodes, evaluation_seed, config.max_episode_steps
            )
            results[policy] = result
            rows.append({"training_seed": config.seed, "policy": policy, **result})
            print(
                f"seed={config.seed} {policy}: "
                f"success={result['success_rate']:.1%}, "
                f"mean_length={result['mean_episode_length']:.2f}",
                flush=True,
            )
        write_json(output_dir / "evaluation.json", results)
    write_csv(args.output_dir / "summary.csv", rows)
    write_json(args.output_dir / "summary.json", {
        "versions": versions(),
        "training_configs": [asdict(config) for config in configs],
        "results": rows,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Learn and save a Q-table")
    add_training_options(training)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--output-dir", type=Path)

    evaluation = commands.add_parser("evaluate", help="Evaluate without learning")
    policy = evaluation.add_mutually_exclusive_group(required=True)
    policy.add_argument("--q-table", type=Path)
    policy.add_argument("--random", action="store_true")
    evaluation.add_argument("--episodes", type=int, default=1_000)
    evaluation.add_argument("--seed", type=int, default=10_000)
    evaluation.add_argument("--max-episode-steps", type=int, default=100)
    evaluation.add_argument("--output", type=Path, help="Optionally save evaluation JSON")

    comparison = commands.add_parser("benchmark", help="Compare learning with random play")
    add_training_options(comparison)
    comparison.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    comparison.add_argument("--eval-episodes", type=int, default=1_000)
    comparison.add_argument("--eval-seed", type=int, default=10_000)
    comparison.add_argument(
        "--output-dir", type=Path, default=Path("runs/frozen_lake/benchmark")
    )

    viewer = commands.add_parser("watch", help="Replay a policy in a window until closed")
    watched_policy = viewer.add_mutually_exclusive_group(required=True)
    watched_policy.add_argument("--q-table", type=Path)
    watched_policy.add_argument("--random", action="store_true")
    viewer.add_argument("--seed", type=int, default=10_000)
    viewer.add_argument("--fps", type=int, default=2, help="Actions per second (1-30)")

    args = parser.parse_args()
    try:
        if args.command == "train":
            config = training_config(args, args.seed)
            q_table, history = train(config)
            output_dir = args.output_dir or Path(f"runs/frozen_lake/seed-{args.seed}")
            save_training(output_dir, config, q_table, history)
            print(f"Saved Q-table and training records to {output_dir}")
        elif args.command == "evaluate":
            table = None if args.random else np.load(args.q_table, allow_pickle=False)
            result = {
                "policy": "random" if args.random else "q_learning",
                "versions": versions(),
                **evaluate(table, args.episodes, args.seed, args.max_episode_steps),
            }
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "watch":
            from .viewer import watch

            table = None if args.random else np.load(args.q_table, allow_pickle=False)
            watch(table, args.seed, args.fps)
        else:
            benchmark(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
