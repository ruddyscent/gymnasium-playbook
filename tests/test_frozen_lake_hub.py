"""Regression tests for reproducible FrozenLake policy loading."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import Mock, patch

import numpy as np

from environments.toy_text.frozen_lake import __main__ as cli
from environments.toy_text.frozen_lake import model_loading
from environments.toy_text.frozen_lake.model_loading import LoadedPolicy
from environments.toy_text.frozen_lake.q_learning import evaluate


REPOSITORY = "example/frozenlake"
REVISION = "b221cf51b99a9ba07cb9d3de65430acdb94162a7"


def run_cli(*arguments: str) -> str:
    output = io.StringIO()
    with patch.object(sys, "argv", ["frozen_lake", *arguments]):
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            cli.main()
    return output.getvalue()


def config(max_episode_steps: int = 6, seed: int = 3) -> dict[str, object]:
    return {
        "environment": {
            "id": "FrozenLake-v1",
            "map_name": "4x4",
            "is_slippery": False,
        },
        "training": {"max_episode_steps": max_episode_steps, "seed": seed},
    }


class HubPolicyLoadingTests(unittest.TestCase):
    def _load_from_files(
        self, table: np.ndarray, metadata: object,
    ) -> tuple[LoadedPolicy, Mock]:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            q_table_path = directory / "q_table.npy"
            config_path = directory / "config.json"
            np.save(q_table_path, table, allow_pickle=False)
            config_path.write_text(json.dumps(metadata), encoding="utf-8")
            download = Mock(side_effect=[q_table_path, config_path])
            with patch.object(model_loading, "hub_download", download):
                policy = model_loading.load_hub_policy(REPOSITORY, REVISION, 3)
        return policy, download

    def test_loads_both_seed_artifacts_at_the_requested_immutable_revision(self) -> None:
        policy, download = self._load_from_files(np.zeros((16, 4)), config())
        self.assertEqual(policy.max_episode_steps, 6)
        self.assertEqual(
            download.call_args_list,
            [
                ((REPOSITORY, "seed-3/q_table.npy", REVISION),),
                ((REPOSITORY, "seed-3/config.json", REVISION),),
            ],
        )

    def test_saved_time_limit_preserves_goal_and_truncation_boundary(self) -> None:
        table = np.zeros((16, 4))
        for state, action in [(0, 1), (4, 1), (8, 2), (9, 1), (13, 2), (14, 2)]:
            table[state, action] = 1.0
        policy, _ = self._load_from_files(table, config(max_episode_steps=6))
        result = evaluate(policy.q_table, episodes=1, max_episode_steps=policy.max_episode_steps)
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["mean_episode_length"], 6.0)
        self.assertEqual(result["truncations"], 1)

    def test_rejects_invalid_tables_before_evaluation(self) -> None:
        for table in (
            np.zeros((4, 4)),
            np.full((16, 4), np.nan),
            np.full((16, 4), np.inf),
            np.full((16, 4), "not-a-number"),
        ):
            with self.subTest(dtype=table.dtype), self.assertRaisesRegex(ValueError, "Q-table"):
                self._load_from_files(table, config())

    def test_rejects_missing_or_unsupported_metadata(self) -> None:
        invalid_metadata: list[object] = [
            [],
            {"environment": config()["environment"]},
            {"environment": {"id": "FrozenLake-v1", "map_name": "8x8", "is_slippery": False}, "training": {"max_episode_steps": 6, "seed": 3}},
            {"environment": {"id": "FrozenLake-v1", "map_name": "4x4", "is_slippery": True}, "training": {"max_episode_steps": 6, "seed": 3}},
            {"environment": config()["environment"], "training": {"max_episode_steps": 0, "seed": 3}},
            {"environment": config()["environment"], "training": {"max_episode_steps": True, "seed": 3}},
        ]
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self._load_from_files(np.zeros((16, 4)), metadata)

    def test_reference_requires_a_repository_commit_and_nonnegative_seed(self) -> None:
        invalid_references = [
            ("repository", REVISION, 0),
            ("/frozenlake", REVISION, 0),
            ("example/", REVISION, 0),
            ("example/frozenlake/extra", REVISION, 0),
            ("example /frozenlake", REVISION, 0),
            (REPOSITORY, "main", 0),
            (REPOSITORY, REVISION, -1),
        ]
        for repository, revision, seed in invalid_references:
            with self.subTest(repository=repository, revision=revision, seed=seed):
                with self.assertRaises(ValueError):
                    model_loading.load_hub_policy(repository, revision, seed)

    def test_requires_matching_nonboolean_training_seed(self) -> None:
        invalid_metadata = [
            {"environment": config()["environment"], "training": {"max_episode_steps": 6}},
            config(seed=4),
            config(seed=True),
        ]
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                ValueError, "training.seed"
            ):
                self._load_from_files(np.zeros((16, 4)), metadata)

    def test_rejects_malformed_config_json(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            q_table_path = directory / "q_table.npy"
            config_path = directory / "config.json"
            np.save(q_table_path, np.zeros((16, 4)), allow_pickle=False)
            config_path.write_text("{", encoding="utf-8")
            with patch.object(
                model_loading, "hub_download", side_effect=[q_table_path, config_path],
            ), self.assertRaisesRegex(ValueError, "valid JSON"):
                model_loading.load_hub_policy(REPOSITORY, REVISION, 0)

    def test_missing_optional_dependency_explains_how_to_install_it(self) -> None:
        missing = ModuleNotFoundError("No module named huggingface_hub")
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(ValueError, "uv sync --locked --extra hub"):
                model_loading.hub_download(REPOSITORY, "seed-0/q_table.npy", REVISION)


class HubCommandLineTests(unittest.TestCase):
    def test_hub_source_requires_revision_and_seed(self) -> None:
        for arguments in (
            ("evaluate", "--hub-repo", REPOSITORY),
            ("watch", "--hub-repo", REPOSITORY, "--hub-revision", REVISION),
            ("evaluate", "--random", "--hub-seed", "0"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as exit:
                run_cli(*arguments)
            self.assertEqual(exit.exception.code, 2)

    def test_evaluate_uses_hub_time_limit_and_rejects_manual_override(self) -> None:
        policy = LoadedPolicy(np.zeros((16, 4)), 7)
        with patch.object(cli, "load_hub_policy", return_value=policy), patch.object(
            cli, "evaluate", return_value={
                "episodes": 1, "seed": 0, "max_episode_steps": 7, "successes": 0,
                "success_rate": 0.0, "mean_episode_length": 7.0,
                "mean_successful_episode_length": None, "truncations": 1,
            },
        ) as evaluation:
            output = run_cli(
                "evaluate", "--hub-repo", REPOSITORY, "--hub-revision", REVISION,
                "--hub-seed", "0", "--episodes", "1", "--seed", "0",
            )
        self.assertEqual(evaluation.call_args.args[3], 7)
        self.assertEqual(
            json.loads(output)["provenance"],
            {"repo_id": REPOSITORY, "revision": REVISION, "training_seed": 0},
        )
        with self.assertRaises(SystemExit) as exit:
            run_cli(
                "evaluate", "--hub-repo", REPOSITORY, "--hub-revision", REVISION,
                "--hub-seed", "0", "--max-episode-steps", "9",
            )
        self.assertEqual(exit.exception.code, 2)

    def test_watch_uses_hub_time_limit(self) -> None:
        policy = LoadedPolicy(np.zeros((16, 4)), 6)
        with patch.object(cli, "load_hub_policy", return_value=policy), patch(
            "environments.toy_text.frozen_lake.viewer.watch"
        ) as watch:
            run_cli(
                "watch", "--hub-repo", REPOSITORY, "--hub-revision", REVISION,
                "--hub-seed", "0",
            )
        self.assertEqual(watch.call_args.args[3], 6)

    def test_local_q_table_uses_the_existing_manual_episode_limit(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "q_table.npy"
            np.save(path, np.zeros((16, 4)), allow_pickle=False)
            output = run_cli(
                "evaluate", "--q-table", str(path), "--episodes", "1",
                "--max-episode-steps", "1",
            )
        result: dict[str, Any] = json.loads(output)
        self.assertEqual(result["max_episode_steps"], 1)
        self.assertEqual(result["truncations"], 1)
        self.assertNotIn("provenance", result)

    def test_explicit_zero_local_episode_limit_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as exit:
            run_cli("evaluate", "--random", "--max-episode-steps", "0")
        self.assertEqual(exit.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
