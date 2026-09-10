"""Optional logging must preserve learning and produce readable event streams."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from importlib.util import find_spec
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Literal
import unittest
from unittest.mock import Mock, patch

import numpy as np

from environments.toy_text.frozen_lake import __main__ as cli
from environments.toy_text.frozen_lake import tensorboard_logging as logging
from environments.toy_text.frozen_lake.q_learning import TrainingConfig, train


def run_cli(*arguments: str) -> None:
    with patch.object(sys, "argv", ["frozen_lake", *arguments]):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main()


def read_events(directory: Path) -> Any:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(
        str(directory), size_guidance={"scalars": 0, "tensors": 0},
    ).Reload()


class OptionalLoggingTests(unittest.TestCase):
    def test_disabled_training_does_not_import_tensorboard(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; "
                "from environments.toy_text.frozen_lake.q_learning import TrainingConfig, train; "
                "from environments.toy_text.frozen_lake import __main__; "
                "train(TrainingConfig(episodes=1)); "
                "assert not any(k == 'tensorboard' or k.startswith('tensorboard.') "
                "for k in sys.modules)"
            )], capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_dependency_is_actionable_and_precedes_training(self) -> None:
        missing = ModuleNotFoundError("No module named tensorboard", name="tensorboard")
        with TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "events"
            with patch.object(logging, "import_module", side_effect=missing):
                with self.assertRaisesRegex(ValueError, "uv sync --locked --extra tensorboard") as error:
                    logging.reserve_run(log_dir, "train", "missing")
                self.assertIn("uv run --locked --extra tensorboard", str(error.exception))
                with patch.object(cli, "train") as training, self.assertRaises(SystemExit) as exit:
                    run_cli("train", "--tensorboard", "--tensorboard-log-dir", str(log_dir))
                self.assertEqual(exit.exception.code, 2)
                training.assert_not_called()
            self.assertFalse(log_dir.exists())

    def test_auxiliary_flags_require_explicit_opt_in(self) -> None:
        for command in ("train", "benchmark"):
            for option in ("--tensorboard-log-dir", "--tensorboard-run-name"):
                with self.subTest(command=command, option=option):
                    with patch.object(cli, "train") as training, self.assertRaises(SystemExit) as exit:
                        run_cli(command, option, "unused")
                    self.assertEqual(exit.exception.code, 2)
                    training.assert_not_called()

    def test_run_names_are_portable_single_components(self) -> None:
        invalid = ["", ".", "..", "../escape", "a/b", "a\\b", "a:", "name.", "CON", "nul.txt", "LPT1", "a" * 101]
        with TemporaryDirectory() as temporary:
            for name in invalid:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    logging.reserve_run(Path(temporary), "train", name)
            self.assertEqual(list(Path(temporary).iterdir()), [])


@unittest.skipUnless(find_spec("tensorboard") is not None, "Install the tensorboard extra for event integration tests")
class TensorBoardIntegrationTests(unittest.TestCase):
    def test_events_match_history_and_metadata_without_changing_learning(self) -> None:
        config = TrainingConfig(seed=0, episodes=1_000)
        expected, expected_history = train(config)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with logging.episode_writer(directory, {"training": asdict(config)}) as sink:
                actual, history = train(config, episode_sink=sink)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(history, expected_history)
            self.assertGreater(actual.max(), 0.0)
            accumulator = read_events(directory)
            fields: tuple[Literal["return", "length", "epsilon", "success", "terminated", "truncated"], ...] = (
                "return", "length", "epsilon", "success", "terminated", "truncated",
            )
            self.assertEqual(set(accumulator.Tags()["scalars"]), {f"episode/{key}" for key in fields})
            for key in fields:
                events = accumulator.Scalars(f"episode/{key}")
                self.assertEqual([event.step for event in events], list(range(1, config.episodes + 1)))
                np.testing.assert_allclose(
                    [event.value for event in events], [row[key] for row in history], rtol=1e-6,
                )
            metadata = accumulator.Tensors("run/config")[0]
            self.assertEqual(metadata.step, 0)
            self.assertEqual(json.loads(metadata.tensor_proto.string_val[0]), {"training": asdict(config)})

    def test_train_cli_keeps_conventional_artifacts_identical(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = ["train", "--seed", "0", "--episodes", "100"]
            run_cli(*common, "--output-dir", str(root / "plain"))
            run_cli(
                *common, "--output-dir", str(root / "logged"), "--tensorboard",
                "--tensorboard-log-dir", str(root / "events"), "--tensorboard-run-name", "comparison",
            )
            for name in ("q_table.npy", "training.csv", "config.json"):
                self.assertEqual((root / "plain" / name).read_bytes(), (root / "logged" / name).read_bytes())
            directory = root / "events" / "train" / "comparison" / "seed-0"
            metadata = json.loads(read_events(directory).Tensors("run/config")[0].tensor_proto.string_val[0])
            self.assertEqual(metadata["command"], "train")
            self.assertEqual(metadata["run_name"], "comparison")
            self.assertEqual(metadata["training"], asdict(TrainingConfig(seed=0, episodes=100)))
            self.assertEqual(metadata["environment"], {
                "id": "FrozenLake-v1", "map_name": "4x4", "is_slippery": False, "max_episode_steps": 100,
            })
            self.assertEqual(set(metadata["versions"]), {"python", "gymnasium", "numpy", "system", "machine", "tensorboard"})
            self.assertEqual(metadata["versions"]["tensorboard"], "2.21.0")

    def test_benchmark_separates_seeds_and_repeated_invocations(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = [
                "benchmark", "--seeds", "2", "5", "--episodes", "3", "--eval-episodes", "2",
                "--eval-seed", "100", "--max-episode-steps", "2", "--output-dir", str(root / "artifacts"),
                "--tensorboard", "--tensorboard-log-dir", str(root / "events"),
            ]
            run_cli(*arguments)
            run_cli(*arguments)
            runs = list((root / "events" / "benchmark").iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertEqual({path.name for path in run.iterdir()}, {"seed-2", "seed-5"})
                for seed in (2, 5):
                    accumulator = read_events(run / f"seed-{seed}")
                    self.assertEqual([event.step for event in accumulator.Scalars("episode/return")], [1, 2, 3])
                    metadata = json.loads(accumulator.Tensors("run/config")[0].tensor_proto.string_val[0])
                    self.assertEqual(metadata["command"], "benchmark")
                    self.assertEqual(metadata["run_name"], run.name)
                    self.assertEqual(metadata["training"]["seed"], seed)
                    self.assertEqual(metadata["benchmark_seeds"], [2, 5])
                    self.assertEqual(metadata["evaluation"], {
                        "episodes": 2, "base_seed": 100, "seed": 100 + seed,
                        "max_episode_steps": 2, "policies": ["q_learning", "random"],
                    })

    def test_existing_root_fails_before_training(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for command in ("train", "benchmark"):
                with self.subTest(command=command):
                    root = directory / command / "collision"
                    root.mkdir(parents=True)
                    with patch.object(cli, "train") as training, self.assertRaises(SystemExit) as exit:
                        run_cli(command, "--tensorboard", "--tensorboard-log-dir", temporary, "--tensorboard-run-name", "collision")
                    self.assertEqual(exit.exception.code, 2)
                    training.assert_not_called()
                    self.assertEqual(list(root.iterdir()), [])

    def test_writer_preserves_completed_events_when_training_fails(self) -> None:
        _, history = train(TrainingConfig(episodes=1, max_episode_steps=1))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with logging.episode_writer(directory, {}) as sink:
                    sink(history[0])
                    raise RuntimeError("training failed")
            accumulator = read_events(directory)
            self.assertEqual(accumulator.Scalars("episode/truncated")[0].value, 1.0)
            self.assertEqual(accumulator.Scalars("episode/terminated")[0].value, 0.0)

    def test_metadata_and_episode_failures_always_flush_and_close(self) -> None:
        _, history = train(TrainingConfig(episodes=1))
        for stage in ("serialization", "metadata", "episode", "flush"):
            with self.subTest(stage=stage), TemporaryDirectory() as temporary:
                writer = Mock()
                if stage == "metadata":
                    writer.add_event.side_effect = RuntimeError("metadata failed")
                elif stage == "episode":
                    writer.add_event.side_effect = [None, RuntimeError("episode failed")]
                elif stage == "flush":
                    writer.flush.side_effect = RuntimeError("flush failed")
                metadata: dict[str, object] = {"unsupported": object()} if stage == "serialization" else {}
                with patch("tensorboard.summary.writer.event_file_writer.EventFileWriter", return_value=writer):
                    with self.assertRaises((RuntimeError, TypeError)):
                        with logging.episode_writer(Path(temporary), metadata) as sink:
                            sink(history[0])
                writer.flush.assert_called_once_with()
                writer.close.assert_called_once_with()

    def test_body_exception_survives_both_cleanup_failures(self) -> None:
        _, history = train(TrainingConfig(episodes=1))
        for stage in ("metadata", "episode", "training"):
            with self.subTest(stage=stage), TemporaryDirectory() as temporary:
                original = ValueError(f"{stage} failed")
                writer = Mock()
                writer.flush.side_effect = RuntimeError("flush failed")
                writer.close.side_effect = OSError("close failed")
                if stage == "metadata":
                    writer.add_event.side_effect = original
                elif stage == "episode":
                    writer.add_event.side_effect = [None, original]
                with patch("tensorboard.summary.writer.event_file_writer.EventFileWriter", return_value=writer):
                    with self.assertRaises(ValueError) as raised:
                        with logging.episode_writer(Path(temporary), {}) as sink:
                            sink(history[0])
                            raise original
                self.assertIs(raised.exception, original)
                self.assertEqual(original.__notes__, [
                    "TensorBoard writer flush also failed: RuntimeError('flush failed')",
                    "TensorBoard writer close also failed: OSError('close failed')",
                ])
                writer.flush.assert_called_once_with()
                writer.close.assert_called_once_with()

    def test_first_cleanup_exception_survives_second_cleanup_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            first = RuntimeError("flush failed")
            writer = Mock()
            writer.flush.side_effect = first
            writer.close.side_effect = OSError("close failed")
            with patch("tensorboard.summary.writer.event_file_writer.EventFileWriter", return_value=writer):
                with self.assertRaises(RuntimeError) as raised:
                    with logging.episode_writer(Path(temporary), {}):
                        pass
            self.assertIs(raised.exception, first)
            self.assertEqual(first.__notes__, [
                "TensorBoard writer close also failed: OSError('close failed')",
            ])
            writer.flush.assert_called_once_with()
            writer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
