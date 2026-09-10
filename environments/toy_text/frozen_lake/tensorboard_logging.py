"""Optional TensorBoard event output, independent of the learning algorithm."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from .q_learning import EpisodeSink, TrainingEpisode


def require_tensorboard() -> None:
    """Load the optional writer only when logging is requested."""
    try:
        import_module("tensorboard.summary.writer.event_file_writer")
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] != "tensorboard":
            raise
        raise ValueError(
            "TensorBoard logging requires the optional dependency. Install with "
            "`uv sync --locked --extra tensorboard`; run with "
            "`uv run --locked --extra tensorboard python -m "
            "environments.toy_text.frozen_lake train --tensorboard` "
            "(or use the benchmark subcommand)."
        ) from error


def reserve_run(log_dir: Path, command: str, run_name: str | None) -> Path:
    """Reserve a fresh invocation directory before any learning starts."""
    name = uuid4().hex if run_name is None else run_name
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", name) is None
        or name.endswith(".")
        or name.split(".")[0].upper() in reserved
    ):
        raise ValueError(
            "TensorBoard run name must be a portable filename component: "
            "1-100 ASCII letters, digits, dots, underscores or hyphens, starting "
            "with a letter or digit; no trailing dot or Windows reserved name"
        )
    require_tensorboard()
    root = log_dir / command / name
    root.mkdir(parents=True, exist_ok=False)
    return root


@contextmanager
def episode_writer(
    log_dir: Path, metadata: Mapping[str, object],
) -> Iterator[EpisodeSink]:
    """Write metadata and completed episodes; always drain and close the writer."""
    require_tensorboard()
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.plugins.text.summary_v2 import text_pb
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(log_dir))
    body_error: BaseException | None = None
    try:
        writer.add_event(Event(
            wall_time=time.time(), step=0,
            summary=text_pb("run/config", json.dumps(metadata, sort_keys=True, allow_nan=False)),
        ))

        def write_episode(row: TrainingEpisode) -> None:
            metrics = {
                "return": row["return"], "length": row["length"],
                "epsilon": row["epsilon"], "success": row["success"],
                "terminated": row["terminated"], "truncated": row["truncated"],
            }
            writer.add_event(Event(
                wall_time=time.time(), step=row["episode"],
                summary=Summary(value=[
                    Summary.Value(tag=f"episode/{tag}", simple_value=float(value))
                    for tag, value in metrics.items()
                ]),
            ))

        yield write_episode
    except BaseException as error:
        body_error = error
        raise
    finally:
        first_error = body_error
        for operation in ("flush", "close"):
            try:
                getattr(writer, operation)()
            except BaseException as cleanup_error:
                if first_error is None:
                    first_error = cleanup_error
                else:
                    first_error.add_note(
                        f"TensorBoard writer {operation} also failed: {cleanup_error!r}"
                    )
        if body_error is None and first_error is not None:
            raise first_error
