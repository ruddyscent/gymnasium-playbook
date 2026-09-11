"""Load and validate local or Hugging Face Hub FrozenLake policies."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .q_learning import QTable, validate_q_table


_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
_REPOSITORY_ID = re.compile(r"[^/\s]+/[^/\s]+")


@dataclass(frozen=True)
class LoadedPolicy:
    """A Q-table and the time limit needed to replay it faithfully."""

    q_table: QTable
    max_episode_steps: int


def load_local_policy(path: Path, max_episode_steps: int) -> LoadedPolicy:
    """Load a local Q-table while retaining the caller's episode limit."""
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    return LoadedPolicy(validate_q_table(np.load(path, allow_pickle=False)), max_episode_steps)


def hub_download(repo_id: str, filename: str, revision: str) -> Path:
    """Fetch one file through Hugging Face Hub's local cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ValueError(
            "Hugging Face Hub support requires `uv sync --locked --extra hub`"
        ) from error
    try:
        return Path(
            hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        )
    except Exception as error:
        raise OSError(
            f"Could not download {filename!r} from {repo_id!r} at {revision!r}"
        ) from error


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Model config must contain a {description} object")
    return value


def _max_episode_steps(config: object, expected_seed: int) -> int:
    metadata = _mapping(config, "top-level JSON")
    environment = _mapping(metadata.get("environment"), "environment")
    expected_environment = {
        "id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": False,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected or (
            key == "is_slippery" and environment.get(key) is not False
        ):
            raise ValueError(
                "Model config only supports FrozenLake-v1 with map_name=4x4 "
                "and is_slippery=false"
            )

    training = _mapping(metadata.get("training"), "training")
    training_seed = training.get("seed")
    if (
        not isinstance(training_seed, int)
        or isinstance(training_seed, bool)
        or training_seed < 0
    ):
        raise ValueError("Model config training.seed must be a nonnegative integer")
    if training_seed != expected_seed:
        raise ValueError(
            "Model config training.seed must match the requested Hub seed"
        )
    max_episode_steps = training.get("max_episode_steps")
    if (
        not isinstance(max_episode_steps, int)
        or isinstance(max_episode_steps, bool)
        or max_episode_steps < 1
    ):
        raise ValueError(
            "Model config training.max_episode_steps must be a positive integer"
        )
    return max_episode_steps


def load_hub_policy(repo_id: str, revision: str, seed: int) -> LoadedPolicy:
    """Load one immutable, seed-specific policy and its saved environment limit."""
    if not _REPOSITORY_ID.fullmatch(repo_id):
        raise ValueError("Hub repository must have the form 'owner/name'")
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("Hub revision must be a 40-character Git commit SHA")
    if seed < 0:
        raise ValueError("Hub seed must be nonnegative")

    artifact_dir = f"seed-{seed}"
    q_table_path = hub_download(repo_id, f"{artifact_dir}/q_table.npy", revision)
    config_path = hub_download(repo_id, f"{artifact_dir}/config.json", revision)
    try:
        q_table = validate_q_table(np.load(q_table_path, allow_pickle=False))
        config: object = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Model config.json must contain valid JSON") from error
    return LoadedPolicy(q_table, _max_episode_steps(config, seed))
