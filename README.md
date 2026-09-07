# gymnasium-playbook
A practical playbook for solving Gymnasium and Arcade Learning Environment (ALE) environments with reinforcement learning.

## Environment setup

Use **uv 0.12.10** and **CPython 3.13.15** on each machine. The Python patch version is pinned in `.python-version`; `pyproject.toml` keeps resolution within Python 3.13, and `uv.lock` records exact dependency versions across platforms. Keep these files in version control and use `--locked` to prevent implicit dependency updates.

Install uv using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). For the pinned release:

macOS / Ubuntu:

```sh
curl -LsSf https://astral.sh/uv/0.12.10/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://astral.sh/uv/0.12.10/install.ps1 | iex
```

From the repository root, the following commands are identical on all three operating systems:

```sh
uv python install
uv sync --locked
uv run --locked python scripts/check_environments.py
```

This creates a local `.venv` with Gymnasium 1.3.0, NumPy 2.5.3, and rendering dependencies for Toy Text and Classic Control. No virtual-environment activation is necessary. The checks reset and step environments and render RGB arrays without opening a window; they do not train an agent.

### Optional environment families

| Family | Installation | Smoke check |
| --- | --- | --- |
| Box2D | `uv sync --locked --extra box2d` | `uv run --locked --extra box2d python scripts/check_environments.py --box2d` |
| MuJoCo | `uv sync --locked --extra mujoco` | `uv run --locked --extra mujoco python scripts/check_environments.py --mujoco` |
| ALE / Atari | `uv sync --locked --extra atari` | `uv run --locked --extra atari python scripts/check_environments.py --atari` |
| All families | `uv sync --locked --all-extras` | `uv run --locked --all-extras python scripts/check_environments.py --all` |

Use the same extra flags for subsequent `uv run` commands: uv synchronizes the environment and can remove unselected optional packages. ALE is pinned to 0.12.1 and MuJoCo to 3.12.0. Use the named extras rather than Gymnasium's broad `all` extra. PyTorch and GPU-specific dependencies can be added when implementing neural agents; they are not needed for these environment checks.

### Platform support

The full dependency set targets macOS 13+ on Apple Silicon, Ubuntu 22.04+ on x86-64, and Windows 10/11 on x86-64. The same Python and dependency versions are used, with platform-specific wheels selected by uv. Intel Macs and Windows/Linux ARM are not full-stack targets: current ALE/MuJoCo or Box2D wheel availability differs. Basic Toy Text and Classic Control may still work there.

The CI matrix checks Ubuntu, Windows, and Apple Silicon macOS using the lockfile and binary wheels only. A successful lock or cross-platform resolution does not itself prove runtime compatibility; CI must run on each OS after changes are pushed.

Box2D has Python 3.13 wheels for the target platforms, avoiding a local C++ build. MuJoCo smoke checks exercise physics only: GUI/offscreen rendering additionally depends on system graphics drivers and an appropriate OpenGL backend. On headless Linux, configure EGL/OSMesa and its system libraries before rendering MuJoCo. For SDL rendering on a headless host, set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` (the CI workflow sets both).

Do not create a local `gymnasium/` Python package: it would shadow the installed library. Keep environment experiments under a distinct project directory name.

## Learning path

The first worked example is [FrozenLake: tabular Q-learning](environments/toy_text/frozen_lake/README.md), with runnable commands, regression tests, and a three-seed benchmark against random play. The detailed explanation is maintained on the `blog` branch in `posts/frozen-lake-q-learning.md`.

Build reinforcement learning algorithms progressively, starting with tabular methods and moving to neural networks and continuous control:

FrozenLake → CliffWalking → Taxi → CartPole → LunarLander (discrete) → Pendulum → HalfCheetah → Walker2d → Ant

See the [learning roadmap](docs/learning-roadmap.md) for the full sequence, suggested algorithms, and experimental guidelines. This is a recommended study order, not a delivery schedule or a list of completed implementations.

The repository roadmap is the canonical public guide. Track concrete implementation work in [Issues](https://github.com/ruddyscent/gymnasium-playbook/issues), starting with [tabular Q-learning for FrozenLake](https://github.com/ruddyscent/gymnasium-playbook/issues/2).
