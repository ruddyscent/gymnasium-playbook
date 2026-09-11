# FrozenLake: tabular Q-learning

Implementation for [issue #2](https://github.com/ruddyscent/gymnasium-playbook/issues/2), using `FrozenLake-v1`, `map_name="4x4"`, `is_slippery=False`, and a 100-step limit.

The detailed explanation and measured results are maintained separately in `posts/frozen-lake-q-learning.md` on the `blog` branch.

## Run the example

Run all commands from the repository root. They work without shell-specific activation or line continuation. Use the pinned Python and dependencies from the root README:

```sh
uv sync --locked
uv run --locked python -m environments.toy_text.frozen_lake --help
```

Train a table:

```sh
uv run --locked python -m environments.toy_text.frozen_lake train --seed 0
```

Evaluate the saved table, then a uniform random policy under the same evaluation settings:

```sh
uv run --locked python -m environments.toy_text.frozen_lake evaluate --q-table runs/frozen_lake/seed-0/q_table.npy --episodes 1000 --seed 10000
uv run --locked python -m environments.toy_text.frozen_lake evaluate --random --episodes 1000 --seed 10000
```

### Evaluate a published policy

Hugging Face Hub support is optional. Install the locked `hub` extra, then select
both a training seed and an immutable 40-character commit SHA. The commands use
the Hub cache; the first run downloads `seed-0/q_table.npy` and its matching
`seed-0/config.json`, and later runs can reuse the cached files.
`--hub-repo` must use the exact nonempty `owner/name` form.

```sh
uv sync --locked --extra hub
uv run --locked --extra hub python -m environments.toy_text.frozen_lake evaluate --hub-repo ruddyscent/gymnasium-playbook-frozenlake-q-learning --hub-revision b221cf51b99a9ba07cb9d3de65430acdb94162a7 --hub-seed 0 --episodes 1000 --seed 10000
uv run --locked --extra hub python -m environments.toy_text.frozen_lake watch --hub-repo ruddyscent/gymnasium-playbook-frozenlake-q-learning --hub-revision b221cf51b99a9ba07cb9d3de65430acdb94162a7 --hub-seed 0
```

Hub policies use the positive `training.max_episode_steps` stored in their
`config.json`; `watch` and `evaluate` apply it automatically. The loader rejects
missing or malformed metadata, non-finite or non-numeric tables, and any
environment other than deterministic 4x4 `FrozenLake-v1`. It also requires the
saved nonnegative `training.seed` to match `--hub-seed`, so the table and
configuration cannot be paired across training runs. Hub evaluation JSON includes
a `provenance` object with the repository ID, immutable revision, and selected
training seed. For a local
`--q-table`, the existing 100-step default remains available and
`--max-episode-steps` can select a different limit. `--q-table`, `--random`, and
`--hub-repo` are mutually exclusive.

Run the three-seed comparison:

```sh
uv run --locked python -m environments.toy_text.frozen_lake benchmark --seeds 0 1 2 --episodes 10000 --eval-episodes 1000 --eval-seed 10000
```

Both `train` and `benchmark` accept `--learning-rate`, `--discount`, `--epsilon-start`, `--epsilon-end`, `--epsilon-decay`, `--max-episode-steps`, and `--output-dir`. Run the relevant subcommand with `--help` for defaults. `evaluate` accepts `--output` to save its JSON report. If training uses a different episode limit, explicitly pass that same limit to standalone evaluation.

### Artifacts

By default, all generated files are under the ignored `runs/frozen_lake/` directory:

- `seed-0/q_table.npy`: the trained NumPy table, saved without pickle.
- `seed-0/training.csv`: episode number, epsilon, return, length, success, and ending flags.
- `seed-0/config.json`: environment settings, hyperparameters, seed, and software versions.
- `benchmark/seed-<seed>/`: the same training files plus `evaluation.json` for both policies.
- `benchmark/summary.csv` and `summary.json`: the comparison across seeds, with configurations and software versions in the JSON report.

Reusing an output directory overwrites that run's files. Use a different `--output-dir`, such as `runs/frozen_lake/slow-decay`, when comparing experiments. Keep custom output paths under `runs/` or another ignored artifact directory. The CSV files can later provide data for blog figures without adding a plotting dependency to the learning example.

## Optional TensorBoard logging

Logging is disabled by default. The optional TensorBoard dependency writes and displays events without installing TensorFlow or PyTorch:

```sh
uv sync --locked --extra tensorboard
uv run --locked --extra tensorboard python -m environments.toy_text.frozen_lake train --seed 0 --tensorboard --tensorboard-run-name first-run
uv run --locked --extra tensorboard python -m environments.toy_text.frozen_lake benchmark --seeds 0 1 2 --tensorboard
uv run --locked --extra tensorboard tensorboard --logdir runs/frozen_lake/tensorboard
```

Open the local URL printed by TensorBoard. Keep `--extra tensorboard` on subsequent `uv run` commands: uv can remove unselected optional dependencies. The viewer may report that TensorFlow is absent; scalar and text summaries work with this reduced feature set.

`--tensorboard-log-dir` changes the base event directory, which defaults to `runs/frozen_lake/tensorboard`. Each invocation reserves a fresh root before training:

```text
runs/frozen_lake/tensorboard/
  train/first-run/seed-0/events.out.tfevents.*
  benchmark/<generated-UUID>/seed-0/events.out.tfevents.*
                             seed-1/events.out.tfevents.*
                             seed-2/events.out.tfevents.*
```

Without `--tensorboard-run-name`, each invocation gets a new UUID; a benchmark shares that name across its seeds. Explicit names must be unused, contain 1-100 ASCII letters, digits, dots, underscores or hyphens, start with a letter or digit, and cannot end in a dot or use a Windows reserved filename. An existing run root is rejected before training, keeping repeated experiments separate. Both directory and name options require `--tensorboard`. Custom log paths should stay under an ignored directory such as `runs/`.

Every completed episode emits these scalar tags at the same **1-based episode number** used in `training.csv`:

| Tag | Value |
| --- | --- |
| `episode/return` | Sum of rewards in the episode |
| `episode/length` | Number of environment steps |
| `episode/epsilon` | Exploration probability used for that episode |
| `episode/success` | 1 if the goal was reached, otherwise 0 |
| `episode/terminated` | 1 if the underlying task ended, otherwise 0 |
| `episode/truncated` | 1 if the time limit ended the rollout, otherwise 0 |

Termination and truncation can both be true; only termination disables Q-learning bootstrapping. The Text dashboard's `run/config` summary at step 0 records the command, run name, environment, full training configuration, software/platform versions, and benchmark seeds/evaluation inputs where relevant. Conventional Q-table, CSV and JSON artifacts remain unchanged. Evaluation and policy playback do not write events. Writers flush and close when training finishes or raises an exception; incomplete episodes have no scalar record. TensorBoard stores scalar values as float32, so readback can differ from CSV values by floating-point rounding.

To inspect saved events without a browser:

```sh
uv run --locked --extra tensorboard tensorboard --inspect --logdir runs/frozen_lake/tensorboard
```

Regression tests read events with `tensorboard.backend.event_processing.event_accumulator.EventAccumulator`, using `Scalars(tag)` for episode records and `Tensors("run/config")` for JSON metadata. They verify emitted values and steps, independent seed/repeat streams, failure cleanup, and unchanged learning results. Headless event readback does not verify the browser UI or interactive graphics. The optional native fast data loader is not required on Windows or Apple Silicon; use `--load_fast=false` when explicitly selecting the portable loader.

## Watch the policy

```sh
uv run --locked python -m environments.toy_text.frozen_lake watch --q-table runs/frozen_lake/seed-0/q_table.npy
```

The window replays the greedy policy at two actions per second, pauses at the end, and repeats. Close the window or press Escape to exit. Use `--fps 4` to change playback speed, or replace `--q-table ...` with `--random` to watch random play. A desktop display is required; this viewer does not modify the Q-table.

## Tests

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked --extra tensorboard --extra hub python -m unittest discover -s tests -v
uv run --locked --group typecheck --extra tensorboard --extra hub python -m mypy environments/toy_text/frozen_lake tests
```

The first command checks the base installation, where event integration tests skip if TensorBoard is absent. The second executes those tests with the locked extra. The nondefault `typecheck` group supplies mypy for the FrozenLake package and tests; TensorBoard itself has no type declarations, so that external library boundary is not statically checked. CI runs these checks on Ubuntu x86-64, Windows x86-64 and Apple Silicon macOS.
