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

## Watch the policy

```sh
uv run --locked python -m environments.toy_text.frozen_lake watch --q-table runs/frozen_lake/seed-0/q_table.npy
```

The window replays the greedy policy at two actions per second, pauses at the end, and repeats. Close the window or press Escape to exit. Use `--fps 4` to change playback speed, or replace `--q-table ...` with `--random` to watch random play. A desktop display is required; this viewer does not modify the Q-table.

## Tests

```sh
uv run --locked python -m unittest discover -s tests -v
```
