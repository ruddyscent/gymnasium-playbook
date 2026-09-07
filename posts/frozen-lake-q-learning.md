# FrozenLake: tabular Q-learning

Draft explanation for [issue #2](https://github.com/ruddyscent/gymnasium-playbook/issues/2), maintained on the `blog` branch. The implementation was merged into `main` through [PR #4](https://github.com/ruddyscent/gymnasium-playbook/pull/4). This draft describes [commit `35e65db`](https://github.com/ruddyscent/gymnasium-playbook/tree/35e65dbb3da53574df4fe4ae85a230a9b5aaf3fa), with the code under `environments/toy_text/frozen_lake/`. Run the commands below from a `main` checkout containing that commit, or a separate checkout of the exact commit for reproduction. The `blog` worktree holds the explanation and does not contain the implementation or uv setup described here.

Start with `q_learning.py`, where the table, action selection, Bellman update, and episode loop are explicit. `__main__.py` contains the command-line, benchmark, and artifact-handling code. `viewer.py` replays a fixed policy in a window without training updates. There is no reinforcement learning algorithm library or neural network.

## The task

Use `FrozenLake-v1`, `map_name="4x4"`, `is_slippery=False`, the default rewards, and a 100-step episode limit. This fixes the map and removes slippery transitions:

```text
S F F F       0  1  2  3
F H F H       4  5  6  7
F F F H       8  9 10 11
H F F G      12 13 14 15
```

`S` is the start, `F` is safe ice, `H` is a hole, and `G` is the goal. The observation is the cell number. Actions are `0=LEFT`, `1=DOWN`, `2=RIGHT`, and `3=UP`. Reaching the goal gives reward 1; all other transitions give 0. Both a hole and the goal terminate an episode. These details come from the [official FrozenLake documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

The learner starts with a zero-filled table of shape `(16, 4)` and learns only from `reset()` and `step()`. It does not inspect the transition model or use a hand-written path. The known route in the tests is an independent fixture for checking evaluation, not a training input.

## How the learning works

Read the implementation in this order:

1. `TrainingConfig`: the hyperparameters and their allowed ranges.
2. `epsilon_greedy`: choose between exploration and exploitation.
3. `q_learning_target` and `update_q_value`: compute and apply one Bellman update.
4. `train`: repeat interaction and updates, recording each episode.
5. `evaluate`: run a fixed policy in a fresh environment, without updates.

For each observed transition, the update is:

```text
target = reward                                      if terminated
target = reward + discount * max(Q[next_state])       otherwise
Q[state, action] += learning_rate * (target - Q[state, action])
```

The next-state maximum makes this Q-learning: it learns toward a greedy policy even while its behavior still explores. A successful goal transition first raises the value of the action just before the goal; subsequent visits propagate that value toward the start.

For example, if the current estimate is 2, reward is 1, next-state maximum is 4, discount is 0.9, and learning rate is 0.25, the target is 4.6 and the updated estimate is 2.65. This arithmetic example is covered by a regression test.

### Exploration and ties

During training, choose a uniformly random action with probability epsilon. Otherwise, choose uniformly among actions tied for the maximum value. Random tie-breaking matters at initialization: plain `argmax` would always select action 0 (LEFT) from the all-zero table.

Epsilon follows `max(epsilon_end, epsilon_start * epsilon_decay**episode)`, where `episode` starts at zero. With the defaults it declines from 1.0 to a floor of 0.05. Training therefore keeps taking exploratory actions even after it learns a successful route.

Evaluation uses `argmax` with a fixed tie rule and no epsilon exploration. An untrained all-zero table repeatedly chooses LEFT and times out. This makes a useful sanity check before learning.

### Termination is different from truncation

End a rollout on `terminated or truncated`, but disable bootstrapping only for `terminated`. A time limit cuts off data collection without making the next state's value zero. In this example the learner models the underlying task without treating the 100-step wrapper limit as part of the state. If a hole or goal is reached on the final allowed step, both flags can be true: termination takes precedence in the target. See [Gymnasium's time-limit guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

### Types make the data flow explicit

`QTable` is an alias for `NDArray[np.float64]`: each entry is a floating-point action-value estimate. The annotation documents the element type; it does not enforce the `(16, 4)` shape. Loaded tables therefore also need runtime shape and finite-value checks.

`TrainingEpisode` and `EvaluationResult` are `TypedDict` definitions for the records written to CSV and JSON. They make fields and their types visible to readers and static checking while retaining ordinary dictionaries at runtime. `TrainingEpisode` uses the functional `TypedDict` syntax because its `return` field is a Python keyword. `mean_successful_episode_length` is `float | None`, since there may be no successful episodes to average.

Typed function arguments and return values help distinguish an observation, an action, a Q-table, and an evaluation report. They complement runtime validation and tests. After annotations were added, all three benchmark seeds produced identical Q-tables, training CSV files, and evaluation results compared with the original implementation.

## Run the example

Run all commands from the implementation repository root described above, rather than from the `blog` worktree. They work without shell-specific activation or line continuation. Use the pinned Python and dependencies from the root README:

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

Reproduce the three-seed comparison below:

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

## Watch the learned policy

After training seed 0, replay its greedy policy:

```sh
uv run --locked python -m environments.toy_text.frozen_lake watch --q-table runs/frozen_lake/seed-0/q_table.npy
```

Compare it with uniformly random play:

```sh
uv run --locked python -m environments.toy_text.frozen_lake watch --random --fps 4
```

The viewer uses the same deterministic 4×4 map and a 100-step limit. It reads the Q-table without updating it, advances at two actions per second by default, pauses for two seconds when an episode ends, and repeats. `--fps` controls actions per second (1–30); use the same value for both policies when comparing their behavior. Close the window or press Escape to exit. A desktop display is required. The viewer uses Pygame, already included through the Toy Text dependencies.

Follow the cell numbers in the route below while watching the learned policy. Random play can end quickly by falling into a hole, so use the reported success rates alongside the animation.

## Observed results

Measured on 2026-09-07 with Python 3.13.15, Gymnasium 1.3.0, and NumPy 2.5.3 on macOS ARM64. Each seed used 10,000 training episodes, learning rate 0.1, discount 0.99, epsilon 1.0 → 0.05 with decay 0.999, and a 100-step limit. Each policy was evaluated for 1,000 episodes; evaluation seed is `10000 + training_seed`. Random-policy runs use the corresponding evaluation seed, not a learned table.

| Training seed | Evaluation seed | Q-learning success | Q-learning mean steps | Random success | Random mean steps |
| --- | --- | --- | --- | --- | --- |
| 0 | 10000 | 1000/1000 (100%) | 6.000 | 15/1000 (1.5%) | 7.556 |
| 1 | 10001 | 1000/1000 (100%) | 6.000 | 12/1000 (1.2%) | 7.857 |
| 2 | 10002 | 1000/1000 (100%) | 6.000 | 11/1000 (1.1%) | 7.642 |

Mean steps includes failures as well as successes. Short episodes can mean falling into a hole, so episode length alone is not a performance measure. The raw JSON/CSV also records mean length on successful episodes and truncation counts.

For seed 0, greedy evaluation followed `0 → 4 → 8 → 9 → 13 → 14 → 15` (DOWN, DOWN, RIGHT, DOWN, RIGHT, RIGHT). Six steps matches the Manhattan-distance lower bound between opposite corners, so this is a shortest successful route on this fixed map. The learned start-state value for DOWN was approximately 0.95099005, matching `0.99**5`: the terminal reward arrives on the sixth action.

The first successful training episodes were 38, 31, and 30 for seeds 0, 1, and 2. Success counts in the final 100 training episodes were 91, 97, and 93. These are exploratory training results and should not be confused with the 100% greedy evaluation scores.

The evaluation map and start state are fixed and transitions are deterministic, so a greedy policy repeats the same trajectory. A thousand successful evaluations here does not demonstrate generalization to unseen maps or slippery dynamics. The three training seeds test sensitivity to exploration in this configuration; they are not evidence that every seed or hyperparameter choice will succeed.

## Verification and material for a blog post

```sh
uv run --locked python -m unittest discover -s tests -v
```

All 14 regression tests passed locally, including a repeat run after merge. Tests check target arithmetic, holes and goals, real time-limit transitions, simultaneous termination and truncation, action selection, reproducible learning, read-only evaluation, and reporting of success and episode length. The measured benchmark above is separate from the regression suite.

[PR #4 CI](https://github.com/ruddyscent/gymnasium-playbook/actions/runs/34082553918) passed on Ubuntu 24.04, Windows 2022, and macOS 14 (Apple Silicon), using the shared uv lockfile and pinned Python version. Each platform ran base-environment smoke checks, the regression suite, and smoke checks with all optional environment families installed. The three-seed performance measurements above were made on macOS ARM64; this CI result does not establish full benchmark results or graphical replay behavior on every platform.

Static type checking passed for the seven Python files present when annotations were added. The subsequently added viewer was outside that check. Its initialization and close path and the CLI were checked locally; full visual behavior across all three platforms remains unverified.

The [issue's acceptance verification](https://github.com/ruddyscent/gymnasium-playbook/issues/2) records evidence for all five completion criteria, and its [implementation insights](https://github.com/ruddyscent/gymnasium-playbook/issues/2#issuecomment-5564941383) include the detailed measurements. The earlier comment's pre-merge and pending-CI notes describe the state at the time of that comment; the merged implementation and CI status above supersede them.

A future post can follow the same sequence as the code: describe the map and rewards, explain the zero table and epsilon-greedy policy, trace a single update, distinguish the two ending flags, and then compare training curves with greedy and random evaluation. Use the saved CSV data and actual trajectories for figures. Discuss the fixed-map limitation before extending to slippery transitions. This draft supplies reproducible notes; it is not a claim that the later blog post has been written or published.
