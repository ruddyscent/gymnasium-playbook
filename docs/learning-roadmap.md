# Learning roadmap

This roadmap suggests an order for learning reinforcement learning by implementing algorithms and applying them to Gymnasium and Arcade Learning Environment (ALE) environments. It balances implementation complexity, exploration difficulty, training stability, and computational cost.

This is an educational recommendation, not an official difficulty ranking, measured benchmark, delivery schedule, or list of completed implementations. Relative difficulty depends on the algorithm, environment configuration, and target performance. Nearby environments can be studied in a different order.

This document is the canonical public roadmap. Use repository issues for concrete implementation work and personal notes for learning journals and exploratory experiment notes.

## Scope

The roadmap has two tracks: Toy Text, Classic Control, Box2D, and MuJoCo for foundational and continuous control; and selected single-agent ALE/Atari games for visual discrete control. The ALE track is a representative curriculum, not an exhaustive ranking of every game, mode, difficulty, or multi-agent variant. Pin environment IDs, package versions, and configuration when implementing each experiment.

## Gymnasium sequence

| Step | Environment and configuration | Suggested algorithms | Learning focus |
| --- | --- | --- | --- |
| 1 | FrozenLake: 4×4, `is_slippery=False` | Tabular Q-learning | Action values, Bellman updates, epsilon-greedy exploration |
| 2 | CliffWalking | SARSA → Q-learning | On-policy versus off-policy learning |
| 3 | Blackjack | Monte Carlo control | Estimating values from episode returns |
| 4 | Taxi | Q-learning → Expected SARSA | Larger discrete state spaces and action masking |
| 5 | FrozenLake: `is_slippery=True` | Compare tabular algorithms | Stochastic transitions and sparse rewards |
| 6 | CartPole | DQN → REINFORCE → A2C | Neural value functions and policy gradients |
| 7 | Acrobot | DQN → Double DQN | Delayed outcomes and momentum-based control |
| 8 | MountainCar | DQN with exploration comparisons | Discovering successful trajectories |
| 9 | LunarLander: discrete actions | Double DQN → Dueling DQN → PPO | More complex control and training stability |
| 10 | Pendulum | DDPG → TD3 → SAC | Continuous actions, actor–critic methods, action bounds |
| 11 | MountainCarContinuous | TD3 / SAC | Continuous exploration and goal rewards |
| 12 | LunarLander: continuous actions | PPO / SAC | Coordinating multiple continuous control inputs |
| 13 | InvertedPendulum → InvertedDoublePendulum | PPO / SAC | MuJoCo basics and balance control |
| 14 | Reacher → Swimmer | PPO / SAC | Multi-joint control and algorithm sensitivity |
| 15 | HalfCheetah → Hopper → Walker2d | Compare SAC / PPO / TD3 | Locomotion, falls, observation normalization |
| 16 | Pusher → Ant | SAC / PPO | Object contact and more complex multi-joint control |
| 17 | BipedalWalker → Hardcore variant | PPO / SAC | Walking, obstacles, and longer-horizon behavior |
| 18 | CarRacing | CNN + PPO / SAC | Image observations, visual features, computational cost |
| 19 | Humanoid → HumanoidStandup | PPO / SAC | High-dimensional control and larger training budgets |

Arrows in the algorithm column suggest a progression of concepts. Slashes indicate alternatives to select or compare; implementing every algorithm is not a prerequisite for moving forward. Use continuous actions when applying SAC to CarRacing.

## Gymnasium core path

For a first pass through tabular learning, neural discrete control, and continuous control:

1. FrozenLake: deterministic transitions and tabular Q-learning.
2. CliffWalking: compare SARSA and Q-learning.
3. Taxi: extend tabular control.
4. CartPole: introduce DQN and policy gradients.
5. LunarLander: discrete actions with DQN variants and PPO.
6. Pendulum: continuous actions with TD3 and SAC.
7. HalfCheetah: multi-joint locomotion.
8. Walker2d: balance and locomotion.
9. Ant: more complex multi-joint control.

CartPole is also a reasonable starting point for someone already familiar with tabular reinforcement learning. FrozenLake is first here to include those foundations.

## ALE / Atari sequence

Start this track after validating DQN on CartPole and preferably discrete LunarLander. Completing the MuJoCo track or learning continuous-control algorithms is not a prerequisite. For the PPO alternative, first validate the implementation on CartPole.

The ordering below assumes image observations and discrete actions using `ALE/<Game>-v5`. Game names link to their official descriptions. RAM observations are an optional separate experiment; their smaller input size does not guarantee an easier task. All suggested DQN and PPO implementations in this track use a convolutional encoder.

### Recommended order

| Step | Environment ID and documentation | Suggested algorithms | Learning focus and reason for placement |
| --- | --- | --- | --- |
| A1 | [ALE/Pong-v5](https://ale.farama.org/environments/pong/) | CNN DQN; optionally CNN PPO | Establish image preprocessing, frame stacking, replay, and target networks with a visually simple paddle task |
| A2 | [ALE/Breakout-v5](https://ale.farama.org/environments/breakout/) | DQN → Double DQN | Ball motion, launching the ball, life handling, and longer-term brick-clearing behavior |
| A3 | [ALE/Boxing-v5](https://ale.farama.org/environments/boxing/) | Double DQN / PPO | Expand to movement-and-fire combinations with direct scoring feedback |
| A4 | [ALE/Freeway-v5](https://ale.farama.org/environments/freeway/) | Double DQN; compare exploration schedules | A small action set with delayed crossing rewards; simple controls do not imply easy exploration |
| A5 | [ALE/SpaceInvaders-v5](https://ale.farama.org/environments/space_invaders/) | Double DQN → Dueling DQN | Coordinate aiming, firing, and avoidance; compare value-network architectures |
| A6 | [ALE/Enduro-v5](https://ale.farama.org/environments/enduro/) | Dueling Double DQN / PPO | Sustained driving behavior and changing visual conditions |
| A7 | [ALE/Qbert-v5](https://ale.farama.org/environments/qbert/) | Add prioritized replay and n-step returns | Spatial progress and delayed consequences; study replay sampling and credit assignment |
| A8 | [ALE/Seaquest-v5](https://ale.farama.org/environments/seaquest/) | Incremental Rainbow / PPO baseline | Combine combat, diver rescue, and oxygen management; immediate scoring can compete with survival |
| A9 | [ALE/MsPacman-v5](https://ale.farama.org/environments/ms_pacman/) | Rainbow / PPO | Maze navigation, multiple moving threats, and longer-horizon action selection |
| A10 | [ALE/Frostbite-v5](https://ale.farama.org/environments/frostbite/) | Rainbow; optional PPO with RND comparison | Multi-step construction and survival objectives; distinguish exploration failures from control failures |
| A11 | [ALE/Gravitar-v5](https://ale.farama.org/environments/gravitar/) | Rainbow / PPO baseline; exploration experiments | Precise inertial control combined with challenging exploration |
| A12 | [ALE/MontezumaRevenge-v5](https://ale.farama.org/environments/montezuma_revenge/) | PPO + RND; optional hierarchical methods | Sparse rewards, keys, rooms, and temporally extended exploration |
| A13 | [ALE/PrivateEye-v5](https://ale.farama.org/environments/private_eye/) | Baseline → intrinsic motivation / hierarchical experiments | Long task sequences and exploration; treat as an advanced investigation |
| A14 | [ALE/Pitfall-v5](https://ale.farama.org/environments/pitfall/) | Baseline → advanced exploration research | Long-horizon treasure discovery with penalties; an optional research target rather than a routine graduation test |

This ordering is a curriculum proposal, not evidence that one listed algorithm will solve its assigned game. In particular, the last three games are not guaranteed to yield strong scores merely by adding intrinsic rewards or increasing the training budget. Define bounded milestones and report failures as well as successes.

### ALE core path

Pong → Breakout → SpaceInvaders → Qbert → Seaquest → MsPacman

Use these six games for a first pass through visual control and DQN improvements. Add MontezumaRevenge only when deliberately studying exploration. The remaining games provide comparisons and extensions rather than prerequisites for the core path.

### Algorithm implementation progression

1. Establish CNN DQN on Pong and Breakout with a shared, tested observation pipeline.
2. Add Double DQN and a dueling network separately, comparing each change on the same games.
3. Introduce prioritized experience replay and n-step returns one at a time.
4. Add categorical distributional value learning (C51) and noisy networks to complete the six components of [Rainbow](https://arxiv.org/abs/1710.02298). Document subsets as partial combinations rather than full Rainbow.
5. Use CNN PPO as an alternative baseline; it is not a required successor to Rainbow.
6. For exploration studies, compare PPO with and without [Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894). RND supplies a novelty-based intrinsic reward; it is not a standalone control algorithm.

Re-test changes on familiar games before moving down the curriculum. A game appearing later does not require a new algorithm, and the table does not replace ablation experiments.

### ALE experiment protocol

ALE results depend strongly on environment and wrapper settings. Use the [ALE specifications](https://ale.farama.org/env-spec/) and [Gymnasium AtariPreprocessing documentation](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing) when defining the pipeline.

- Record the game ID, ALE/Gymnasium versions, observation type, mode, difficulty, and minimal versus full action set. Use the default game mode and difficulty initially.
- Keep `repeat_action_probability=0.25` as the baseline sticky-action setting. Label deterministic debugging runs separately; historical `NoFrameskip-v4` scores are not directly interchangeable with the proposed protocol.
- Apply action repetition in one place. With `AtariPreprocessing(frame_skip=4)`, set the base environment's `frameskip=1` to avoid repeating actions twice. Sticky actions and frame skipping are different settings.
- A starting visual pipeline is max-pooling over the last two frames, 84×84 grayscale images, and a stack of four observations. `AtariPreprocessing` does not perform frame stacking; add it separately. Revisit grayscale if a task depends on color distinctions.
- Record random no-op starts and any game-specific FIRE/reset actions. Inspect action meanings rather than assuming every game uses the same indices or startup behavior.
- Make life-loss termination and reward clipping explicit training choices. Evaluate full games with original extrinsic rewards, without ending evaluation at each lost life; document time limits and evaluation exploration.
- Report emulator frames separately from agent decisions and optimizer updates. Compare algorithms at matched frame budgets across multiple seeds, with evaluation episode counts, score dispersion, wall time, and hardware recorded.
- Keep intrinsic rewards out of reported game scores. For difficult exploration games, supplement scores with clearly defined discovery milestones and state how those metrics are obtained.

These are proposed experiment settings, not results from training this repository's agents. Avoid comparing raw scores across different games as though they shared a common scale.

## Experiment guidelines

### Separate environment difficulty from implementation complexity

A small observation space does not guarantee easy learning. MountainCar can be difficult because successful trajectories are hard to discover. Conversely, InvertedPendulum can be approachable once a continuous-control implementation is available.

More complex environments do not always require more complex algorithms. Reuse PPO or SAC to investigate normalization, exploration, training budgets, and hyperparameters.

### Validate new algorithms on familiar environments

Validate PPO on CartPole and SAC on Pendulum before moving to harder environments. Changing the algorithm and environment together makes implementation errors harder to distinguish from task difficulty.

### Record comparable experiments

Record the environment ID, package versions, configuration, seeds, training steps, and evaluation procedure. Compare evaluation results and learning curves across multiple training seeds rather than reporting only the best run. Keep results with modified rewards or environment settings separate from default-environment results.

## Implementation tracking

Create issues as work becomes actionable, with bounded scope and explicit acceptance criteria. Link the corresponding implementation PRs and results from those issues.

- First implementation: [Tabular Q-learning for FrozenLake (#2)](https://github.com/ruddyscent/gymnasium-playbook/issues/2).

## References

The official documentation describes the environment families and their properties. The detailed ordering and algorithm assignments above are recommendations for this project's learning goals.

- [Toy Text](https://gymnasium.farama.org/environments/toy_text/): small discrete state and action spaces suitable for debugging reinforcement learning implementations.
- [Classic Control](https://gymnasium.farama.org/environments/classic_control/): CartPole, Acrobot, MountainCar, MountainCarContinuous, and Pendulum.
- [Box2D](https://gymnasium.farama.org/environments/box2d/): LunarLander, BipedalWalker, and CarRacing.
- [MuJoCo](https://gymnasium.farama.org/environments/mujoco/): continuous-control environments with multi-joint physics simulations.
- [Arcade Learning Environment](https://ale.farama.org/): Atari games exposed through the Gymnasium API; individual game documentation is linked in the ALE sequence.
- [ALE environment specifications](https://ale.farama.org/env-spec/): sticky actions, action sets, modes, and difficulties.
- [Gymnasium AtariPreprocessing](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing): image preprocessing and frame-skip configuration.
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298): the combined DQN extensions used in the suggested implementation progression.
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894): intrinsic motivation for exploration, including Atari experiments.
