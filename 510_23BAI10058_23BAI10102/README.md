# RL-Based Flappy Bird Autonomous Agent

## Team Members
- 23BAI10058 HARSH YADAV
- 23BAI10102 URMI BARMAN

A **Deep Q-Network (DQN)** agent trained with [stable-baselines3](https://stable-baselines3.readthedocs.io/) to autonomously navigate a Flappy Bird–like environment built on the [Gymnasium](https://gymnasium.farama.org/) API.

---

## Project Structure

```
├── env.py          # Custom Gymnasium environment & observation wrapper
├── train.py        # DQN training script
├── evaluate.py     # Model evaluation & rendering
├── models/         # Saved model checkpoints
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install gymnasium stable-baselines3 pygame numpy

# Train the agent (≈100 000 timesteps)
python train.py

# Evaluate (headless)
python evaluate.py

# Evaluate with rendering
python evaluate.py --render
```

---

## Drone / Autonomous Navigation Mapping

This project models a simplified autonomous navigation problem:

| Flappy Bird       | Autonomous Drone                   |
|--------------------|-------------------------------------|
| Bird               | Drone / UAV                        |
| Pipes              | Physical obstacles (buildings, trees)|
| Pipe gap           | Safe navigable corridor             |
| Flap action        | Thrust / altitude adjustment        |
| Gravity            | Gravitational pull on the vehicle   |
| Observation vector | Onboard sensor readings             |

The agent must make real-time, sequential decisions about when to apply thrust, mirroring how a drone controller decides when to adjust altitude to avoid obstacles.

---

## Reward Function Justification

| Signal        | Value   | Purpose                                                  |
|---------------|---------|----------------------------------------------------------|
| Survival      | +0.1    | Encourages staying alive; provides continuous gradient    |
| Pipe passed   | +1.0    | Rewards forward navigation progress                      |
| Collision     | −10.0   | Strong negative signal to deter crashing                 |

The large penalty-to-reward ratio ensures the agent strongly prioritises safety while still being incentivised to make forward progress.

---

## Observation Noise

`NoisyObservationWrapper` injects Gaussian noise (μ = 0, σ = 0.02) into every observation vector.

**Why?**
- Real-world sensors (GPS, LIDAR, cameras) are inherently noisy.
- Training under noise forces the agent to learn **robust** policies that don't overfit to perfect state information.
- This is a standard technique in **sim-to-real transfer** for robotics.

---

## Domain Randomization

At every episode `reset()`, the environment slightly randomises:

- **Gravity** — ± 0.05 around the base value
- **Pipe gap size** — ± 20 px around the base gap

**Why?**
- Prevents the agent from memorising a single fixed dynamics model.
- Improves **generalisation** to unseen conditions, analogous to varying wind speeds or obstacle sizes in real-world deployment.
- A well-established technique in reinforcement learning for robotics (Tobin et al., 2017).

---

## DQN Configuration

| Hyperparameter           | Value     |
|--------------------------|-----------|
| Policy                   | MlpPolicy |
| Learning rate            | 1 × 10⁻³  |
| Replay buffer size       | 50 000    |
| Batch size               | 32        |
| Discount factor (γ)      | 0.99      |
| Exploration fraction     | 0.2       |
| Final exploration ε      | 0.02      |
| Target network update    | every 500 steps |
| Training timesteps       | 100 000   |

---

## Optional Rendering

When `pygame` is installed, run evaluation with `--render` to visualise the agent in real-time at ~30 FPS. The display shows the bird, pipes, and current score.

---

## Dependencies

- Python 3.8+
- `gymnasium`
- `stable-baselines3`
- `numpy`
- `pygame` (optional, for rendering)
