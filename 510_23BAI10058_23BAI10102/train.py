"""
train.py — Train a DQN agent on the FlappyBird environment.

Usage:
    python train.py
"""

import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from env import FlappyBirdEnv, NoisyObservationWrapper


def make_env(noisy=True):
    """Create and wrap the Flappy Bird environment."""
    env = FlappyBirdEnv()
    if noisy:
        env = NoisyObservationWrapper(env, noise_std=0.02)
    return env


def main():
    # Paths
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "flappy_dqn")

    # Environments
    train_env = make_env(noisy=True)
    eval_env = make_env(noisy=False)  # evaluate on clean observations

    # Evaluation callback — saves best model automatically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # DQN agent — tuned for better performance
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=5e-4,
        buffer_size=50_000,
        batch_size=32,
        gamma=0.99,
        learning_starts=2000,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=log_dir,
        verbose=1,
    )

    total_timesteps = 500_000

    print("=" * 60)
    print("  Starting DQN training on FlappyBirdEnv")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Model save path : {model_path}.zip")
    print(f"  TensorBoard log : {log_dir}")
    print("=" * 60)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    model.save(model_path)
    print(f"\nTraining complete. Model saved to {model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
