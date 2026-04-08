"""
evaluate.py — Evaluate a trained DQN agent on the FlappyBird environment.

Usage:
    python evaluate.py                  # headless evaluation (5 episodes)
    python evaluate.py --render         # with pygame rendering
    python evaluate.py --episodes 20    # custom number of episodes
"""

import argparse
import os

import numpy as np
from stable_baselines3 import DQN
from env import FlappyBirdEnv


def evaluate(model_path: str, episodes: int = 5, render: bool = False):
    """Load a trained model and run evaluation episodes."""
    render_mode = "human" if render else None
    env = FlappyBirdEnv(render_mode=render_mode)

    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Running {episodes} evaluation episode(s)...\n")

    all_rewards = []
    all_scores = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        score = info.get("score", 0)
        all_rewards.append(total_reward)
        all_scores.append(score)
        print(f"  Episode {ep:3d}  |  Score: {score:4d}  |  Total Reward: {total_reward:8.2f}")

    print("\n" + "=" * 55)
    print(f"  Mean Reward : {np.mean(all_rewards):8.2f}  ±  {np.std(all_rewards):.2f}")
    print(f"  Mean Score  : {np.mean(all_scores):8.2f}  ±  {np.std(all_scores):.2f}")
    print(f"  Max Score   : {int(np.max(all_scores))}")
    print("=" * 55)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Flappy Bird DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models", "flappy_dqn.zip"),
        help="Path to the trained model .zip file",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment with pygame")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: Model file not found at '{args.model}'")
        print("Train a model first with: python train.py")
        return

    evaluate(args.model, episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
