#!/usr/bin/env python3
"""
Evaluation Script - Test trained policy
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TODO: from rl.drone_env import DroneHoverEnv
# TODO: from agents.policy_runner import PolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    args = parser.parse_args()

    print(f"Evaluating policy: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")

    # TODO: Implement evaluation loop
    # policy = PolicyRunner(args.checkpoint)
    # env = DroneHoverEnv(render_mode='human' if args.render else None)
    #
    # for ep in range(args.episodes):
    #     state = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action = policy.get_action(state)
    #         state, reward, done, _ = env.step(action)
    #         total_reward += reward
    #     print(f"Episode {ep}: Reward = {total_reward:.2f}")

    print("\nTODO: Implement in src/agents/policy_runner.py")


if __name__ == "__main__":
    main()
