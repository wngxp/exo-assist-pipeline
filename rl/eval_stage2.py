import gym
import numpy as np
from stable_baselines3 import PPO

ENV_NAME = "myoLegWalk-v0"  # adjust if needed
MODEL_PATH = "rl_output/stage2_exo.zip"

env = gym.make(ENV_NAME)

model = PPO.load(MODEL_PATH)

obs = env.reset()

torques = []
rewards = []

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # store stuff
    torques.append(action)
    rewards.append(reward)

    if done:
        obs = env.reset()

torques = np.array(torques)
rewards = np.array(rewards)

np.savez("eval_data.npz", torques=torques, rewards=rewards)

print("Saved eval_data.npz")