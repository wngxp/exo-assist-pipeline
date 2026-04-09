from myosuite.utils import gym
import deprl
from deprl import env_wrappers

env = gym.make("myoLegWalk-v0", reset_type="random")
env = env_wrappers.GymWrapper(env)

policy = deprl.load_baseline(env)

obs = env.reset()
episode_steps = 0

for i in range(300):
    action = policy(obs)
    result = env.step(action)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result

    episode_steps += 1
    if done:
        print(f"[BASELINE END] step={episode_steps} reward={reward}")
        break

env.close()
print("finished")