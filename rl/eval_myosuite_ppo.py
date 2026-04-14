from myosuite.utils import gym as myogym
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = myogym.make("myoLegWalk-v0")
env = Monitor(env)

model = PPO.load("rl_output/walker_policy_1p5M", env=env, device="cpu")

num_episodes = 10
results = []

for ep in range(num_episodes):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        steps += 1

    results.append(steps)
    print(f"Episode {ep+1}: {steps} steps")

print("\nSummary:")
print(f"Mean: {sum(results)/len(results):.2f}")
print(f"Min: {min(results)}")
print(f"Max: {max(results)}")