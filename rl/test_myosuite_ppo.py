from myosuite.utils import gym as myogym
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = myogym.make("myoLegWalk-v0")
env = Monitor(env)

model = PPO.load("rl_output/walker_policy", env=env, device="cpu")

model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)

model.save("rl_output/walker_policy_1p5M")
print("Saved walker to rl_output/walker_policy_1p5M.zip")

# ===== Step 2: Evaluate =====
reset_out = env.reset()
obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
episode_steps = 0

for i in range(2000):
    action, _ = model.predict(obs, deterministic=True)

    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result

    episode_steps += 1

    if i % 20 == 0:
        print(f"step={episode_steps}")

    if done:
        print(f"[EVAL END] step={episode_steps}, reward={reward}")
        print("info:", info)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_steps = 0