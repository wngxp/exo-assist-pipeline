from stable_baselines3 import PPO


def load_reference_walker(path):
    return PPO.load(path, device="cpu")
