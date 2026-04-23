from __future__ import annotations

import unittest
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_isaac.config.loader import load_project_config
from rl_isaac.tasks.registry import make_env


class HumanoidWalkEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "humanoid_walk.toml"
        self.config = load_project_config(config_path)

    def test_reset_and_step_produce_expected_fields(self) -> None:
        env = make_env(self.config)
        observation = env.reset(seed=123)

        self.assertIn("forward_velocity", observation)
        self.assertIn("target_velocity_error", observation)

        result = env.step([0.2] * self.config.task.action.dimension)
        self.assertIn("reward_breakdown", result.info)
        self.assertIn("applied_action", result.info)
        self.assertFalse(result.terminated)
        self.assertGreater(result.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
