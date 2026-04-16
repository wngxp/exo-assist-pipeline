from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import myosuite  # noqa: F401
import numpy as np
from myosuite.utils import gym as myogym

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.baselines.load_deprl_reference import load_deprl_reference, wrap_deprl_env

ENV_ID = "myoLegWalk-v0"
FPS = 30
MAX_EPISODE_LENGTH = 1000
OUTPUT_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage1_reference.mp4")


def _normalize_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info

    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), info

    raise ValueError(f"Unexpected step result length: {len(result)}")


def _frame_to_uint8(frame):
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def run_episode(model, env, render_env, episode_idx):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    frames = []
    first_frame = render_env.render()
    if first_frame is not None:
        frames.append(_frame_to_uint8(first_frame))

    done = False
    steps = 0

    while not done and steps < MAX_EPISODE_LENGTH:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = _normalize_step(env.step(action))
        del reward, info
        steps += 1

        frame = render_env.render()
        if frame is not None:
            frames.append(_frame_to_uint8(frame))

    terminated_early = done
    print(
        f"[STAGE1 RENDER] episode={episode_idx} "
        f"length={steps} terminated_early={terminated_early}"
    )
    return frames, steps, terminated_early


def save_video(frames):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUTPUT_PATH, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved video to: {OUTPUT_PATH}")


def main():
    base_env = myogym.make(ENV_ID, reset_type="random", render_mode="rgb_array")
    env = wrap_deprl_env(base_env)

    try:
        model = load_deprl_reference(env)

        all_frames = []
        episode_lengths = []

        for episode_idx in range(1, 3):
            frames, steps, terminated_early = run_episode(
                model, env, base_env, episode_idx
            )
            all_frames.extend(frames)
            episode_lengths.append(steps)

            if episode_idx == 1 and terminated_early:
                print("First rollout terminated early; capturing a second episode.")
                continue
            break

        if not all_frames:
            raise RuntimeError("No frames were captured from env.render().")

        save_video(all_frames)
        print(f"Episode lengths: {episode_lengths}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
