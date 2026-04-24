"""Render the DEP-RL stage-1 reference walker to video.

Purpose:
- load the local DEP-RL baseline
- capture an offscreen rollout
- save a reference MP4 for handoff/demo purposes

Inputs:
- local DEP-RL baseline under `rl/baselines_DEPRL/`

Outputs:
- `rl/rl_output/stage1/reference/stage1_reference.mp4`

Command:
- `python rl/stage1_compare/render_stage1_reference.py`

Status:
- partial
- useful for visualization, but render success still depends on the original
  MuJoCo/OpenGL setup
"""

from __future__ import annotations

import os
from pathlib import Path

# Headless offscreen rendering setup for remote Linux machines.
# Must be configured before MuJoCo / dm_control / MyoSuite imports.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import myosuite  # noqa: F401
import numpy as np
from myosuite.utils import gym as myogym

from rl.baselines.load_deprl_reference import load_deprl_reference, wrap_deprl_env

ENV_ID = "myoLegWalk-v0"
FPS = 30
MAX_EPISODE_LENGTH = 1000
MAX_RENDER_TRIES = 10
MIN_GOOD_EPISODE_LENGTH = 900
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 720
RL_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = RL_DIR / "rl_output" / "stage1" / "reference" / "stage1_reference.mp4"


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
    if np.issubdtype(frame.dtype, np.floating) and frame.size > 0 and np.nanmax(frame) <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def find_simulator(root_env):
    seen = set()
    stack = [root_env]

    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        sim = getattr(current, "sim", None)
        if sim is not None:
            model = getattr(sim, "model", None)
            data = getattr(sim, "data", None)
            if model is not None and data is not None:
                return sim, model, data

        for attr in ("env", "wrapped_env", "base_env", "unwrapped"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)

    raise RuntimeError(
        "Could not access MuJoCo simulator/model/data from the environment wrappers."
    )


def resolve_render_size(model):
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT

    vis = getattr(model, "vis", None)
    if vis is not None:
        global_vis = getattr(vis, "global_", None)
        if global_vis is not None:
            offwidth = int(getattr(global_vis, "offwidth", 0) or 0)
            offheight = int(getattr(global_vis, "offheight", 0) or 0)
            if offwidth > 0:
                width = offwidth
            if offheight > 0:
                height = offheight

    return width, height


def build_offscreen_renderer(sim, model, data, width, height):
    sim_render = getattr(sim, "render", None)
    if callable(sim_render):
        def render_frame():
            try:
                frame = sim_render(height=height, width=width, camera_id=0)
            except TypeError:
                try:
                    frame = sim_render(height=height, width=width)
                except TypeError:
                    try:
                        frame = sim_render(width=width, height=height, mode="offscreen")
                    except TypeError:
                        frame = sim_render(width, height)
            return _frame_to_uint8(frame)

        return render_frame

    try:
        import mujoco
    except ModuleNotFoundError:
        mujoco = None

    if mujoco is not None:
        try:
            raw_model = model
            for attr in ("_model", "model"):
                candidate = getattr(raw_model, attr, None)
                if candidate is not None:
                    raw_model = candidate

            raw_data = data
            for attr in ("_data", "data"):
                candidate = getattr(raw_data, attr, None)
                if candidate is not None:
                    raw_data = candidate

            renderer = mujoco.Renderer(raw_model, height=height, width=width)

            def render_frame():
                renderer.update_scene(raw_data)
                frame = renderer.render()
                return _frame_to_uint8(frame)

            return render_frame
        except Exception:
            pass

    try:
        import mujoco_py
    except ModuleNotFoundError:
        mujoco_py = None

    if mujoco_py is not None:
        context = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
        if hasattr(sim, "add_render_context"):
            sim.add_render_context(context)

        def render_frame():
            context.render(width, height)
            frame = context.read_pixels(width, height, depth=False)
            return _frame_to_uint8(np.flipud(frame))

        return render_frame

    raise RuntimeError(
        "Could not create a MuJoCo offscreen renderer. "
        "Tried simulator.render(), mujoco.Renderer, and mujoco_py offscreen context. "
        "On a headless Linux machine, make sure offscreen rendering is configured "
        "before imports (for example MUJOCO_GL=egl)."
    )


def run_episode(model, env, render_frame):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    frames = [render_frame()]
    done = False
    steps = 0

    while not done and steps < MAX_EPISODE_LENGTH:
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        obs, reward, done, info = _normalize_step(result)
        del reward, info
        steps += 1
        frames.append(render_frame())

    terminated_early = done and steps < MAX_EPISODE_LENGTH
    return frames, steps, terminated_early


def save_video(frames):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUTPUT_PATH, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved video to: {OUTPUT_PATH}")


def main():
    base_env = myogym.make(ENV_ID, reset_type="random")
    env = wrap_deprl_env(base_env)

    try:
        model = load_deprl_reference(env)
        sim, model_mj, data_mj = find_simulator(env)
        width, height = resolve_render_size(model_mj)
        render_frame = build_offscreen_renderer(sim, model_mj, data_mj, width, height)

        best_frames = None
        best_steps = -1
        best_terminated_early = True

        for trial in range(1, MAX_RENDER_TRIES + 1):
            frames, steps, terminated_early = run_episode(model, env, render_frame)
            print(
                f"[RENDER] trial={trial} steps={steps} terminated_early={terminated_early}"
            )

            if steps > best_steps:
                best_frames = frames
                best_steps = steps
                best_terminated_early = terminated_early

            if steps >= MIN_GOOD_EPISODE_LENGTH:
                print(
                    f"Using trial {trial} as render episode because it reached "
                    f"{steps} steps."
                )
                break

        if not best_frames:
            raise RuntimeError("No RGB frames were captured from the MuJoCo renderer.")

        print(f"Selected render episode length: {best_steps}")
        print(f"Selected render terminated early: {best_terminated_early}")
        save_video(best_frames)
    finally:
        env.close()


if __name__ == "__main__":
    main()
