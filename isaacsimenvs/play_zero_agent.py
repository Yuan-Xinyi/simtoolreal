"""Visualize the SimToolReal task under a *zero agent* (all-zero actions).

No policy, no checkpoint — just resets the env (task initialization) and steps
zero-valued actions so you can see the initial scene: robot start pose, object
placement, and goal visualization.

Two ways to watch:

1. Live GUI (needs a display; this repo's server has DISPLAY set):
       .venv_isaacsim/bin/python isaacsimenvs/play_zero_agent.py \
           --num_envs 4 --num_assets_per_type 2 --steps 300

2. Headless -> mp4 (works anywhere; frames captured via the env's `viewer`
   camera, DirectRLEnv.render('rgb_array')):
       .venv_isaacsim/bin/python isaacsimenvs/play_zero_agent.py \
           --headless --video --num_envs 4 --num_assets_per_type 2 --steps 300

Output (video mode): isaacsimenvs/videos/simtoolreal_zero_agent.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent / "videos"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-agent visualizer for SimToolReal.")
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0", help="Gym task id")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300, help="Policy steps to run")
    parser.add_argument("--video", action="store_true", help="Capture frames and write an mp4 (implies cameras)")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output mp4 path (default: videos/simtoolreal_zero_agent.mp4)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Video capture renders through the offscreen viewer camera -> needs cameras.
    if args.video:
        args.enable_cameras = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=cfg, render_mode=render_mode)
    inner = env.unwrapped

    action_dim = cfg.action_space
    zero_action = torch.zeros((args.num_envs, action_dim), device=inner.device, dtype=torch.float32)

    obs, _ = env.reset()  # task initialization
    print(f"[zero_agent] reset OK — policy obs {obs['policy'].shape}", flush=True)

    frames: list = []
    for step_i in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if args.video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    if args.video:
        import imageio

        out_path = Path(args.out) if args.out else VIDEO_DIR / "simtoolreal_zero_agent.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_path), frames, fps=args.video_fps)
        print(f"[zero_agent] wrote {len(frames)} frames to {out_path}", flush=True)
    else:
        print(f"[zero_agent] ran {args.steps} zero-action steps", flush=True)

    env.close()
    # Kit teardown can hang; force-exit (matches the repo's other scripts).
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
