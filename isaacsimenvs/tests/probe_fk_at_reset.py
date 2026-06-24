"""Probe: are robot body world poses fresh right after reset (pre-step)?

Resets, reads fingertip world positions BEFORE stepping, optionally forces an
articulation data refresh, then steps once and re-reads. If pre-step poses
match post-step poses, body_state_w is already valid at reset and the
hand-overlap resampler can read it directly; otherwise it's stale and we need
an explicit forward-kinematics refresh.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    ft_ids = inner._fingertip_body_ids

    def ft_pos():
        return inner.robot.data.body_state_w[:, ft_ids, 0:3].clone()

    env.reset()
    pre = ft_pos()

    # Try an explicit refresh without stepping physics.
    try:
        inner.robot.update(0.0)
        refreshed = ft_pos()
        print(f"[probe] robot.update(0.0) delta vs pre = "
              f"{(refreshed - pre).abs().max().item():.6f} m")
        pre = refreshed
    except Exception as e:  # noqa: BLE001
        print(f"[probe] robot.update(0.0) raised: {e}")

    zero = torch.zeros((args.num_envs, inner.cfg.action_space), device=inner.device)
    _, _, term, _, _ = env.step(zero)
    post = ft_pos()

    delta = (post - pre).abs()
    print(f"[probe] |post-step − pre-step| fingertip pos: "
          f"max={delta.max().item():.6f}  mean={delta.mean().item():.6f} m")
    print(f"[probe] terminated this step: {int(term.sum().item())}/{args.num_envs}")
    print("[probe] pre-step fingertip z (env 0):",
          [round(v, 4) for v in pre[0, :, 2].tolist()])
    print("[probe] post-step fingertip z (env 0):",
          [round(v, 4) for v in post[0, :, 2].tolist()])

    env.close()
    app.close()


if __name__ == "__main__":
    main()
