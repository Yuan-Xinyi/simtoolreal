"""Headless diagnostic: reset_object_clearance vs reset-time launch/bounce.

The "bouncing/launching" failure (issue #17) is a *penetration* artifact: when
the object spawns overlapping the table, PhysX resolves the overlap on the
first physics step with a large depenetration impulse, so the object — whose
velocity is zeroed at reset — suddenly acquires a big speed and flies off.

A clean signature: after exactly ONE policy step (dt = decimation/120 ≈ 1/60 s),
pure gravity gives only g*dt ≈ 0.16 m/s. Anything well above that (> ~0.5 m/s)
right after reset is a depenetration launch, not free fall. We sweep clearance
and report the fraction of envs whose object speed after step 1 exceeds a set
of thresholds, plus the eventual fall@N rate for context.

NOTE: eventual fall@N under ZERO actions has a floor that clearance cannot
remove — with no hand holding it, the object freely topples off the narrow
table. That floor is expected physics, not the launch bug.

    python isaacsimenvs/tests/sweep_reset_clearance.py \
      --num_envs 512 --rounds 4 --window 80
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_assets_per_type", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4, help="reset windows per clearance")
    parser.add_argument("--window", type=int, default=80, help="zero-action steps per window")
    parser.add_argument(
        "--clearances",
        type=str,
        default="none,0.005,0.02,0.05,0.1",
        help="comma list; 'none' disables the clamp (legacy behavior)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    n = args.num_envs
    device = inner.device
    action_dim = inner.cfg.action_space
    zero_action = torch.zeros((n, action_dim), device=device, dtype=torch.float32)

    speed_thresholds = [0.5, 1.0, 2.0]

    def parse_clearance(tok: str):
        return None if tok.strip().lower() == "none" else float(tok)

    clearances = [parse_clearance(t) for t in args.clearances.split(",")]

    print(f"\n[diag] num_envs={n}  rounds={args.rounds}  window={args.window}  "
          f"assets_per_type={args.num_assets_per_type}")
    print(f"[diag] table_top_offset={inner.cfg.reset.table_top_offset}  "
          f"table_object_z_offset={inner.cfg.reset.table_object_z_offset}")
    print(f"[diag] free-fall speed after 1 step ~ {9.81/60:.3f} m/s; "
          f"launch flagged at > {speed_thresholds[0]} m/s\n")

    avoid_modes = [False, True]

    rows = []
    for avoid in avoid_modes:
      for clearance in clearances:
        inner.cfg.reset.reset_object_clearance = clearance
        inner.cfg.reset.reset_avoid_hand = avoid
        total = args.rounds * n

        launch_counts = {t: 0 for t in speed_thresholds}
        peak_speed_sum = 0.0
        peak_speed_max = 0.0
        fall_seen_total = 0

        for _ in range(args.rounds):
            env.reset()
            # speed right after the first physics step = depenetration signature
            _, _, terminated, _, _ = env.step(zero_action)
            spd1 = torch.linalg.vector_norm(inner.object.data.root_lin_vel_w, dim=-1)
            for t in speed_thresholds:
                launch_counts[t] += int((spd1 > t).sum().item())
            peak_speed_sum += float(spd1.sum().item())
            peak_speed_max = max(peak_speed_max, float(spd1.max().item()))

            fall_seen = inner._termination_reasons["fall"] & terminated
            for _step in range(2, args.window + 1):
                _, _, terminated, _, _ = env.step(zero_action)
                fall_seen = fall_seen | (inner._termination_reasons["fall"] & terminated)
            fall_seen_total += int(fall_seen.sum().item())

        row = {
            "clearance": clearance,
            "avoid": avoid,
            "launch": {t: launch_counts[t] / total for t in speed_thresholds},
            "mean_spd1": peak_speed_sum / total,
            "max_spd1": peak_speed_max,
            "fall": fall_seen_total / total,
        }
        rows.append(row)

        label = "none" if clearance is None else f"{clearance:.3f}"
        print(
            f"[diag] avoid_hand={int(avoid)}  clearance={label:>6}  "
            f"launch>0.5={row['launch'][0.5]*100:5.1f}%  "
            f">1.0={row['launch'][1.0]*100:5.1f}%  "
            f">2.0={row['launch'][2.0]*100:5.1f}%  | "
            f"mean_spd1={row['mean_spd1']:.3f}  max_spd1={row['max_spd1']:5.2f} m/s  | "
            f"fall@{args.window}={row['fall']*100:5.1f}%"
        )

    print("\n[diag] === summary ===")
    print(f"{'avoid':>5}  {'clearance':>10}  {'launch>0.5':>10}  {'>1.0':>7}  {'>2.0':>7}  "
          f"{'max_spd1':>9}  {'fall@'+str(args.window):>8}")
    for row in rows:
        label = "none" if row["clearance"] is None else f"{row['clearance']:.3f}"
        print(
            f"{int(row['avoid']):>5}  {label:>10}  {row['launch'][0.5]*100:9.1f}%  "
            f"{row['launch'][1.0]*100:6.1f}%  {row['launch'][2.0]*100:6.1f}%  "
            f"{row['max_spd1']:8.2f}  {row['fall']*100:7.1f}%"
        )
    print()

    env.close()
    app.close()


if __name__ == "__main__":
    main()
