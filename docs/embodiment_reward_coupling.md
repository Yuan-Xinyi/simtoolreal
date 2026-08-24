# Why porting the hand changed the grasp strategy

Recorded 2026-08-17, during the iiwa14+Sharpa → xArm7+XHand port.

## Observation

With the reward function **unchanged** from the original SimToolReal task, the
xArm7+XHand policy learned to lift tools by **sliding the flat finger plane
under the object and cupping it against the palm**, rather than grasping from
above with the fingertips. The original iiwa14+Sharpa policy did not do this.

Nothing about the reward, the objects, the table or the task was modified —
only the robot.

## Root cause

**The reward specifies a goal, not a strategy. RL finds the cheapest strategy
inside the set of motions the embodiment can produce, so changing the
embodiment can move the argmax without changing the reward.**

The approach-phase term is `distance_delta_reward`: the sum, over all five
fingertips, of the reduction in distance to the object's **root frame**. It
has no notion of approach direction, of contact, or of opposition between
fingers. It is therefore *strategy-agnostic* — it scores whatever configuration
happens to bring five points near one point.

What that term's optimum looks like depends on the hand's kinematics — but see
the correction below before reading any mechanism into the difference.

> **Correction (2026-08-19).** An earlier version of this document explained the
> difference by saying Sharpa has six abduction joints and can spread its
> fingers while the XHand cannot. **That is factually wrong.** Counting joints
> in the URDF is not the same as reading their travel. In the asset actually
> used — `iiwa14_left_sharpa_adjusted_restricted.urdf`, and the file name means
> what it says — the four fingers' abduction joints are clamped to
> **±0.035 rad, i.e. 4° total**. Sharpa's fingers do not spread either. The
> premise of the original explanation does not exist.

Counting only joints with ≥30° of travel, the verified difference is not
lateral freedom but **curl depth and thumb dexterity**:

| | Sharpa | XHand |
|---|---|---|
| Usable flexion joints per finger | **3** — MCP 100° + PIP 100° + DIP 80° | **2** — 110° + 110° |
| Usable thumb joints | **4**, including MCP_AA 40° of opposition | 3 |
| Four-finger abduction | 4° (locked) | 0–22° (locked) |

A three-segment finger can curl into a hook that closes around a handle; a
two-segment finger can only pinch. Whether *that* is what decides the grasp
strategy is an open hypothesis — it has not been tested, and the previous
confident story in this space turned out to rest on an unchecked number.

Two secondary factors are real regardless: `lifting_reward` scores only the
object's rise in z, so a scoop lifts exactly as well as a pinch; and the
XHand's palm (190 × 94 × 47 mm) is a large flat surface — a good shovel.

The one thing that is certain is the negative result: **the reward asks for
nothing that distinguishes a fingertip grasp from a scoop.** A re-read of the
original Isaac Gym implementation (2026-08-19) confirms this is true of the
original too, not just of this port — every mechanism that might have
specified a grasp is either identical here or switched off there:

| mechanism | original setting |
|---|---|
| `object_lin/ang_vel_penalty` | scale `0.0` — disabled |
| `hand_delta_penalty` | multiplied by `0` in code — disabled |
| `finger_rew_coeffs` (per-finger weighting) | all ones, never modified |
| `withTableForceSensor` (press-the-table termination) | `False` |
| `resetWhenDropped` | `False` |
| contact forces / normals / opposition | absent from the reward entirely |
| fingertip and palm offsets, action pipeline, object reset | identical to this port |

So whatever makes the original's grasps acceptable, **it is not a reward term** —
there is no such term to inherit. That also means the premise "the original
grasps properly" deserves its own check: it may be a property of the Sharpa
morphology under this weak reward, or it may be an impression that has never
been measured.

## What this is not

The contact-physics defects fixed earlier in the port (convex-hull colliders
inflating the fingers 1.5–3.5×, missing armature, an over-damped wrist) are a
separate issue. Fixing them raised reward by roughly an order of magnitude and
made grasping work at all, but did **not** change which strategy is optimal, so
the scooping survived. Physics decides how well a strategy executes; the reward
plus the reachable motion set decides which strategy wins.

## Generalisation

Manipulation reward functions are almost never embodiment-neutral. They are
co-designed with a morphology, and the terms that "work" silently encode
assumptions about what is easy for that morphology. "Bring the fingertips close
to the object" specifies a goal, not a strategy; which strategy achieves it
most cheaply is decided by the hand. Port the reward to a new hand and the
assumption fails quietly: the reward curve still looks healthy, only the
behaviour changes.

A second, harder-won lesson from this document's own history: **a joint that
exists in the URDF is not a degree of freedom.** The explanation that stood
here for two days rested on counting `_AA` joints without reading their
`<limit>` — and every one of them was clamped to 4°. Read the travel, not the
tree.

This is why the sibling `xhand_inhand` repo's `pick_tool_token/grasp_signals.py`
constrains **contact topology** instead of proximity — `wrap_quality` requires
thumb contact opposed to the second-strongest non-thumb contact, pad-aligned,
with a palm-facing gate, and takes the `min` of all components so a strong
component cannot mask a missing one. Topology is morphology-explicit; proximity
is morphology-implicit, and only the former survives a hardware change.

## Fixing it — options considered

1. **Geometric gating (cheap, morphology-implicit).** Only credit a fingertip's
   approach while it is above the object's mid-height; penalise fingertips below
   the object while horizontally inside its footprint; multiply the 300-point
   lift bonus by a bounded [0,1] "approached from above" factor rather than
   hard-gating it. Uses only quantities the env already computes. Would need
   re-tuning again on the next hand.
2. **Contact topology (correct, costs contact sensors).** Port `wrap_quality`.
   Requires per-fingertip contact forces and normals; the SimToolReal env
   currently has no contact sensors.
3. **Termination (blunt).** End the episode when a lift occurs with a fingertip
   below the object. Effective but risks the policy simply never lifting.

Prefer multiplicative bounded factors over hard gates, for the same reason
`wrap_quality` uses `min` over components: a hard gate that is slightly too
strict silently kills learning.

## Controlled experiment

To confirm the morphology (not the arm) is responsible, we build **xArm7 +
Sharpa** — same arm, same reward, same objects, only the hand swapped back —
and observe whether the scooping disappears. Result recorded below once the run
has trained.

## Verification status — the causal claim above is NOT yet confirmed

Everything above the "Controlled experiment" heading was reasoned from joint
counts and the form of the reward, **before** the behaviour was measured. A
first quantitative probe does not reproduce it:

Sampling the eight most recent `interactive_viewer` snapshots of each run and
locating the frame where the object first rises 5 cm off the table, then asking
how many fingertips are above the object's centre at that instant:

| run | stage | fingertips above object centre | median dz |
|---|---|---|---|
| xArm7+XHand | epoch 34k, best 3135 | 34/40 (85%) | +0.019 m |
| xArm7+Sharpa | epoch 6k, best ~250 | 19/40 (48%) | −0.016 m |

That is the opposite of the prediction. Two caveats keep it from being
decisive, and both must be resolved before either conclusion is drawn:

1. **The runs are at incomparable stages** — a mature policy against one that
   has only just learned to lift at all.
2. **The probe is weak, and demonstrably has false positives.** Rendering the
   hand at the detected lift frame shows at least one episode where the hand is
   nowhere near the object, i.e. the "lift" was not a grasp. And "fingertip
   above the object's centre" is in any case a poor proxy for "not scooped" —
   an object can ride on the finger undersides with the tips still above its
   centre.

So the mechanism described above remains a *hypothesis with an unverified
premise*: it explains a behaviour that has been observed by eye but not yet
characterised quantitatively. Do not cite it as a finding until the probe is
sound and the two runs are compared at matched maturity.

## Correction (2026-08-24) — two geometric findings retracted

Both of the geometry results this document was built on turn out to be
measurement errors, not properties of the hardware. Recording them here because
each one cost a training run.

### The mount roll was never a free parameter

The 15°-vs-195° `sharpa_mount` experiments could not have tested what they
claimed to. In the merged URDF, `link8_joint` is the identity transform and the
`sharpa_mount` origin is a pure rotation about **z** — which is exactly
`joint7`'s axis. Any mount roll is therefore absorbed by `joint7`, and `joint7`
has ±180° of travel. Rolling the mount only shifts `joint7`'s zero.

What the experiments actually varied was the **home pose**: the roll was changed
while the joint values were held fixed, so the hand moved. The observed effects
were real, but they were effects of the home pose, and attributing them to the
mount sent the next two runs after the wrong variable.

*A fixed joint between two links is only a design parameter if it is not
collinear with an actuated axis next to it. Check the axis before ablating it.*

### The opposition measurements used an object ~4× too large

The opposition criterion was evaluated against a box with half-extents
`(0.14, 0.03, 0.025)`. The object's actual half-extents are
`cfg.fixed_size / 2 = (0.071, 0.015, 0.014)` — the earlier numbers were full
extents used as half extents, giving a box roughly 4× too big by volume. Almost
any fingertip "touched" it, so the reported angles (reference 151°, the 195°
build 52.5°) describe nothing real. A second bug compounded it: fingertips
*inside* the box fell back to a constant approach direction, which pinned some
later measurements to exactly 90.0°.

With the real object and table geometry (table top 0.53, footprint 0.475 × 0.4)
and a grasp station calibrated **on the reference itself** rather than invented,
the result reverses:

| configuration | thumb touches | finger touches | best opposition |
|---|---|---|---|
| reference iiwa14 + Sharpa | 4901 | 3477 | **180.0°** |
| xArm7 + Sharpa, solved home | 4200 | 3885 | **180.0°** |

The xArm7 is not geometrically handicapped for a thumb-opposed pinch. It has the
same opposition capability as the reference.

The calibration also shows the grasp station must sit **behind** the object
(+3…+6 cm in y): at y ≤ 0 the fingers register zero touches at every height,
because the hand approaches from +y and the fingers extend in −y.

### What actually differed

The port's home pose was solved to match the original's **flange**. That is the
wrong link: the original's hand hangs off `iiwa14_link_ee`, 4.5 cm beyond
`iiwa14_link_7`, so equal flange poses leave the two hands 4.5 cm apart. Solving
against `left_hand_C_MC` — the same link on both robots — reproduces the
reference hand pose to 0.00 cm, with x/z axis alignment 1.0000 / 0.9947 and the
same +17.8 cm clearance over the table.

*Match the link whose geometry you care about, not the one that is convenient to
compute.*

### Known residual: 5% of goals are unreachable

Sampling the goal volume uniformly, 114/120 targets are reachable (median miss
0.00 cm); the misses are confined to the far-top corners, up to 13 cm short.
This is the xArm7's shorter reach (1.039 m vs the KUKA's 1.261 m) and is
**pre-existing**. The goal volume is deliberately left unchanged so the run stays
comparable to the KUKA baseline curve.
