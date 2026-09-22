# Footrack — Project Plan

Real-time tactical positioning system for football training: computes ideal player
positions from live game state and displays them to coaches and players, ultimately
projected onto the pitch surface.

Status: planning. Last updated 2026-09-21.

---

## 1. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Output, v1 | Desktop analyst tool over replay | No hardware at all — see §1.1 |
| Output, v2 | Live screen / tablet, then projection on grass | Swappable renderer backend, not a rewrite |
| Interaction, v1 | Replay scrub + drag-to-override | Live continuous overlay is v2 |
| Tracking, dev | Replay from public datasets | No hardware dependency to start |
| Tracking, live | Source interface: LPS / CV / replay all interchangeable | De-risks the vest access unknown |
| Model | Layered stack L1–L5, learned model as one voter | Explainability is the only validation channel |
| Training data | Public tracking datasets | Free, immediate, sufficient for L1–L4b |
| Target club | Senior amateur / semi-pro | Drives the capability-constraint requirement (§3.3) |
| Effort | ~20h/week, sustained | Roadmap sliced into independently demoable wins |

### 1.1 v1 product definition

**Primary user: the analyst (you). Surface: a desktop tool over replay data.**

v1 ships with **no hardware**. No vests, no cameras, no live pipeline, no latency
budget, no projection. Those are v2.

**v1 features:**

1. **Ideal position marks with deviation-only display.** Marks appear only for players
   meaningfully out of position. Less visual noise than 11 permanent dots, and it is
   self-teaching — no mark means you are right.
2. **Manual override with re-solve.** Drag one player; the other ten rearrange
   coherently around that constraint.

**Backlog (v2+):** freeze-frame on whistle, predicted-futures display, auto-selected
teachable moments, style counterfactuals, cue firing log, coach freehand drawing,
heatmaps, live pipeline, projection. Full feature analysis in FEATURES.md.

### 1.2 Commercial track

Goal is to build a company. That does not change the technical path, but it changes what
the early phases are *for*.

**The risk to manage: seven months with nothing in front of a coach.** Path A's weakness
is not technical, it is that you can build the whole thing on an untested assumption
about what coaches want. The fix is already on the roadmap and costs nothing extra —
**the analyst tool is showable at week 9.** Show it then.

**Recruit 3-5 external coaches as raters in Phase 2.** This is the single highest-value
change to the plan. It is simultaneously:

- better training data (multiple raters beats a sample size of one)
- customer discovery (every rating session is a user interview)
- distribution (raters become pilot users, and later references)

Frame it to them as helping train the model, which is true, and costs them 20 minutes.

**The moat is the rated corpus, not the architecture.** Any competent ML team can build
a transformer over tracking data — the datasets are public. **Nobody else has coach
judgement labelled against positioning.** That data is expensive to collect, compounds
over time, and is the actual defensible asset. Treat it as one: version it, back it up,
track its provenance.

**Consider that the analyst tool may be the business, not the stepping stone.** A desktop
match-analysis product that says "here is where your shape broke down, here is the drill"
ships over the internet, scales to any number of clubs, and needs no hardware.
Projection installs one club at a time and is capital-intensive. There is a real chance
the unglamorous version is the better company, with projection as the thing that gets you
written about rather than the thing that pays. Do not abandon projection — but do not
assume it is the product.

**Now live, previously deferred:**

- **GDPR.** Player positional data is personal data. A commercial product needs a lawful
  basis, a data processing agreement with each club, and a retention policy. Not urgent
  this month; no longer deferrable indefinitely.
- **Pricing signal.** Ask every coach you recruit what they currently pay for analysis
  tools, and what they would pay for this. A cheap question with disproportionate
  information.
- **Who is the buyer?** Clubs, academies, or federations. Different products.

**Two consequences worth stating explicitly:**

- **The analyst tool and the coach-rating tool are the same artifact.** Scrub to a
  frame, see proposed marks, drag one to where it should be — every correction is a
  training label. Data collection is a side effect of using the product, not separate
  work. This merges the old Phase 2 into Phase 1.

- **Manual override with re-solve is the masked-agent head at inference time.** Pin the
  dragged player as observed, treat the others as unknown, infill. MODEL.md task 2 is
  therefore not merely robustness against vest dropout — it is the engine of the
  headline v1 feature, and its priority rises accordingly.

---

## 2. Architecture

Seven layers. Each is independently shippable. Every layer above L1 degrades
gracefully to the one below it, so the system always has an answer.

```
  L6  Renderer            screen now -> projector backend later
        ^
  L5  Arbiter             blends L1-L4, attaches an explanation to every mark
        ^                 also: forecasts to t+latency  (see 3.1)
  L4  Learned policy      GNN/transformer over the 23-entity set
  L3  Objective optimizer pitch control, lane coverage, xT-gain
  L2  Constraint solver   coach principles as hard/soft constraints
  L1  Formation template  coach-authored reference + deformation rules
        ^
  L0  State estimation    tracking -> pitch-frame state + phase classifier
```

### L0 — State estimation

Normalizes any tracking source into one canonical state:
22 player positions + velocities, ball position + velocity, phase, attack direction.

- **Attack-direction normalization.** Always represent as attacking left-to-right.
  Halves the effective problem and is required for sane data augmentation.
- **Phase classifier.** Because we handle all phases, we must know which one we are in.
  Self-contained and tractable: in-possession / out-of-possession / transition /
  set piece. Start with rules on ball ownership plus a hysteresis window; upgrade to a
  small sequence model later. Do not skip the hysteresis — naive frame-wise
  classification flickers badly at turnovers.
- **Source interface.** One abstract class, four implementations: replay-from-file,
  LPS-live, LPS-post-hoc, camera-CV. Write this on day one.

### L1 — Formation template

Coach defines a shape (4-3-3, 5-2-3, etc.) as reference positions in normalized
pitch coords, plus deformation rules: shift by ball x/y, compress by line height,
expand by width principle. Pure geometry, no learning.

This layer is the floor. It is never *wrong*, only unsubtle — and it is what the
system falls back to when everything above it is uncertain or unavailable.

### L2 — Constraint solver

Coach principles expressed as constraints:

- vertical compactness <= 25m
- defensive line at or above halfway
- ball-side winger tucks inside
- minimum two players between ball and goal

Hard constraints are inviolable; soft ones carry weights. Solve as a constrained
optimization over L1 output. This layer is what makes the system speak the coach's
language rather than the model's.

### L3 — Objective optimizer

Continuous scoring functions over candidate positions. Inherit these, do not
re-derive them:

- **Pitch control** (Spearman) — probability each team controls each point
- **xT / expected threat** — value of ball possession by location
- **VAEP** — action valuation
- **Passing-lane coverage** — geometric, cheap, coach-legible
- **Voronoi / dominant regions** — space occupation

Optimize a weighted sum. The weights are a coaching philosophy made numeric, which
means they are a *feature* — different coaches get different weight profiles.

### L4 — Learned policy

The "figures it out itself" layer. **Full specification in MODEL.md** — summarized here.

It is a **world model first**: a spatio-temporal attention encoder over 23 entities,
pretrained on self-supervised trajectory forecasting and masked-agent modeling, with
policy and value heads added afterward. Positioning is derived from predicted futures
rather than memorized, which is simultaneously the fix for data scarcity and the
explainability mechanism.

**Compute note:** this model is small. Data is the binding constraint, not GPU.
A single consumer GPU or free-tier Colab is sufficient throughout. Do not let
infrastructure become a distraction.

### L5 — Arbiter

Combines L1–L4 into one answer per player, and — critically — **attaches a reason**.
Every mark on the pitch must be traceable to a stated cause: "3m narrower, to close
the passing lane to #10." Marks without reasons are unteachable and untrustworthy.

Also owns forecasting (§3.1).

### L6 — Renderer

Abstract draw calls -> backend. Backends: 2D replay canvas, coach tablet, pitchside
screen, projector array. Keep the geometry in pitch coordinates until the last
possible moment, so adding a backend is a projection matrix rather than a rewrite.

---

## 3. The three hard problems

### 3.1 Latency — you need a forecaster, not a solver

Continuous live overlay imposes a budget that forces an architectural change.

| Stage | Budget |
|---|---|
| Tracking acquisition (LPS ~100ms / CV ~150-300ms) | 100-300 ms |
| Network / ingest | 10-30 ms |
| L0-L5 compute | 20-80 ms |
| Render + display/projector lag | 20-50 ms |
| **Total** | **~250-450 ms** |

Players cover **1-3m** in that window. Rendering "where they should be now" produces
a mark that is already stale when the photons land.

**Therefore: L5 must output positions for t + latency.** A short-horizon (~400ms)
trajectory predictor is a required component, not a refinement. Fortunately, 400ms of
player motion is highly predictable from position + velocity — a constant-velocity
model with damping gets most of the way, and a small learned residual gets the rest.
Measure your actual end-to-end latency early and budget against the real number.

### 3.2 There is no ground truth for "ideal"

No dataset can tell you whether a suggested position is correct. You have no test set.

**Mitigation — build one.** This is the step most attempts skip, and it is the
highest-value work in the entire project:

1. Sample a few hundred diverse frames from public data.
2. Render the state plus the system's proposed positions.
3. Have coaches (you, plus others at the club) rate each: agree / disagree /
   disagree-and-here-is-where-it-should-be.
4. That corpus becomes your evaluation metric, your regression suite, and eventually
   fine-tuning data.

**Coach agreement rate is your north-star metric.** Not loss, not xT. Every layer
gets evaluated against it. Start collecting in Phase 2 — before the learned model
exists — so you have a baseline the model must beat.

### 3.3 Elite-trained models do not transfer to semi-pro

A policy learned from professional tracking outputs positions that assume
professional physical capability. Your players cannot sustain them — wrong sprint
distances, unreachable recovery runs, unsustainable pressing intensity. The model
will be *tactically* right and *practically* useless.

**Mitigation — a player capability model.** Per player: max speed, acceleration
profile, sustainable high-intensity distance, current fatigue state. Constrain L3/L5
so target positions are actually reachable by *that* player in the time available.

**This is where the club's existing GPS vests earn their keep — and it needs no live
access at all.** Post-hoc dock exports are entirely sufficient to build physical
profiles. It converts the biggest infrastructure unknown into a solved input, and it
is a genuine differentiator: nobody selling tactical software models whether the
player in front of you can actually do the thing being asked.

---

## 4. Model training ladder

Each rung is shippable and independently evaluable against coach agreement.

**L4a — Behaviour cloning.** Predict actual elite player positions, conditioned on
formation + phase + ball. Learns "what a good team does."

- Loss: per-player position delta, masked for missing tracks
- Augmentation: mirror across the long axis (free 2x); attack-direction normalization
- Watch for **regression to the mean**. Predicting "everyone drifts toward the
  centroid" scores decently and is worthless. Evaluate on tactically *interesting*
  frames, not uniformly sampled ones.

**L4b — Advantage-weighted regression toward xT-gain.** Reweight the BC objective by
the advantage of the observed action. Now it learns *good* positioning rather than
*typical* positioning. **This is the first rung where output can exceed the training
data**, and it may honestly be as far as you need to go for a working product.

**L4c — Offline RL** (IQL / CQL family) on real tracking. Learns policies better than
anything observed, with no simulator. Real risk of distributional drift — confident
nonsense in states unlike the data. Mitigate with conservative penalties and a hard
fallback to L3 when out-of-distribution (trigger on ensemble disagreement or a
learned uncertainty head).

**L4d — Self-play RL in simulation.** Google Research Football is the existing
environment. The "learns it itself from scratch" endpoint.

- **Non-negotiable: anchor with a KL penalty toward the L4a policy.** Unconstrained
  self-play agents exploit simulator physics and converge on tactics that are optimal
  in-sim and absurd on grass. The KL anchor keeps discovered behaviour inside the
  manifold of things humans can actually execute.
- Treat as research, not roadmap. Do not let it block the product.

### Data sources (all free)

| Source | Content | Use |
|---|---|---|
| **Metrica Sports** | 3 matches, full 2D tracking @25Hz + events | Cleanest starting point |
| **SkillCorner open** | ~9 matches, broadcast-derived tracking | Note: incomplete — only players in frame |
| **DFL / Bundesliga open** | ~7 matches, full positional @25Hz | Good volume, clean |
| **StatsBomb open** | Large event dataset + 360 freeze-frames | Freeze-frames, not continuous |
| **SoccerNet-GSR** | Video + tracking benchmarks | If you later need CV |
| **Google Research Football** | Simulator | L4d only |

**Key library: `kloppy`** — normalizes tracking and event data across Metrica,
SkillCorner, StatsBomb, Tracab and others into a single data model. This is
effectively your L0 source interface, already written. Use it.

Also: `socceraction` (VAEP), `mplsoccer` (viz), `floodlight` (analysis).

**Realistic volume: dozens of matches, not thousands.** Frames are heavily
autocorrelated — 3M player-frames per match is nowhere near 3M independent samples.
Size the model accordingly; a small, well-regularized GNN will beat a large one.

---

## 5. Hardware plan

### Phase A — Development (now)

Laptop. Public data. Zero spend. Everything through L4b is achievable here.

### Phase B — On-pitch, screen-based (~EUR 1-3k)

| Item | Purpose | Cost (EUR) |
|---|---|---|
| Coach tablet (iPad + Pencil) | Freehand drawing, live control | 600-1200 |
| Pitchside display (large TV on trolley) | Player-facing output | 500-1500 |
| 4K action cam or used PTZ + mast | Ball tracking, session capture | 200-800 |
| GPS vest post-hoc exports | Player capability profiles | Club already owns |

First configuration that runs at a real training session.

### Phase C — Projection pilot (~EUR 15-25k)

Run the photometry honestly before committing:

- Full pitch (105 x 68m = **7,140 m²**) at 50 lux above daylight ambient needs
  ~357,000 lumens *delivered to the surface*. With optical loss, keystone waste,
  edge-blend overlap, and grass albedo (~0.2, versus ~0.85 for a screen), the real
  requirement is **1-1.5M lumens of output**. At ~30,000 lumens per high-end laser
  projector that is 30-50 units — **EUR 2-5M plus rigging and power.** Not viable.

- **The viable configuration: a 30 x 20m working area (600 m²), at night or dusk with
  floodlights controlled.** Low ambient drops the legibility threshold to ~5-10 lux,
  cutting requirements roughly 5-10x. Result: **~10-20k lumens output, 2 projectors,
  EUR 15-25k** including mounting.

Also required: steep-angle geometric warping and edge blending onto a horizontal
plane (standard projection-mapping territory — MadMapper, Resolume, or OpenCV-based
warping), plus elevated mounting. Rent a scissor lift for the pilot; masts later.

**Conclusion: partial-pitch, low-ambient projection is the only honest target.**
Design the demo around it rather than discovering it late.

---

## 6. Roadmap

At ~20h/week. Each phase ends in something you can show someone.

**Phase 0 — Data foundation (weeks 1-3)**
kloppy ingest, canonical state representation, attack-direction normalization,
replay engine, 2D renderer, phase classifier v1.
*Demo: replay a real match in your own renderer with phase labels.*

**Phase 1 — Rules, objectives, and the analyst tool (weeks 4-9)**
L1 formation templates + deformation. L3 pitch control, xT, lane coverage. **Plus the
v1 tool shell**: scrub a replay, render proposed marks with deviation-only display,
drag a player to correct it, persist every correction as a label.
*Demo: you analyse a real match in your own tool and disagree with it in writing.*

**Phase 2 — Rating through use, and first contact (weeks 10-13)**
Use the tool. Accumulate a few hundred corrected frames as a by-product. Establish the
coach-agreement baseline for L1+L3. **Recruit 3-5 external coaches as raters** (§1.2) —
data, customer discovery and distribution in one activity.
*Demo: a number, and five coaches who have used your tool.*

**Phase 3a — World model pretraining (weeks 14-19)**
MODEL.md stages S1-S2: trajectory forecasting + masked agent modeling, then event and
value heads. No product output yet.
*Demo: the forecaster beats a constant-velocity baseline at 1s and 3s, and you can
show a coach three plausible futures for any frame.*

**Phase 3b — Policy head (weeks 20-25)**
MODEL.md stages S3-S4: behaviour cloning, advantage-weighted regression, finetune on
the corrections from Phase 2. Evaluate against the held-out rated frames.
*Demo: the learned model beats the rules on coach agreement.*

**Phase 4 — Override and explanation (weeks 26-30) — v1 ships**
L2 constraint solver, L5 arbiter with per-mark explanations and stability gating (§10
of MODEL.md), and manual override with re-solve via the masked-agent head.
*Demo: drag a player, watch the other ten rearrange coherently, with a reason attached
to each mark.*

**~7 months to v1, with zero hardware spend.** Assume 30% slip — normal, not failure.

### v2 and beyond

**Phase 5 — Live** — trajectory forecaster at 25Hz for latency compensation, real
tracking integration, Phase B hardware. *Live overlay on a screen at a session.*

**Phase 6 — Projection pilot** — Phase C hardware, geometric calibration, night session.

**Optional, any time after Phase 3 — Pose estimation** (MODEL.md §7.3), only if Phase 2
error analysis shows body orientation is a clustered cause of disagreement.

Also v2: freeze-frame on whistle, predicted-futures display, auto-selected teachable
moments, style counterfactuals, cue firing log, freehand drawing.

---

## 7. Open questions

1. **Vest vendor and package?** One email to the club resolves the single biggest
   schedule risk. Ask specifically: vendor, whether the package includes a live
   streaming API, and whether post-hoc per-player physical exports are available.
   Post-hoc alone unblocks §3.3, which matters more than live access does.
2. **Who else can rate frames?** Coach agreement is the north-star metric, and a
   sample size of one is thin. Two or three raters at the club would materially
   strengthen it.
3. **Which formations does the club actually play?** L1 should be built around the
   real shapes in use, not a generic library. Related: finalize the role taxonomy
   (MODEL.md §5.1) — the draft there is a template to be revised, not a spec.
4. **Who is the buyer — clubs, academies, or federations?** Now the live commercial
   question (§1.2). Different buyers imply different products, pricing and sales motions.

Resolved: commercial intent confirmed — the goal is a company, not a thesis or a tool
for one club. See §1.2 for what that changes.

Resolved: video access confirmed — club sessions can be filmed and some match footage
is available. This makes pose estimation reachable later (MODEL.md §7.3) and, more
immediately, enables the club-specific data source in MODEL.md §8.
