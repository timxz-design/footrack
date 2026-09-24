# Footrack — Model Design

Companion to PLAN.md. Specifies the L4 learned layer: architecture, training
curriculum, inputs, and how player and team specifics enter.

Status: design agreed 2026-09-21.

---

## 1. Core principle: world model first

The model is **not** a function from game state to ideal position. It is a world model
that predicts what happens next, with policy and value heads bolted on top.

```
  Policy head      ideal position + uncertainty      <- the product
  Value head       xT / danger
  Event head       next event type
        ^
  Shared encoder   spatio-temporal attention, 23 entities
        ^
  Pretrained on    "what happens in the next 1-3 seconds?"   <- free labels
```

Three reasons this ordering is correct:

1. **It solves data scarcity sideways.** Trajectory forecasting is self-supervised —
   every frame labels the frame before it. ~84 matches yield millions of forecasting
   examples while yielding almost no "ideal position" examples.
2. **Positioning becomes derivable, not memorized.** A candidate position is good if,
   across plausible futures, it leads to good states.
3. **It is the explanation mechanism.** "Stand here, because if the ball goes there,
   you are the one who covers it." Marks without reasons are unteachable (PLAN §L5).

### 1.1 Why this works without good/bad labels

The most common point of confusion: *how can 84 matches train anything when nothing in
them is labelled good or bad?*

**They do not teach good or bad. They teach what happens.** Those are separate problems,
and only the second needs judgement.

**It is next-token prediction.** An LLM never sees a label saying which sentences are
good — you hide the next word and ask it to guess, and the text contains its own
answers. Grammar, facts and reasoning emerge as a side effect of trying to predict.

Identically here: hide the next 2 seconds, ask where all 23 entities go. The match
already contains the answer. Supervision is free.

**What prediction forces the model to learn.** Ball goes out to the right winger. To
predict the next 2 seconds, the model must get right that the left-back tucks in, the
far-side midfielder shifts across, the striker moves to the near post, and the defensive
line adjusts height. Predicting that everyone stands still is badly wrong and takes a
large gradient.

**The only way to reduce prediction error is to actually learn ball-oriented shifting,
team shape and pressing structure.** Nobody labelled "correct shifting" — the data shows
it ten thousand times, and there is no way to predict well without internalising it.

The subtlest thing it is forced to learn: **the run starts before the pass.** Predicting
a striker's movement requires implicitly representing that the pass is coming. That is
tactical cause-and-effect, extracted from pure prediction.

**Where good and bad actually enter** — three sources, increasing cost, decreasing volume:

1. **xT — computed, not labelled.** Expected threat values each pitch location by how
   often possession there historically led to a goal, derived from millions of past
   sequences. A statistic, not an opinion. Inherited from the analytics literature at
   zero labelling cost.
2. **What actually happened.** Every sequence has an outcome — possession lost, shot
   taken, territory gained. Known retrospectively, for free, on every sequence. This is
   the advantage signal.
3. **Coach ratings.** A few hundred frames — and they need only be a few hundred
   *because* 1 and 2 did the heavy lifting.

**The ordering is the trick:**

```
84 matches      ->  how football works       free, millions of examples
xT              ->  what is desirable        computed, no labels
coach ratings   ->  correcting the taste     a few hundred
```

Each stage needs orders of magnitude less data than the one before, because the previous
stage did the work. "Goals are good" is useless alone — it does not tell you where to
stand. You need a model of consequences first, and that is what the 84 matches buy.

**On the number 84.** It counts *contexts*, not training examples. 84 matches x 90 min
x 10Hz is ~4.5M frames, each supervising 23 agents x ~20 future timesteps — on the order
of a billion supervised values. 84 is small for tactical *variety*, which is precisely
why model size stays at 2-10M parameters (§3.1). It is not small for learning how
players move.

---

## 2. Training tasks

One encoder, five heads. Tasks 1-4 pretrain it; task 5 is the product.

| # | Task | Label source | Volume | Role |
|---|---|---|---|---|
| 1 | **Trajectory forecasting** | Self-supervised | Millions | Main pretraining |
| 2 | **Masked agent modeling** | Self-supervised | Millions | Representation + handles vest dropout and occlusion in production |
| 3 | **Event prediction** (pass/shot/turnover within N sec) | Event data (StatsBomb) | ~10^5 | Grounds encoder in tactical semantics |
| 4 | **Value regression** (xT / VAEP) | Computed | Millions | Value head |
| 5 | **Ideal position** | Coach ratings + advantage weighting | ~10^2-10^3 | The product head |

Task 5 has four orders of magnitude less data than task 1. That is acceptable *only*
because tasks 1-4 have already taught the encoder what football is.

**Task 2 note:** masked agent modeling is BERT-for-football and doubles as a production
feature. When a vest drops out or a player is occluded, the model inpaints rather than
fails.

---

## 3. Architecture

### 3.1 Tokens and attention

**23 tokens**: 22 players + ball.

At 23 agents, "GNN vs transformer" is a non-question — 23² = 529 attention pairs is
free, and a transformer with attention over agents *is* a fully-connected attention
GNN. Use attention; skip the graph machinery. (TacticAI used an explicit GNN because
corner kicks needed specific geometric equivariances. General play does not.)

**Factorized spatio-temporal blocks**: alternate attention-over-agents with
attention-over-time. Full joint attention costs (23 x T)² for no real benefit.

**History**: ~2-3 seconds. Downsample to **5-10Hz** — 25Hz is far more temporal
resolution than tactics needs, and subsampling cuts sequence length 3-5x for
negligible information loss. Keep 25Hz only for the short-horizon latency compensator
(§6).

**Size**: 4-8 layers, d_model 128-256, ~2-10M parameters. Resist going bigger.

**29 matches are ungated and cached** — 2 Metrica, 7 DFL, 20 SkillCorner — for roughly
1.5M frames at 10Hz. PFF's 64 would take that to 93 if the access request lands, but is
not required.

At a 25-frame stride that is ~59k windows, ~236k after symmetry augmentation, each
carrying dense per-agent supervision across 23 agents and 20 future timesteps. That
supports the size range above. It does **not** support going larger: 29 matches is ~29
independent tactical contexts regardless of frame count, and the binding constraint is
tactical variety rather than sample count.

### 3.2 Multi-modal output — non-negotiable

Football futures are genuinely multimodal. The ball goes left *or* right; the average
of those is a position no one should ever occupy. A single-mode model trained with MSE
produces exactly the centroid-drift mush that makes trajectory models worthless.

**Use K modes (3-6) with a variety loss** — the closest mode receives most of the
gradient — plus a cross-entropy over mode logits. Simple and stable at this data
scale; CVAE or diffusion heads are not worth it below ~1000 matches.

> **Measured correction — two distinct failures, found in sequence.**
>
> **1. Pure winner-take-all starves the losers.** Giving gradient *only* to the closest
> mode meant whichever mode was best early won every sample and took all the gradient.
> After two epochs one mode won 100% of validation windows with the other five stranded
> at initialisation, 28m from anything sensible. Fixed with **relaxed winner-take-all**:
> the winner takes `1 - epsilon`, losers share `epsilon` (default 0.05).
>
> **2. A pooled selection metric drowns out the ball.** After fixing (1), collapse
> persisted — still 1 of 6 modes. But the modes were *not* collapsed; the metric was
> blind. Selecting on all 23 entities used 1 mode; selecting on the ball alone used all
> 6, with a realistic distribution. The 22 predictable players carry **~87%** of the
> pooled error, so the mode fitting *them* best wins regardless of what it does with the
> ball — and the ball is the only entity whose future is genuinely multimodal.
>
> Fixed by scoring ball and players separately and combining with
> `ball_selection_weight` (default **1.0** — 0.5 was measured and still collapsed). The
> trajectory loss still covers every entity; only the choice of winner is reweighted.
> This is also the semantically right framing: the ball's future largely determines the
> players' futures, so "which mode is this scene in" mostly means "where does the ball go".
>
> **It costs accuracy.** At matched epochs on two matches, weight 1.0 raised effective
> mode count from 1.0 to ~2.4 and cost ~10% displacement error — the winner is chosen for
> ball fit but trained on all 23 entities, so its gradient is less focused. Taken anyway:
> collapse discards multimodality entirely, wasting five of six modes and leaving no
> uncertainty estimate, which removes the predicted-futures feature and the honest-
> uncertainty story. Revisit on the full corpus.
>
> **3. `epsilon` keeps modes alive but makes them converge.** With the first two
> fixed, all six modes trained — and drifted toward the same prediction. Mean inter-mode
> spread fell to **0.77m at t+2s**, against 5-10m of actual player movement, so the modes
> were saying the same thing in six different ways. The loser share pulls every mode
> toward the winner's target on every sample, so given enough steps they collapse
> together. `epsilon` likely needs annealing to zero once modes have separated, rather
> than being held constant. Unresolved — it belongs with the K and horizon decisions,
> and wants the full corpus.
>
> **Read inter-mode spread and winner perplexity together.** They fail in opposite
> directions: 28m spread with one winner means dead modes; 0.77m spread with spread-out
> winners means converged ones. Either number alone is ambiguous.
>
> **Two general lessons:**
>
> 1. Mode diversity and mode *selection* are separate problems. A collapsed winner
>    distribution does not prove the modes collapsed — measure inter-mode spread before
>    concluding. The epsilon fix looked like a failure when it had in fact worked.
> 2. Count modes by **perplexity of the winner distribution**, not by how many are
>    non-zero. `[1170, 1360, 70, 65, 42, 109]` reads as "6/6 alive" while being ~2.7
>    effective modes.
>
> **Open, and not resolvable on two matches: K=6 is over-provisioned for a 2s horizon.**
> Effective mode count sits at ~2.4 even with clean selection, because over two seconds
> the ball mostly continues — the bifurcations that would justify six modes play out over
> longer windows. Either drop K to 3-4 or extend the horizon. Decide on the full corpus;
> forcing six modes into balanced use on this much data would be fitting noise.

This is also a pedagogical gift. Show the coach: *"the model sees three things that
could happen here — this position covers all three."* That is precisely how coaches
teach defensive positioning.

### 3.3 Symmetry — where scarce data is won

| Symmetry | Gain | How |
|---|---|---|
| Permutation equivariance within team | Structural | Free from attention — **no positional encoding on the agent axis** |
| Left-right mirror | 2x | Augmentation |
| Attack-direction normalization | 2x | Always represent as attacking left-to-right |
| Team swap | 2x | Relabel attacker/defender — model must stay conditioned on which team it advises |

**Critical anti-symmetry: the pitch is NOT translation invariant.** This is the key
divergence from the autonomous-driving trajectory literature (AgentFormer, Trajectron++,
SceneTransformer, Wayformer) that this architecture otherwise borrows from. There is no
map topology, but there *are* absolute landmarks — goals, boxes, touchlines — and 30m
from goal means something 30m from halfway does not.

Use absolute coordinates plus explicit goal-relative features. Do not let a
driving-derived architecture quietly impose translation equivariance.

---

## 4. Input representation

### 4.1 Per-agent token features

| Feature | Dims | Notes |
|---|---|---|
| Position x, y | 2 | Absolute pitch coords, normalized |
| Velocity vx, vy | 2 | |
| Speed | 1 | |
| Velocity direction | 2 | **As (cos, sin)** — never raw radians (discontinuity at ±pi) |
| Entity type | 3 | own / opponent / ball, one-hot |
| Role embedding | 16-32 | Learned, from the taxonomy in §5.1 |
| Physical attributes | 3-4 | max speed, accel, stamina index — normalized |
| Tendency residual | 4-8 | Per-player, hierarchically shrunk (§5.2) |
| Dist + bearing to ball | 3 | dist, cos, sin |
| Dist + bearing to each goal | 6 | |
| **Body orientation** | 3 | **(cos, sin) + validity bit — zero-filled and masked for now (§7)** |
| Valid/active mask | 1 | Dropout, occlusion, masked-agent task |

**Reserve the orientation slot now even though it is unavailable.** Adding orientation
later then becomes a data-acquisition problem rather than an architecture rewrite.

### 4.2 Context features

Global token (CLS-style) or broadcast to all tokens:

- phase (4-way: in-possession / out-of-possession / transition / set piece)
- score differential, time remaining
- own team style vector (§5.3), opponent team style vector
- coach style override flag

---

## 5. How player and team specifics enter

### 5.1 Role taxonomy

Defined from coaching knowledge, not clustered from data. This is where domain
expertise is directly encoded, and it is the part of the system a black box cannot
replicate. The taxonomy also feeds L1 templates and L2 constraints, so it is a shared
vocabulary across the whole stack, not just a model input.

Starting draft, ~14 roles — **to be revised by the coach, this is a template not a spec:**

```
GK                      sweeper-keeper / line-keeper variants
CB  ball-playing        steps into midfield, progressive passing
CB  stopper             aggressive front-foot defending
CB  wide (back 3)       covers channel, steps wide
FB  defensive           holds width, stays deep
WB  attacking           high and wide, overlaps
FB  inverted            tucks into midfield in possession
MF  holding (6)         screens back line, positional discipline
MF  box-to-box (8)      vertical range, arrives in box
MF  deep playmaker      dictates tempo from deep
MF  advanced (10)       operates between lines
WF  touchline winger    holds width, 1v1
WF  inverted winger     cuts inside into half-space
CF  target / pressing 9 depending on out-of-possession role
```

Each role carries: reference position in L1, default constraints in L2, objective
weights in L3, and a learned embedding in L4.

> **Measured: the role embedding was never populated.** For several training runs
> `role_ids` was zero and `player_ids` was -1 for every player, so the model was asked to
> place a hidden player knowing only their team and physical profile. Its reconstructions
> collapsed onto a single lateral line, and its one reliable guess was the goalkeeper —
> inferred from being slow, the only identity signal that reached it.
>
> Populating both (roles inferred from average position, see `footrack/roles.py`) cut
> reconstruction error from 14.9m to 11.8m against an 18.7m centroid baseline, and
> reached the previous run's final quality by epoch 3 of 6. Forecast error barely moved:
> identity helps you infer *where someone is*, not how they are moving.
>
> **Still open — base roles do not encode side.** A left-back and a right-back are both
> `FB`, so lateral spread recovered only from 0.8m to 2.9m across a 68m pitch. Which wing
> a player occupies is meant to come from the tendency embedding, which is now wired but
> starts at zero and needs far more than two matches per player. Encoding side in the
> role instead would work, but mirror augmentation would then have to swap left and right
> roles as it flips y — a real interaction to handle, not a free addition.

### 5.2 Player representation — factorized, not ID-based

**The cold-start trap:** a learned embedding per player ID works beautifully in
training and is useless at the club, because semi-pro players have zero frames in
elite tracking data. An ID embedding cannot transfer to a player it has never seen.

| Component | Source | Transfers to new players? |
|---|---|---|
| Physical (max speed, accel, stamina) | **GPS vests — post-hoc export is sufficient** | Yes — measure directly |
| Role | Coach assigns | Yes |
| Tendency residual (4-8d) | Learned per player | Shrinks to role mean |

**Hierarchical shrinkage for the tendency residual.** Model it as

```
  tendency_player ~ N(role_mean, sigma^2)
```

Empirical-Bayes shrinkage toward the role mean, with shrinkage strength decaying as
frames-per-player accumulates. This handles cold start automatically: an unseen player
starts at their role's mean and individualizes as data arrives. Far better than an L2
penalty with a hand-tuned constant.

**Physical attributes enter twice:**
- as **input to prediction** — a quick full-back changes which futures are reachable
- as **constraint on output** — do not demand a run he cannot make (PLAN §3.3)

### 5.3 Team tendencies

Same reasoning: a learned team embedding does not transfer to the club. Compute a
**style descriptor from observable statistics** instead —

- PPDA (pressing intensity)
- average defensive line height
- build-up directness
- possession share
- average width in/out of possession
- counter-press recovery rate

These are computable for your own club from your own sessions, so they transfer by
construction.

**Make it a control surface, not just an inference.** Let the coach override the style
vector and ask counterfactuals: *"what would our shape look like here if we pressed
like a high-block team?"* This turns the model into a tactical sandbox and is likely
one of the more sellable features in the system.

---

## 6. Prediction horizons — three problems, three methods

Do not use one model for all three.

| Horizon | Use | Method |
|---|---|---|
| **~400ms** | Latency compensation (PLAN §3.1) | Near-deterministic. Constant-velocity + damping, small learned residual. Runs at 25Hz. |
| **1-3s** | Tactical prediction — who receives, where the ball goes | Genuinely multimodal. The K-mode world model. Runs at 5-10Hz. |
| **5-10s** | Possession outcome | Too uncertain to forecast as trajectories. Use value functions (xT/VAEP). |

---

## 7. Visual cues: orientation, gaze, body language

### 7.1 The availability problem

Body orientation is one of the strongest short-horizon predictors in football — hip
orientation dictates which way a defender can turn; a receiver's body shape says
whether they can play forward. Jordet's scanning research shows head-movement frequency
predicts performance. The instinct that this matters is correct.

| Cue | GPS vests | Public tracking data | Video |
|---|---|---|---|
| Position, velocity | Yes | Yes | Yes |
| **Velocity direction** | Yes | Yes | Yes |
| Body/hip orientation when slow or stationary | No | No | Yes (pose estimation) |
| Head / gaze direction | No | No | Hard — needs resolution |
| Body language, intent | No | No | Research frontier |

**Every public dataset in the plan — Metrica, SkillCorner open, DFL — is 2D centroid
positions only.** Orientation cannot be trained from them at all.

### 7.2 Strategy

**1. Velocity direction is a free ~80% proxy.** Above ~2 m/s, body orientation
essentially *is* velocity direction. Costs nothing, present in every dataset. Day one.

**2. The residual concentrates in the moments that matter most** — the defender setting
their feet, the midfielder opening their body before receiving. Low-velocity moments
are exactly where the proxy fails and where orientation carries most information.

**3. Resolve it empirically, not by guessing.** In Phase 2 coach ratings, tag whether
orientation was the deciding factor in each disagreement. If those cluster, there is a
measured case for pose estimation. If not, a CV project has been avoided.

**4. Fatigue is the actionable part of "body language", and the vests already have it**
— accumulated high-intensity distance, decay in peak speed across a session. The useful
fraction of body-language reading comes from hardware the club owns.

### 7.3 If pose estimation is pursued

Realistic off-the-shelf pipeline (CV is not the strong suit here, so assemble rather
than build):

- Detection: YOLOv8 / RT-DETR
- Tracking: ByteTrack / BoT-SORT
- Homography: from pitch line detection (SoccerNet-GSR baselines, roboflow sports repos)
- Pose: RTMPose / ViTPose on player crops -> hip and shoulder keypoints -> orientation

Expect: detection good, **ID switches frequent** from a single mast camera, homography
workable if lines are visible, orientation noisy at distance. Needs 4K and a reasonably
close camera position. Multi-week project even with off-the-shelf parts. Defer until
after Phase 3.

---

## 8. Two data sources, two jobs

This division is the organizing principle of the whole data strategy:

| Source | Provides | Feeds |
|---|---|---|
| **Elite public data** (Metrica, SkillCorner, DFL, StatsBomb) | Tactical prior — what good football looks like | Pretraining tasks 1-4; role embeddings; L4a/L4b |
| **Club footage + GPS vests** | Player-specific layer — who *these* players are | Physical attributes, tendency residuals, team style vector, orientation (later), coach-rating set |

Elite data teaches the model football. Club data teaches it *your* players. Neither
substitutes for the other, and the factorized player representation (§5.2) is precisely
the seam where they join.

---

## 9. Training curriculum

| Stage | Content | Gate |
|---|---|---|
| **S0** | Data pipeline via kloppy; canonical state; augmentation | Replay + augmented batches verified |
| **S1** | Pretrain tasks 1+2 (forecast, masked agent) | Forecast beats constant-velocity baseline at 1s and 3s |
| **S2** | Add tasks 3+4 (event, value), multi-task | Event AUC and xT correlation acceptable |
| **S3** | Policy head: behaviour cloning, then advantage-weighted (L4b) | Beats L1+L3 rules on coach agreement |
| **S4** | Coach-rating finetune — tiny data, low LR, head-only or adapter | Coach agreement improves without forecast regression |
| **S5** | Offline RL (IQL/CQL), then self-play with KL anchor | Research. Must not block product. |

**Evaluation discipline throughout:**
- Hold out whole *matches*, never random frames — adjacent frames are near-duplicates
  and random splits will produce badly inflated scores.
- Evaluate on tactically *interesting* frames, not uniformly sampled ones. Uniform
  sampling is dominated by low-information moments where predicting "everyone drifts
  toward the centroid" scores well and means nothing.
- **Coach agreement rate remains the north star.** Forecast loss is a proxy for a proxy.
- Track **output stability** alongside accuracy (§10). An accurate but jittery system is
  a failed product, and nothing in the loss function will tell you that.

---

## 11. Team adaptation: one general model, conditioned per club

**Never train a team-specific model from scratch.** Pretrain general, adapt by
conditioning.

### 11.1 Why not team-specific

1. **You would learn their mistakes.** Behaviour cloning a semi-pro side produces a
   semi-pro-side imitator — a system that instructs players to do exactly what they
   already do wrong. This alone settles the question.
2. **Vests cannot supply the training data anyway.** GPS gives your eleven only — no
   opponents, no ball. The model needs the full 23-entity state, so own-team tracking
   requires CV on match footage. You would pay a real cost to obtain worse data.
3. **Coverage.** A rare press trigger appears perhaps three times in one team-season and
   three hundred times across the 84-match public corpus.
4. **It does not scale as a business.** Per-customer training runs mean season-long
   onboarding and consulting margins. Conditioning means a club is configured in an
   afternoon.

### 11.2 The adaptation ladder

Team-specificity lives in the **conditioning inputs, not the parameters**. This is what
the factorized representation in §5 exists for.

| Mechanism | Team data needed | Notes |
|---|---|---|
| Formation + role assignment (L1) | **None** | Coach declares |
| Constraints (L2) | **None** | Coach declares |
| Objective weights (L3) | **None** | Coaching philosophy as numbers |
| Team style vector (§5.3) | 3-5 matches | Statistics, not training |
| Player physical attributes (§5.2) | 3-5 sessions | Vests, or derived from tracking |
| Tendency residuals (§5.2) | ~10+ matches/player | Hierarchical shrinkage fills in gradually |
| Coach-correction finetune | ~200-500 corrections | **Head-only** — see §11.4 |

The first three rows require no training whatsoever and deliver most of the
team-specificity.

### 11.3 Where team data does belong

Not as imitation targets — as **evaluated** examples:

- **Through advantage weighting.** A bad sequence carries negative advantage, so the
  model learns from the team's mistakes *as mistakes* rather than as targets.
- **As coach corrections.** These are supervised "ideal", not observed "typical".

The division: **elite data teaches what right looks like; team data teaches what went
wrong.** Never let team data act as a behaviour-cloning target.

### 11.4 Finetuning discipline

With a few hundred correction labels against a 5M-parameter model, full finetuning
overfits catastrophically and destroys the pretrained representations.

- **Freeze the encoder, train the head only.** Simplest and almost certainly sufficient
  at this scale.
- LoRA/adapters if head-only proves insufficient.
- Low LR, early stopping, and always verify the forecast head has not regressed —
  forecast quality degrading after a policy finetune is the signal that you have
  overfitted.

---

## 10. Output stability and learnability

A correct instruction that changes every 200ms teaches nothing. Players learn a small
number of repeatable principles, not a continuously moving coordinate. Stability is
therefore a **hard product requirement**, not a polish item — and it must be imposed
explicitly, because no training objective produces it for free.

Three mechanisms, applied in the arbiter (L5):

### 10.1 Rules as the floor — deviation gating

Model output is used only when **both**:

- model uncertainty is below a threshold, **and**
- predicted improvement over the L1/L2 rule baseline exceeds a margin `delta`

Otherwise the system falls back to the rule layers. This makes output consistent by
construction, and makes failure look like conventional coaching rather than something
strange. `delta` is a directly tunable conservatism dial — raise it and the system
speaks only when it has something clearly worth saying.

### 10.2 Temporal hysteresis

A mark moves only if:

- the new target differs from the current mark by more than `d_min` (start ~2-3m), **and**
- the value improvement exceeds `epsilon`, **and**
- both conditions hold continuously for a dwell time (start ~0.5-1s)

Plus **rate limiting** on mark movement, so it glides rather than teleports.

Without dwell time, marks flicker at decision boundaries — the model oscillating
between two near-equal options is correct on every frame and useless on all of them.

**The tension to tune:** under continuous live overlay, too much hysteresis is stale,
too little is unreadable. Tune dwell time against coach feedback, not intuition.

### 10.3 Vocabulary compression

The on-pitch output should not be "stand at (34.2, 51.8)". It should be a small set of
named, repeatable cues — target ~10-20 total.

Method: cluster the policy's *deviations from the rule baseline* across many frames.
Recurring clusters become named cues, each with a trigger condition and a named action
("ball goes wide-left -> far winger tucks to the far post line"). The coach names them;
this is domain work, not clustering work.

Then the live system reports **which cue fired**, and the mark is that cue's target.
This makes output teachable between sessions, reviewable afterward, and gives players
a vocabulary they can hold in their heads on a pitch where nobody is looking at a
screen.

### 10.4 Metrics

Report alongside coach agreement:

- mark movements per player per minute
- mean dwell time per mark
- fraction of frames where the model overrode the rule baseline (the "speaking rate")
- cue distribution — a healthy system fires a handful of cues often, not 200 once each
