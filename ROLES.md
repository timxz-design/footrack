# Footrack — Role Taxonomy

**Status: DRAFT for coaching review.** The structure is considered; the numbers are a
starting point to be corrected. Every weight and constraint here is meant to be argued
with.

Feeds four layers: L1 reference positions, L2 constraints, L3 objective weights,
L4 role embeddings.

---

## 1. Coordinate convention

Normalized, centre-origin, attack-direction normalized (always attacking +x):

```
  x ∈ [-1, +1]    -1 = own goal line,  +1 = opponent goal line
  y ∈ [-1, +1]    -1 = left touchline, +1 = right touchline
```

Mirror augmentation is then simply `y -> -y` (MODEL.md §3.3).

Conversion on a 105 x 68m pitch: `x_m = (x+1)/2 * 105`, `y_m = y * 34`.

**Five-channel model** for the y axis, used throughout:

| Channel | y range | Centre |
|---|---|---|
| Left wing | -1.00 to -0.60 | -0.80 |
| Left half-space | -0.60 to -0.20 | -0.40 |
| Centre | -0.20 to +0.20 | 0.00 |
| Right half-space | +0.20 to +0.60 | +0.40 |
| Right wing | +0.60 to +1.00 | +0.80 |

---

## 2. Separation of concerns

A common mistake is to put positions in the role. Positions are formation-dependent — a
CB in a back three stands differently than in a back four — so:

- **The formation template owns positions** (§5). One file per formation.
- **The role owns behaviour** (§4): objective weights, constraints, ball response.

One taxonomy then serves every formation.

**Design principle: roles differ in what they optimize, not merely where they stand.**
Two roles occupying the same space with different priorities are genuinely different
roles. Two roles in different places optimizing the same thing are one role with two
reference positions.

---

## 3. The hierarchy

**8 base roles x modifiers.** The L4 embedding is `base_embedding + modifier_embedding`,
so a rare combination still inherits a sensible representation from its base.

| Base | Modifiers |
|---|---|
| **GK** | line, sweeper |
| **CB** | ball-playing, stopper, wide *(back three)* |
| **FB** | defensive, attacking *(wing-back)*, inverted |
| **DM** | holding, deep-playmaker |
| **CM** | box-to-box, shuttler |
| **AM** | ten, free-eight |
| **W** | touchline, inverted |
| **ST** | target, pressing, false |

~20 combinations, degrading gracefully to 8.

---

## 4. Role definitions

### 4.1 Objective weights

Coarse ordinal scale — **0 = ignore, 1 = minor, 2 = important, 3 = primary.** Coarse is
deliberate: it is authorable by a coach and avoids false precision.

*In possession:* **W**idth · **D**epth *(threat in behind)* · **P**rogression · **S**pace occupation
*Out of possession:* **BP** ball pressure · **LC** lane coverage · **GP** goal protection · **C** compactness

| Role | W | D | P | S | BP | LC | GP | C |
|---|---|---|---|---|---|---|---|---|
| GK · line | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 0 |
| GK · sweeper | 0 | 0 | 2 | 1 | 1 | 0 | 3 | 1 |
| CB · ball-playing | 0 | 0 | 3 | 1 | 1 | 2 | 3 | 2 |
| CB · stopper | 0 | 0 | 1 | 1 | 3 | 2 | 3 | 2 |
| CB · wide | 1 | 0 | 2 | 2 | 2 | 2 | 3 | 2 |
| FB · defensive | 2 | 1 | 1 | 1 | 2 | 3 | 2 | 3 |
| FB · attacking | 3 | 2 | 2 | 2 | 2 | 2 | 1 | 2 |
| FB · inverted | 1 | 1 | 3 | 3 | 1 | 3 | 2 | 3 |
| DM · holding | 0 | 0 | 2 | 2 | 1 | 3 | 3 | 3 |
| DM · deep-playmaker | 0 | 0 | 3 | 2 | 1 | 2 | 2 | 3 |
| CM · box-to-box | 1 | 2 | 2 | 2 | 3 | 2 | 2 | 2 |
| CM · shuttler | 1 | 1 | 2 | 2 | 3 | 3 | 2 | 3 |
| AM · ten | 0 | 2 | 3 | 3 | 1 | 2 | 1 | 1 |
| AM · free-eight | 1 | 2 | 3 | 3 | 2 | 2 | 1 | 2 |
| W · touchline | 3 | 2 | 1 | 1 | 1 | 2 | 0 | 1 |
| W · inverted | 1 | 3 | 2 | 3 | 1 | 2 | 0 | 1 |
| ST · target | 0 | 2 | 2 | 1 | 1 | 1 | 0 | 0 |
| ST · pressing | 0 | 3 | 1 | 1 | 3 | 2 | 0 | 1 |
| ST · false | 1 | 1 | 3 | 3 | 2 | 2 | 0 | 1 |

### 4.2 Ball response

How the reference position deforms with ball position (L1 deformation rules).

- `bx` — forward/back gain with ball x. 1.0 = moves 1:1 with the ball.
- `by` — lateral gain with ball y.
- `asym` — does the role behave differently ball-near vs ball-far? **This is often the
  defining trait** (an inverted winger is defined by what it does when the ball is far).

| Role | bx | by | asym | Ball-far behaviour |
|---|---|---|---|---|
| GK · line | 0.15 | 0.10 | low | — |
| GK · sweeper | 0.35 | 0.15 | low | — |
| CB · ball-playing | 0.70 | 0.30 | low | — |
| CB · stopper | 0.75 | 0.35 | low | — |
| CB · wide | 0.70 | 0.45 | med | Tucks central, becomes third CB |
| FB · defensive | 0.70 | 0.45 | high | Tucks into back three |
| FB · attacking | 0.85 | 0.55 | high | Drops to back line |
| FB · inverted | 0.60 | 0.35 | med | Holds central pivot position |
| DM · holding | 0.50 | 0.60 | med | Screens ball-side |
| DM · deep-playmaker | 0.55 | 0.50 | low | — |
| CM · box-to-box | 0.75 | 0.60 | med | Arrives late in box |
| CM · shuttler | 0.65 | 0.75 | high | Slides across to ball side |
| AM · ten | 0.70 | 0.50 | med | Occupies far half-space |
| AM · free-eight | 0.75 | 0.65 | high | Drifts to ball side |
| W · touchline | 0.80 | 0.25 | **v.high** | **Holds width, stays high and wide** |
| W · inverted | 0.75 | 0.35 | **v.high** | **Tucks to far half-space / back post** |
| ST · target | 0.80 | 0.30 | med | Occupies near post |
| ST · pressing | 0.85 | 0.55 | med | Screens the far CB |
| ST · false | 0.70 | 0.40 | med | Drops between lines |

### 4.3 Role constraints (L2)

Hard constraints are inviolable; soft carry weights.

| Role | Constraint | Type |
|---|---|---|
| GK (all) | Inside penalty area unless sweeping | hard |
| GK · sweeper | May advance to the defensive line height when line is above halfway | soft |
| CB (all) | Goal-side of the deepest opponent attacker | **hard** |
| CB (all) | Within 8-14m of adjacent defender | soft, strong |
| CB · stopper | May break the line to press a receiver within 10m | soft |
| FB (all) | Never higher than the ball when out of possession | **hard** |
| FB · attacking | Holds the wing channel in possession | soft, strong |
| FB · inverted | Occupies the pivot line, not the wing, in possession | soft, strong |
| DM (all) | Between ball and own goal | **hard** |
| DM · holding | Within 12m of the CB line when defending in own third | soft, strong |
| CM (all) | At least one CM goal-side of the ball at all times | **hard**, team-scoped |
| AM (all) | Positioned between opposition defensive and midfield lines in possession | soft, strong |
| W · touchline | Within 8m of the touchline in possession | soft, strong |
| W · inverted | Occupies the half-space, not the wing, in possession | soft, strong |
| ST · pressing | First to press the ball-carrying CB when press is triggered | soft, strong |
| ST (all) | Stays on the last defender's shoulder in possession | soft |

---

## 5. Team-level constraints (L2)

**Not derivable from individual roles.** These are where most coaching actually lives,
and they are the reason L2 exists as a separate layer.

| Constraint | Type | Note |
|---|---|---|
| Vertical compactness <= 25-30m out of possession | **hard** | The single most-drilled principle |
| Exactly one player per flank occupies the wing channel in possession | soft, strong | Resolves FB/W width conflicts dynamically |
| At least 2 players goal-side of the ball | **hard** | Counter-attack protection |
| Back line horizontal spacing 8-14m between adjacent defenders | soft, strong | |
| At least 3 passing options within 20m of the carrier | soft | |
| No more than 2 players in the same channel x third cell | soft | **Anti-clustering** — prevents the optimizer collapsing players together |
| Defensive line height >= agreed minimum (e.g. 35m from own goal) | soft | Team style parameter |

**On the width constraint:** this is the clean illustration of why team constraints
cannot be reduced to role positions. Whether the full-back or the winger holds width
depends on what the other is doing. Encoding "hold width" in both roles produces two
players on the touchline; encoding it once at team level lets the solver assign it
dynamically. Most width, rotation and overload principles have this shape.

---

## 6. Worked formation file: 4-3-3

Reference positions for the neutral state — **own possession, ball at centre circle,
medium block.** All deformation is relative to these.

| Slot | Role | x | y | x (m) | y (m from centre) |
|---|---|---|---|---|---|
| GK | GK · line | -0.92 | 0.00 | 4 | 0 |
| LCB | CB · ball-playing | -0.60 | -0.18 | 21 | -6 |
| RCB | CB · stopper | -0.60 | +0.18 | 21 | +6 |
| LB | FB · attacking | -0.50 | -0.68 | 26 | -23 |
| RB | FB · attacking | -0.50 | +0.68 | 26 | +23 |
| DM | DM · holding | -0.32 | 0.00 | 36 | 0 |
| LCM | CM · box-to-box | -0.10 | -0.30 | 47 | -10 |
| RCM | CM · box-to-box | -0.10 | +0.30 | 47 | +10 |
| LW | W · touchline | +0.30 | -0.80 | 68 | -27 |
| RW | W · touchline | +0.30 | +0.80 | 68 | +27 |
| ST | ST · pressing | +0.40 | 0.00 | 74 | 0 |

Swapping `W · touchline` for `W · inverted` moves the winger to `y = ±0.40` and the
team-level width constraint (§5) automatically pushes the full-back out to `±0.80`. That
cascade is the taxonomy working as intended — change one role, and the shape
reorganizes coherently rather than breaking.

---

## 7. Wiring into the layers

| Layer | Consumes |
|---|---|
| **L1** | Formation reference positions (§6) + ball response (§4.2) |
| **L2** | Role constraints (§4.3) + team constraints (§5) |
| **L3** | Objective weights (§4.1) as the scoring function's per-player weighting |
| **L4** | `base_embedding + modifier_embedding` as the role feature in MODEL.md §4.1 |

---

## 8. Open items for coaching review

1. **Are 8 base roles the right cut?** Specifically: is `AM` distinct enough from `CM`
   at semi-pro level, or do they collapse in practice?
2. **Which formations does the club actually play?** §6 is a worked example; the real
   formation files should reflect real shapes.
3. **The objective weights are the least confident part of this document.** They encode a
   coaching philosophy, and they encode *mine* rather than yours. Expect to revise most
   rows.
4. **Is the compactness range 25-30m right for your level?** Elite sides run tighter than
   semi-pro sides can sustain — this interacts directly with the player capability
   constraint (PLAN §3.3).
5. **Missing modifiers?** Roles the club uses that have no home in §3.
