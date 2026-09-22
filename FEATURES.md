# Footrack — Feature List

What is actually useful, how, and what each thing costs to build.

Key: **M** = needs the learned model · **T** = needs live tracking · **P** = needs
projection · **—** = needs none of those

---

## 1. Offside line on the backline — T, P

A live line across the pitch at the second-rearmost defender.

**Scenario — defensive line discipline.** A back four is drilled to step up together.
Currently the coach shouts, the players argue about who was late. With the line
projected, the moment one defender plays everyone onside it is *visible to all of them*
simultaneously. The feedback loop collapses from "coach's word against mine" to
"I can see it."

**Scenario — timing runs.** A striker practising runs in behind gets instant, silent
feedback on whether they went early. No whistle needed.

**Honest notes**
- Trivial to compute — it is just the second-rearmost defender's x-coordinate.
- **A teaching aid, not VAR.** At 300ms latency and a back line stepping at 2-3 m/s, the
  line is ~1m off. Fine for teaching the concept; do not let anyone treat it as a ruling.
- **Lines are the most projection-feasible content**, though not for the obvious reason.
  Projectors modulate per-pixel and discard blocked light, so a thin line gets the same
  lux as a filled shape. The advantage is perceptual: human vision detects thin,
  high-contrast lines at far lower contrast than it reads filled regions or text. A line
  is legible at ambient levels where a heatmap is mud.

**Verdict: high value, low effort. One of the best ratios in the list.**

---

## 2. Freehand drawing projected on the pitch — P

Coach draws on a tablet; it appears on the grass, life-size, in place.

**Scenario — the board-to-pitch gap.** Coach stops a session, gathers 11 players round a
whiteboard, and says "imagine you're here." Every player now has to mentally rotate and
scale a 2D diagram onto the 3D space they are standing in. **That transform is exactly
where instruction is lost**, especially for younger and lower-level players. Drawing the
pressing arrow on the grass *where the player is standing* removes the transform
entirely.

**Scenario — pre-built sessions.** Draw your session plan the night before, recall each
diagram on the pitch by name. Save, replay, animate a movement path over time.

**Honest notes**
- Needs no model, no tracking, no prediction. A projector, a calibrated homography, and
  a tablet.
- This is the tactics board coaches already use and already trust — just life-size and
  in situ. Zero explanation required to sell it.

**Verdict: probably the highest value-per-unit-effort feature in the entire project,
and by a distance the easiest to demo. Technically trivial next to everything else here.**

---

## 3. Empty passing zones / free space — T, P, (M for quality)

Show where usable space actually is.

Define it as **pitch control** — space you would win the race to — not merely "space
with no defender in it." Those differ, and the second one is misleading.

**Scenario — build-up training.** A midfielder who habitually does not scan sees, after
the moment, that there were 15m of free space behind him. That is an "oh" moment a coach
shouting "turn!" never produces.

**The serious caveat — do NOT display this live.** If free space is shown on the ground,
players look at the ground instead of over their shoulder. You would be actively
training *away* the scanning habit, which is one of the most robust predictors of
performance in the research literature (Jordet). You would make players worse at the
real game while making them look better in the drill.

**Use freeze-frame or post-hoc only.** Show it after the moment, never during.

**Verdict: genuinely valuable, but the impressive version (live) is worse than the
modest version (freeze-frame). Ship the modest one.**

---

## 4. Optimal positions from set rules — T, P

Projected marks for where each player should be, given ball position and phase.
In-possession and out-of-possession variants.

**Scenario — shadow play.** An 11v0 pattern-of-play session. Currently the coach stops
play constantly to correct shape. With marks projected, players self-correct while the
ball keeps moving. The session runs at twice the tempo and the coach saves their voice.

**Scenario — onboarding.** A new signing, or a player switching position, learns the
system in one session instead of five.

**Honest notes**
- Rules-only (L1/L2/L3) is enough for this. No model required.
- The marks will be visibly wrong sometimes, and in this context that is **fine** — the
  coach overrides. The system is a fast first draft, not an authority. Framing it that
  way to coaches matters more than accuracy does.

**Verdict: the core feature. Rules version is ~80% of the value at ~10% of the cost.**

---

## 5. Post-match analytics that feed back into training — T, M

**This is the product's actual thesis, and the phrase "implemented in training" is the
whole point.** Most analytics products stop at the insight and leave the coach to invent
the drill.

**Scenario — closing the loop.** Match review finds the left side was exposed on
transition six times, and in five of those the left-back was 8m too high. The system
does not just report that — it generates the drill: transition scenario, left-back
starting position, projected correct recovery path, run it Tuesday.

**Analysis -> drill is the loop nobody closes.** Every competitor produces dashboards
that a coach has to translate into a session by hand. That translation step is where the
value leaks out, and it is the one place the model + projection combination is uniquely
positioned.

**Verdict: the strongest strategic idea in the list. This is what the product is FOR.**

---

## 6. Opponent analytics, pattern detection, recommendations — T, M

**Honest problem: at semi-pro level there is no tracking data on your opponents.** You
would be working from a single filmed match. "Pattern detection" from one match is
observation, not analytics — the sample cannot support the claim.

**Reframe: self-analytics first, opponent second.** You have every one of your own
matches. That is a real dataset and it supports real claims:

**Scenario — recurring failure modes.** "We concede from the left half-space between
the 60th and 75th minute" is detectable across a season of your own matches, and it is
immediately actionable. This is more useful than a thin read on next week's opponent,
and it is the same machinery.

Opponent analysis becomes strong at levels that have tracking data. Build the capability
generically; expect it to pay off later.

**Verdict: valuable, but invert the order. Own-team pattern detection is deliverable
now; opponent analysis is aspirational at this level.**

---

## 7. Preset drills with voice instruction — P, voice

**Scenario — hands-free control.** Coach is 40m from the tablet, arms folded, watching.
"Show 4-4-2 mid block" — marks appear. "Shift left ten" — the block shifts. No walking
back to a device, no breaking attention.

Voice is genuinely the right interface here: a coach's hands are busy, they are far from
any screen, and they are already talking.

**Honest notes**
- Outdoor voice recognition with wind and shouting is hard. Needs a **headset mic**, not
  ambient pickup.
- Use a **constrained command grammar**, not open-ended interpretation. A fixed
  vocabulary that works 99% of the time beats a flexible one that works 85%. A coach who
  gets ignored twice stops using it permanently.
- Preset drill library pairs naturally with saved drawings (#2) into a "session plan"
  concept.

**Verdict: high value for daily usability, moderate effort, one real technical risk
(outdoor ASR).**

---

## Additions worth considering

### 8. Pressing trigger zones — T, P
Project a zone; when the ball enters it, the zone lights up. The trigger becomes
visceral rather than verbal. Teaches *when* to press, which is harder to coach than
*how*.

Dead simple — geometry plus ball position. No model. **Excellent effort-to-value.**

### 9. Ghost trail: actual path vs ideal path — T, (M for "ideal")
Project where the player actually ran alongside where they should have. "You went here.
The run was here."

Works in the analyst tool *and* on the pitch. Cheap once marks exist. Extremely legible
— it is the one visualization that needs no explanation at all.

### 10. Shape adherence score — T
How long did the team hold correct shape? Score it, track it session over session.
Gamifies shape discipline, which is otherwise the least engaging thing to drill.
Particularly effective with youth.

Nearly free once marks exist.

### 11. Tempo cues — P
Project a countdown for build-up: "six seconds from back to front." Trains tempo, which
is almost impossible to coach verbally without disrupting the thing you are timing.

---

## Summary: effort vs value

| # | Feature | Model? | Effort | Value | Note |
|---|---|---|---|---|---|
| 2 | Freehand drawing | No | **Low** | **High** | Best ratio in the list |
| 1 | Offside line | No | **Low** | High | Most projection-feasible content |
| 8 | Pressing trigger zones | No | **Low** | High | Underrated |
| 4 | Optimal positions (rules) | No | Medium | **High** | The core |
| 9 | Ghost trail | Partly | Low | High | Needs no explanation |
| 10 | Adherence score | No | Low | Medium | Free once marks exist |
| 5 | Analytics -> drill loop | Yes | High | **Highest** | The actual thesis |
| 3 | Free space display | Partly | Medium | Medium | Freeze-frame only — see caveat |
| 7 | Voice + preset drills | No | Medium | High | Outdoor ASR is the risk |
| 11 | Tempo cues | No | Low | Medium | |
| 6 | Opponent analytics | Yes | High | Medium | Invert to self-analytics first |

**The pattern: six of the eleven features need no model at all**, and three of the four
best effort-to-value ratios are model-free. The model earns its keep in exactly two
places — the analytics-to-drill loop (#5) and the quality of the position marks (#4).

Everything else is tracking, geometry, and a projector.
