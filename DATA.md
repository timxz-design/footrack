# Footrack — Data Sources

Verified 2026-09-21. Availability and licensing change; re-check before relying on any
entry.

---

## 1. The headline answer: no computer vision needed to start

**The public datasets are already coordinates, not video.** Someone else has already run
the CV. You can begin training the world model with zero CV work.

CV becomes relevant in exactly two later places, both optional:

1. Turning **your own club footage** into tracking data (the player-specific layer,
   MODEL.md §8)
2. Extracting **body orientation** via pose estimation (MODEL.md §7.3)

And even then you assemble off-the-shelf components — you do not write a CNN. The
"CNN" is a pretrained YOLO checkpoint you download.

---

## 2. Tracking datasets (positions — the core training data)

### Full optical tracking — all 22 players, always present

| Dataset | Matches | Detail | License |
|---|---|---|---|
| **DFL / IDSSE-data** | 7 | TRACAB tracking @25Hz + synced DFL event data. German Sport University Cologne. | CC BY 4.0 |
| **Metrica Sports sample data** | 3 | Anonymised, synchronised tracking + events. CSV / EPTS / JSON. | Open |

**~10 matches.** Small, but complete states. **Reserve these as the clean validation
and test set** — honest evaluation needs frames where you know where everyone actually
was.

### Broadcast-derived tracking — only players in camera frame

| Dataset | Matches | Detail | Access |
|---|---|---|---|
| **PFF FC / Gradient Sports — 2022 Men's World Cup** | **64** | Broadcast tracking + event data + play-by-play grades, all 64 games. Loadable via kloppy. | Free on access request (form) |
| **SkillCorner open data** | 10 | A-League 2024/25. Broadcast tracking + physical data, **plus off-ball runs and phases of play labels**. | MIT |

**~74 matches.** This is the bulk of the corpus and the PFF set is the single biggest
free tracking release available.

### The gap problem is not a problem

Broadcast tracking only sees players in frame — typically 70-85% coverage, with players
appearing and disappearing.

**This is exactly what masked agent modeling (MODEL.md task 2) consumes.** The missing
players *are* the mask. Naturally occurring, correctly distributed training signal for
a task you wanted anyway — and the resulting inpainting ability is precisely what you
need in production when a GPS vest drops out mid-session.

The weakness of broadcast data is the strength of the pretraining task. Do not filter
these frames out.

### Two useful bonuses

- **SkillCorner phases of play** — supervised labels for the L0 phase classifier you
  were otherwise going to write as rules. Train it properly instead.
- **PFF play-by-play grades** — per-action quality grades, a potential supervision
  signal for the value head alongside xT/VAEP.

---

## 3. Event datasets (no continuous positions, but large)

| Dataset | Content | Use |
|---|---|---|
| **StatsBomb Open Data** | Full competitions — World Cups, Euros, full Messi-era La Liga seasons, NWSL, FA WSL | Event head (task 3), xT/VAEP fitting |
| **StatsBomb 360** | Freeze-frame player positions at moment of each event | Positional but discontinuous — good for value/event heads, **not** for trajectory forecasting |
| **Wyscout open data** | 7 competitions of event data (2018 public release) | Supplementary |

StatsBomb 360 is underrated here: tens of thousands of positional snapshots. It cannot
train a forecaster (no continuity) but it can train the value and event heads.

---

## 4. Video datasets (only if pursuing CV later)

| Dataset | Content |
|---|---|
| **SoccerNet** | Action spotting, player re-ID, tracking |
| **SoccerNet-GSR** (Game State Reconstruction) | Literally video -> pitch coordinates, **with published baselines** |
| **SoccerTrack / SportsLabKit** | Drone and fisheye footage with tracking annotations |

If you build a club-footage pipeline, start from SoccerNet-GSR baselines rather than
from scratch.

---

## 5. Simulation

**Google Research Football (gfootball)** — the environment for MODEL.md stage S5
self-play. Research only; must not block the product.

---

## 6. Tooling

**`kloppy` is the answer to the format problem.** It loads 9 tracking providers —
HawkEye, Metrica, PFF, SecondSpectrum, Signality, SkillCorner, Sportec, StatsPerform,
Tracab — into one data model. Every dataset above except the video sets loads through
it.

This is effectively the L0 source interface (PLAN.md §L0) already written. Use it, and
write your canonical state representation on top of kloppy's model rather than parsing
CSVs yourself.

Supporting libraries:

| Library | Purpose |
|---|---|
| `kloppy` | Unified tracking + event loading |
| `socceraction` | VAEP implementation |
| `mplsoccer` | Pitch plotting, visualization |
| `floodlight` | Sports data analysis, includes IDSSE loaders |
| `statsbombpy` | StatsBomb open data access |
| `databallpy` | Tracking/event synchronisation |

---

## 7. Volume reality check

| Category | Matches |
|---|---|
| Full optical tracking | ~10 |
| Broadcast tracking | ~74 |
| **Total continuous tracking** | **~84** |

At 10Hz, ~84 matches is roughly 4.5M frames. Sampled as 2-second windows at 0.5s
stride: ~900k windows. With 4x symmetry augmentation (MODEL.md §3.3): **~3.6M training
samples**, each carrying 23 agents x T timesteps of dense per-agent supervision.

That is a comfortable corpus for a 2-10M parameter model — considerably better than the
"dozens of matches" assumption the original plan was sized against, mainly because of
the PFF World Cup release.

**It is still not enough to justify going bigger.** Frames remain heavily
autocorrelated, and 84 matches is ~84 independent tactical contexts, not 3.6M. Hold the
model size guidance in MODEL.md §3.1.

---

## 8. Suggested first week

1. `pip install kloppy socceraction mplsoccer statsbombpy`
2. Load Metrica game 1 via kloppy — smallest, cleanest, fully documented
3. Build the canonical state representation (MODEL.md §4) on top of kloppy's model
4. Implement attack-direction normalization and mirror augmentation
5. Render a replay in your own 2D renderer and watch it back
6. Submit the PFF access request in parallel — it is a form, and the 64 matches are
   the backbone of the corpus, so start the clock early

Step 6 first, actually. Everything else can proceed while it processes.
