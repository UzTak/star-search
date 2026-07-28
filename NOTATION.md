# Notation

Star follows the notation of the [original paper](https://doi.org/10.1007/s40295-022-00350-y).
The source is dense with two-letter index names that are hard to read without
that paper, so this page defines them. Read it before reading the code.

---

## Core idea

A trajectory is a chain of **encounters** (arrivals at a body at an epoch)
joined by **legs** (heliocentric arcs). Star discretizes each encounter into a
grid of candidate (body, epoch) nodes, pre-solves every feasible Lambert arc
between adjacent grids, evaluates the powered flyby that patches each
consecutive leg pair, and then stitches the survivors into full trajectories.
Because arcs are pre-generated and reused, the search cost is polynomial rather
than combinatorial.

```
EncounterDB  ->  LegDatabase  ->  FlybyDB  ->  SegmentDB  ->  OutputDB
  (nodes)        (arcs)        (patches)      (chains)      (trajectories)
```

> **Reference vs. shipped implementation.** The paper's §5.1 *triplet* formulation
> survives as reference code (`build_triplet_dbs`, `init_segment_dbs`,
> `combine_segments`) but has no callers — the shipped path is a streaming
> left-fold in `run_combo`. This is a deliberate design due to memory management: `SegmentDB` rows carry their
> full history (`leg_ILs` is `(N, n_legs)`), so width grows with depth and shared
> prefixes are duplicated, forcing the whole cross-product into RAM.
> `_ComboStageRows` instead keeps 6 fixed columns, replacing the history with a
> `parent_row` pointer into the previous stage's append-only `.npy` spool: O(1)
> row width, each prefix stored once, peak RAM of one stage's frontier. Full
> paths are rebuilt once, for survivors only, in
> `_reconstruct_segment_from_combo_spools`. Trace `run_combo`, not the triplets.

---

## Stages

A problem with `nE` encounters has `nL = nE - 1` legs. "Stage" indexes three
different things; the code disambiguates with a prefix.

| Term | Meaning |
| --- | --- |
| **encounter stage** | Position in the body sequence, `0 … nE-1`. Each has its own body list, epoch window, and time step. |
| **leg stage** | Position in the leg sequence, `0 … nL-1`. Leg stage `i` connects encounter stage `i` to `i+1`. `leg_stage_id == dep_stage_id`. |
| **flyby stage** | A powered flyby at an interior encounter, joining leg stage `i-1` to leg stage `i`. |

`LegBuildConfig` carries `leg_stage_id`, `dep_stage_id`, and `arr_stage_id`
side by side for exactly this reason.

## Index symbols

These are **row indices into a specific database**, not physical quantities.

| Symbol | Indexes | Scope |
| --- | --- | --- |
| `IE` | `EncounterDB.entries` — one encounter node | **Global.** Unique across the whole run, assigned in deterministic `(t_et, IE)` order from 0. |
| `IL` | `LegDatabase` — one candidate arc | **Per leg stage.** `IL = 7` in stage 2 and `IL = 7` in stage 3 are unrelated rows. |
| `IF` | `FlybyDB` — one powered-flyby solution | **Per flyby stage.** |
| `IT` | `TripletDB` — one (leg, flyby, leg) triplet | Per flyby stage. **Reference only** — never constructed on the shipped path (see the note above). |
| `IS` | `SegmentDB` — one partial chain of legs | Per segment. |

The live per-stage structure in the streaming fold is `_ComboStageRows`
(`star/combo.py`), which holds the same leg/flyby indices plus the parent links
used to backtrack a finished chain out of the on-disk spools.

> **The single most common mistake:** `IE` is globally unique, `IL`/`IF`/`IT`/`IS`
> are stage-scoped. Any code holding an `IL` must also know which stage it came
> from. `LegDatabase.leg_id` is just an alias for `IL`.

Within a `LegDatabase` row, the two endpoints are named separately:

| Symbol | Meaning |
| --- | --- |
| `ID` | **D**eparture encounter index — an `IE`, *not* an identifier. |
| `IA` | **A**rrival encounter index — an `IE`. |

## Physical quantities

Units are carried in the suffix (`_km`, `_km_s`, `_km_s2`, `_s`, `_deg`).

| Symbol | Meaning |
| --- | --- |
| `tof` | Time of flight — elapsed seconds between two consecutive encounters. |
| `vinf` | v-infinity — spacecraft velocity relative to the body at an encounter [km/s]. `vinfD_km_s` at departure, `vinfA_km_s` at arrival, both `(N, 3)`. |
| `dv` | Impulsive delta-v magnitude [km/s]. |
| `amin_km` | Minimum flyby altitude above the body surface [km], per body. |
| `mu_km3_s2` | Gravitational parameter. Defaults to the Sun's for heliocentric legs. |

## DSM leveraging

A **deep-space maneuver (DSM)** placed mid-arc lets a leg trade v-infinity
between its two ends — the "leveraging" of the paper.

| Symbol | Meaning |
| --- | --- |
| `dv_lev_km_s` | Applied DSM magnitude on this leg row [km/s]. `0` means a pure Lambert (ballistic) leg. |
| `eta_lev` | Where the DSM sits along the arc, as a fraction of `tof`. `0` = departure, `1` = arrival. |
| `dvlev_max_km_s` | Config: largest DSM to try. `<= 0` disables DSM expansion for that leg. |
| `delta_dvlev_km_s` | Config: step size of the DSM magnitude grid. |
| `lev_type` | Which end to raise and which to lower. A two-character sign pair `(departure, arrival)`: `"+-"` raises departure v-infinity and lowers arrival, `"-+"` the reverse; `"++"` and `"--"` move both the same way. |

Note the spelling drift: configuration fields use `dvlev_*`, while the stored
data column is `dv_lev_km_s`.

## Lambert solver

| Symbol | Meaning |
| --- | --- |
| `n_rev` | Signed revolution count. Sign distinguishes the two multi-revolution branches; `0` is the direct transfer. |
| `lambert_nrev_max` | Config: solver sweeps `0 … n` revolutions. |
| `lambert_hz` | Transfer direction, from the sign of the angular momentum ẑ-component: `+1` counter-clockwise, `-1` clockwise, `0` try both. (Nothing to do with frequency.) |
| `obj_type` | Maneuver-placement objective: `1` = minimum v-infinity, `2` = minimum time of flight. |

## Search-control vocabulary

| Term | Meaning |
| --- | --- |
| **tfilter** | Time-bin decimation. Rows are binned by encounter epoch and body sequence, and only the minimum-`dv` row per bin survives. This is what keeps the candidate count bounded. |
| **preemptive tfilter** | tfilter applied per stage *during* the combination fold, before the row count explodes. Controlled by `dt_filter_preemptive_s`. |
| **final-output tfilter** | tfilter applied once at the end, to completed trajectories (`tfilter_output_db`). |
| **null leg** | A leg of zero duration, used to let a trajectory *skip* an encounter. Requires that encounter stages `i` and `i+1` share a node with identical `(body, epoch)`; the null leg carries that node forward instead of flying a heliocentric arc. |
| **resonant leg** | A leg expanded off a near-resonant Lambert seed, where the spacecraft returns to the same body after an integer number of body revolutions. |
| **leg filter / fixpoint** | Forward-and-backward pruning: a leg row survives only if it has at least one feasible flyby on both sides. Pruning one stage can invalidate a neighbor, so it is iterated to a fixpoint. |

## Time conventions

| Convention | Where |
| --- | --- |
| **ET seconds past J2000** (`t_et`, `*_et_s`) | Everywhere internally. This is what SPICE returns from `str2et`. |
| **Days past J2000** | Problem modules in `example/` only. |
| **UTC strings** | Problem modules and printed output. |

Conversions live in `star/constants/time.py`. Mixing ET seconds with J2000 days
is the most likely silent error for a new contributor — a factor of 86400 will
not raise, it will just produce nonsense trajectories.
