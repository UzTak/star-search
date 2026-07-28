# Star: Global interplanetary mission design tool 

**Star** [1] is a broad-search algorithm for patched-conic interplanetary mission design. 
The unique advantage of this algorithm is that the search time is polynomial by adopting the philosophy of "patch pre-generated Lambert arcs" via powered flyby (i.e., $\Delta V$ at the patch point) with the discretized encounter epochs with each planetary body. With additional heuristics, Star enables a scalable broad-search for multi-flyby trajectories and serves as a strong preliminary trajectory design/analysis tool. 

This codebase summarizes the three algorithms shown here: (Star search logic [1], Efficient mid-arc maneuver placement algorithm [2], and Lambert solver [3]). 

[1] Landau, D., Campagnola, S., & Pellegrini, E. (2022). Star searches for patched-conic trajectories. The Journal of the Astronautical Sciences, 69(6), 1613-1648.

[2] Landau, D. (2018). Efficient maneuver placement for automated trajectory design. Journal of Guidance, Control, and Dynamics, 41(7), 1531-1541.

[3] Arora, N., Russell, R.P. (2013). A fast and robust multiple revolution Lambert algorithm using a cosine transformation. AAS/AIAA Astrodynamics Specialist Conference, AAS 13-728, Hilton Head, SC. 

## Quick Start

### 1. Installation

This repo targets Python 3.10+ and installs the `star-search` distribution,
which exposes the import package `star`. It includes a `uv.lock` file, so the
shortest setup is:

```powershell
uv sync
uv pip install -e .  # import as a package 
```

Run commands from the repo root. Bare problem names such as `earth_to_jupiter` resolve to modules under `example/`.

### 2. Set up SPICE (meta-)kernel

Star reads planetary ephemerides through [SPICE](https://naif.jpl.nasa.gov/naif/), so you need to download the kernels once and point Star at them.

1. Download the generic kernels you need from NAIF, mirrored under
   <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/>:

   | Kernel | Purpose |
   | --- | --- |
   | `lsk/naif0012.tls` | Leapseconds (required) |
   | `spk/planets/de440.bsp` | Planetary barycenters (required) |
   | `spk/satellites/mar099.bsp` | Mars (499) |
   | `spk/satellites/jup365.bsp` | Jupiter (599) and the Galilean moons |
   | `spk/satellites/sat457.bsp` | Saturn (699) |
   | `spk/satellites/ura184_part-1.bsp`, `spk/satellites/ura111xl-799.bsp` | Uranus (799) |


2. Copy the template and edit the kernel directory:

   ```powershell
   copy star\METAKERN.tm.template star\METAKERN.tm
   ```

   Then set `PATH_VALUES` in `star/METAKERN.tm` to the absolute path of your
   kernel directory (forward slashes on Windows) and trim `KERNELS_TO_LOAD` to
   the kernels you actually downloaded.

`star/METAKERN.tm` is git-ignored, so your local paths stay out of version control. Every entry point takes `--metakernel` if you keep your kernels elsewhere.

Star only calls `furnsh`, `str2et`, and `spkezr`; GM and radius values come from tables in `star/constants/`, not from a PCK.

## Examples

0. Optional: estimate memory pressure before a full run.

```powershell
uv run python memory_report.py --problem earth_to_jupiter
```

1. Run the search pipeline and write solutions to `output/earth_to_jupiter.jsonl`.

```powershell
uv run python star.py --problem earth_to_jupiter --metakernel "star/METAKERN.tm"
```

2. Interactive plotting of the saved Pareto front. If you click a solution point, the corresponding trajectory plot is generated. 

```powershell
uv run python plotter.py --problem earth_to_jupiter 
```

The plotter exposes several command-line options; see `star-plot --help` for details.

## Reference runtimes

Rough sense of scale for the bundled problems.
Measured on a 12th Gen Intel Core i7-1260P (12 cores / 16 threads), 32 GB RAM,
Windows 11, Python 3.11, one run each with a warm Numba cache. The first run on a new machine is slower because Numba compiles the JIT kernels.

| Problem | Encounters | Largest leg DB | Largest flyby DB | Trajectories (post tfilter) | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test1_DVEGA` | 4,710 | 8.8 M | 0.07 M | 5,952 | ~74 s |
| `test2_EMEJ` | 6,227 | 0.3 M | - | 33 | ~12 s |
| `earth_to_jupiter` | 3,740 | 6.9 M | 11.5 M | 408 | ~104 s |
| `earth_to_saturn` | 27,330 | 15.3 M | 36.8 M | 459 | ~14 min |
| `earth_to_uranus` | 949 | 38 K | 0.4 M | 752 | ~20 s |
| `jovian_petal` | 14,008 | 52.6 M | 735.1 M | 114,702 | ~3 h 45 min |
| `bepicolombo` | 23,720 | 14.0 M | 102.6 M | 941,219 | ~40 min |

The two levers that matter most are the per-encounter grid step `dt` and the DSM
leveraging grid (`dvlev_max` / `delta_dvlev`); both multiply the leg-database row
count directly. Use `memory_report.py --problem <name>` to estimate before
committing to a long run.

## Notation

The index and variable names throughout the source (`IE`, `IL`, `IF`, `dv_lev`,
`eta_lev`, `tfilter`, "null leg", ...) follow the original paper's conventions [1].
See [NOTATION.md](NOTATION.md) for a glossary before reading the code.

## Tests

The `/tests/` directory contains regression tests for core numerical and
pipeline behavior.

```powershell
uv run pytest
```

## Acknowledgement

Special thanks to Star authors (Damon Landau, Stefano Campagnola, and Etienne Pellegrini) for insightful discussions and cross-validations of preliminary results. 
AI coding tools are used extensively to make this codebase. 

## License

MIT - see [LICENSE](LICENSE).

