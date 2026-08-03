# Cultural diversity, networks, and innovation in cities — replication package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21562685.svg)](https://doi.org/10.5281/zenodo.21562685)

Replication package for the Ph.D. thesis:

> Giambrone, A. (2026). *Cultural diversity, networks, and innovation in
> cities: an agent-based model of co-evolutionary processes.* Ph.D.
> thesis, University of Enna "Kore". Defended 15 July 2026.
> Supervisor: Prof. Raffaele Scuderi.

The package archives the agent-based model, the NSGA-II optimization
driver, the experiment configuration, and the complete output data of
the three thesis runs (seeds 42, 123, 456 — 4,860 simulation
evaluations), together with the analysis conventions needed to
reproduce the thesis statistics from the archived data.

## Contents

| File | Description |
|---|---|
| `Urban_Innovation_Model_v5.1.nlogo` | NetLogo 6.3.0 agent-based model (v5.1 specification; the header documents the full fix changelog). 200 households, 20 firms, 5 universities, 5 institutions; Watts–Strogatz household network, Barabási–Albert firm network; knowledge production function with cultural-diversity elasticity γ = 0.35 |
| `run_trajectory.py` | NSGA-II driver used for the thesis runs (pymoo → NetLogo headless via BehaviorSpace). Archival: kept exactly as executed |
| `run_trajectory_v2.py` | Post-defence driver with deterministic replicate seeding (same experiment; bit-for-bit reproducible outputs) and per-replicate logging |
| `nsga2_config_v5_1_1.json` | Experiment configuration: bounds of the 4 decision variables, population 50, 50 generations, 3 replicates, 300 ticks |
| `data/pareto_seed{42,123,456}.csv` | Final Pareto fronts (42, 21, 25 solutions) — the primary data of thesis Chapter IV |
| `data/trajectory_seed{42,123,456}.csv` | Full evolutionary trajectories (2,500 rows each = 50 generations × 50 individuals). Each front is exactly the non-dominated subset of its generation 50; no failed evaluations |
| `data/trajectory_seed{42,456}_checkpoint.csv` | Generation-50 checkpoints (byte-identical to the final trajectory files) |
| `analyze_fronts.py` | Cross-seed analysis: recomputes the thesis statistics from the fronts (stdlib only). A later reconstruction, not the original analysis tooling |
| `compare_seeds.py` | Thesis fronts against a second set of fronts: per-seed knee points and correlations, group summaries, bootstrap intervals, CDR-regime table. Keys fronts by (group, seed) so a seed present in both sets is compared, not overwritten |
| `paired_driver_check.py` | Whether a design of this size can resolve a relation at all: mean \|r\| against its within-front sampling error, pooled means with bootstrap intervals, a sensitivity view restricted to distinct master seeds, and the v1-vs-v2 comparison on shared seeds |
| `trajectory_channel_check.py` | Decomposes the CDR-innovation association: the knowledge-production response curve, the correlation on the full evaluation cloud versus on the front, and innovation binned by realised diversity |
| `repro_digest.py` | Byte-level digests of a run's output files, blocked by generation and by evaluation index, so a reproducibility check reports *where* two runs diverge and not merely whether they do |
| `run_all_seeds.sh` | Batch runner for the 10-seed extension experiment |
| `.github/workflows/run-10-seeds.yml` | GitHub Actions workflow running the 10-seed extension on hosted runners |
| `.github/workflows/reproducibility-check.yml` | Re-runs a seed already run and prints its digests, testing the driver's bit-for-bit claim on the same runner image |

## How to reproduce the thesis experiment

Requirements: NetLogo 6.3.0, Python 3.11, `pymoo` 0.6.x, `pandas`, `numpy`.
These are the actual dependencies — the thesis's Appendix D additionally
lists `pyNetLogo`, `scipy` and `matplotlib`, which the code does not use
(erratum 4 below).

1. Set `NETLOGO_PATH` in `nsga2_config_v5_1_1.json` to your
   `netlogo-headless.sh`.
2. Run one process per seed (~4.5–5.5 h each on an i7-12700H):

   ```bash
   python3 run_trajectory.py nsga2_config_v5_1_1.json --seed 42    # thesis driver
   # or, for bit-for-bit reproducible runs:
   python3 run_trajectory_v2.py nsga2_config_v5_1_1.json --seed 42
   ```

3. Verify the published statistics from the archived data:

   ```bash
   python3 analyze_fronts.py data
   ```

### Reproducibility scope

The thesis runs (v1 driver) drew per-replicate NetLogo seeds from an
unseeded generator, so re-running reproduces the thesis results
**statistically, not byte-for-byte** (the operational criterion of
thesis Appendix D.6). The post-processing step — archived CSV files →
published statistics — is fully deterministic and verified: see below.
`run_trajectory_v2.py` closes the gap for future runs by deriving every
replicate seed deterministically from the master seed.

## Data dictionary

All CSV files share these columns:

| Column | Meaning |
|---|---|
| `bridging-capital-weight` (BCW) | decision variable ∈ [0, 1] |
| `innovation-diffusion-rate` (IDR) | decision variable ∈ [0, 0.2] |
| `policy-effectiveness` (PE) | decision variable ∈ [0, 1] |
| `cultural-diffusion-rate` (CDR) | decision variable ∈ [0, 0.5] |
| `Obj_Innov_Neg` | −(total innovation output); negated for pymoo's minimization convention |
| `Obj_Div_Neg` | −(cultural diversity index ∈ [0, 1]); same convention |
| `Obj_Gini` | Gini coefficient of household income (minimized directly) |
| `Generation` | trajectory files only: NSGA-II generation (1–50) |

## Verified analysis conventions

The headline statistics of thesis Chapter IV were independently
recomputed from these exact files and reproduce at the reported
precision. The conventions required:

- **Correlations** are Pearson, on the Pareto fronts (per seed, or
  pooled across the three fronts, n = 88). Verified matches include:
  CDR–Diversity −0.94 (pooled), per-seed mean −0.92; CDR–Innovation
  mean +0.40; Innovation–Gini +0.66 / +0.28 / +0.23 by seed; PE–Gini
  −0.41 / −0.33 / −0.38; IDR–Gini +0.02 / +0.65 / +0.22;
  BCW–Diversity −0.12 (pooled).
- **CDR regimes** (thesis §IV.4) partition the pooled front at
  CDR < 0.01 (low), 0.01 ≤ CDR < 0.05 (mid), CDR ≥ 0.05 (high).
  Regime means reproduce exactly (Gini 0.155 / 0.152 / 0.151;
  innovation 279,179 / 389,273 / 374,934).
- **Knee points** (thesis §IV.5, Table IV.7): per seed, the front
  solution maximizing the equally-weighted sum of the three objectives,
  each min-max normalized within that seed's front. Knee
  decision-variable values reproduce exactly (knee CDR
  0.0011 / 0.0056 / 0.0029; mean 0.003, CV 57%).

### Errata (no conclusion affected)

1. The Table IV.7 knee *innovation shares* (54.4% / 64.9% / 63.1%) are
   computed against the **global** maximum innovation across the three
   fronts (632,393), not each seed's own front maximum as the
   surrounding text states.
2. The seed-42 *diversity share* prints as 99.1% where the within-front
   convention used for the other two seeds gives 99.5%.
3. §IV.2's statement that the CDR–innovation relationship is
   "consistently positive across both the trajectory population and
   the Pareto front" holds robustly on the Pareto fronts
   (+0.30 / +0.41 / +0.49 per seed) but only marginally on the
   trajectories (pooled r = +0.05; per-seed +0.15 / +0.11 / −0.08) —
   it should be read as a Pareto-front property.
4. Appendix D lists `pyNetLogo` 0.5.2 as the Python–NetLogo bridge.
   **The code does not use pyNetLogo.** Both drivers couple to NetLogo by
   writing a BehaviorSpace experiment to a temporary XML file and invoking
   `netlogo-headless` as a subprocess with `--setup-file`, reading results
   back from the `--table` CSV. Install the requirements listed above, not
   pyNetLogo. Appendix D also lists `scipy` and `matplotlib`, which no
   script in this package imports (they were presumably used for the
   thesis figures, outside the replication package). This affects only how
   a replicator prepares the environment; every reported result is
   unchanged.

## Extension study (post-defence)

`run_trajectory_v2.py`, `run_all_seeds.sh` and the GitHub Actions
workflow extend the experiment from 3 to 10 seeds
(42 123 456 + 789 1234 2345 3456 4567 5678 6789) under deterministic
replicate seeding. Because the original replicate seeds were
unrecorded, v2 re-runs of the thesis seeds match the archived data
statistically, not byte-wise. Extension outputs are not part of
release v1.0.0, which archives the package exactly as it corresponds
to the thesis as defended.

## How to cite

See `CITATION.cff` (GitHub renders a "Cite this repository" button).
Cite the thesis for the scientific content and this package's DOI for
the artifacts: version DOI
[10.5281/zenodo.21562686](https://doi.org/10.5281/zenodo.21562686)
for the v1.0.0 thesis-as-defended snapshot, or all-versions DOI
[10.5281/zenodo.21562685](https://doi.org/10.5281/zenodo.21562685)
(always resolves to the latest version).

## Licenses

- Code, model, and documentation: [MIT](LICENSE)
- Data (`data/`): [CC BY 4.0](LICENSE-DATA)
