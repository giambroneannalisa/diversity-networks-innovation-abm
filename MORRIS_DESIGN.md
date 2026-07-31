# Morris screening — protocol

Global sensitivity screening of the structural parameters that the thesis
experiment holds fixed. The NSGA-II analysis explores four policy decision
variables inside a ~65-dimensional parameter space whose remaining
coordinates were set by hand (several of them explicitly tuned so that the
cultural-diffusion rate would be the binding lever — see FIX 28/30b/32 in
the model changelog). This experiment measures how much the model's three
outputs depend on ten of those hand-set coordinates.

## How to launch

GitHub → **Actions** → **Morris screening** → *Run workflow*. Defaults
(r = 20, reference = knee, sa_seed = 20260731) reproduce the canonical
design: 20 trajectories × 11 points × 3 replicates = **660 NetLogo runs**,
roughly 1–2 hours on one hosted runner.

The workflow refuses to run the screening unless the **equivalence job**
passes first: with no SA parameter set, `Urban_Innovation_Model_v5.2-SA`
must produce bit-identical BehaviorSpace output to `v5.1` under the same
seed. This proves the SA variant changed *parameterisation*, not behaviour.

## The model variant (v5.2-SA)

`Urban_Innovation_Model_v5.2-SA.nlogo` differs from v5.1 only in that ten
hard-coded values become BehaviorSpace-settable globals with guarded
defaults equal to the old literals. Three additional tied globals
(`kpf-beta-hc`, `rescue-jump-mid`, `rescue-threshold-lo`) exist so that
default arithmetic stays bit-identical; the driver derives them:
β = 0.65 − γ (constant returns), mid-jump = (base + max)/2,
low threshold = threshold/2. A value of 0 means "not set" for every SA
global except `reinforcement-threshold`, whose default is genuinely 0 —
hence SA ranges start strictly above 0.

## Factors and ranges

| # | factor (NetLogo global) | default (v5.1) | range | why it is in the screening |
|---|---|---|---|---|
| 1 | `coordination-cost-coefficient` | 0.5 | [0.05, 1.0] | fixes the KPF diversity hump's optimum (≈0.65); lower bound ≈ cost off |
| 2 | `kpf-gamma-cd` | 0.35 | [0.15, 0.55] | the thesis's headline elasticity; β compensates to keep Σ=1 |
| 3 | `mutation-prob` | 0.005 | [0.001, 0.02] | FIX 30b hand-tuned (was 0.02) so CDR dominates the diversity equilibrium |
| 4 | `jump-prob-base` | 0.2 | [0.02, 0.5] | baseline share of cultural mutations that are global jumps |
| 5 | `rescue-threshold` | 0.3 | [0.02, 0.5] | diversity thermostat trigger; lower bound ≈ thermostat off |
| 6 | `rescue-jump-max` | 0.8 | [0.2, 1.0] | thermostat strength at full alarm |
| 7 | `initial-cultures` | 8 | {4…16} | cardinality of the cultural space (integer) |
| 8 | `ws-rewiring-prob` | 0.10 | [0.01, 0.5] | small-world shortcut share of the household network |
| 9 | `knowledge-spillover-radius` | 3 | {2…6} | spatial reach of firm diversity and knowledge pools (integer) |
| 10 | `reinforcement-threshold` | 0 | [0, 0.5] | simple → complex contagion dial (FIX 32 set it to 0) |

Excluded: `imitation-prob` — dead code since FIX 30 replaced its gate with
CDR·0.5; population sizes and MAX_TICKS — scale parameters better treated
in a dedicated scaling study.

## Method

Morris elementary effects, p = 4 levels, Δ = 2/3 in the unit cube,
r trajectories of k+1 = 11 points (one-at-a-time steps in random order,
random start, random step sign). Every point runs N_REPLICATES = 3 NetLogo
replicates whose seeds derive from `SeedSequence([sa_seed, point_index])` —
the entire experiment, design included, is a pure function of `sa_seed`.

The four decision variables are frozen at a reference point:

- `knee` (default): CDR 0.003, BCW 0.41, IDR 0.09, PE 0.72 — the pooled
  knee of the 13-front analysis; sensitivities *of the thesis's compromise
  solution*.
- `high-cdr`: CDR 0.20, others equal — sensitivities of the
  innovation-heavy regime. Run both to check whether the ranking is
  regime-dependent.

Outputs per run: `total-innovation-output`, `cultural-diversity-index`,
`gini-coefficient` at t = 300.

`analyze_morris.py` reports, per output and factor, the elementary-effect
statistics **μ\*** (mean |EE|: overall importance), **μ** (signed mean:
direction), **σ** (spread: σ > μ\* flags nonlinearity/interactions), with
effects expressed per full swing of each factor's range so they are
directly comparable across factors.

## Decision rule and next phase

Factors with μ\* below ~10% of the largest μ\* on every output are declared
negligible and frozen at defaults. The survivors (expected: a handful) go to
a **Sobol** variance decomposition via Saltelli sampling — N(k′+2) points
with N = 1024 — run as a matrix workflow if k′ makes it heavy. The two
questions Sobol must answer: what share of output variance the
coordination-cost coefficient owns (it fixes the diversity hump the
CDR–innovation result rides on), and whether the thermostat parameters
interact with `mutation-prob` in setting the measured width of the
diversity–innovation trade-off.

## Validation performed before first launch

The driver and analyzer were exercised end-to-end against a mock simulator
implementing a known linear function: recovered μ\* matched the analytic
coefficients (95.1 vs 95 expected; 24.1 vs 24.5; 0.608 vs 0.608 with
correct sign; exact zeros on a constant output). The equivalence job
guards the model side on every run.
