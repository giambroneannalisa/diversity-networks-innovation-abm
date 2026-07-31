#!/usr/bin/env python3
"""
run_morris.py — Morris elementary-effects screening for the ten structural
parameters of Urban_Innovation_Model_v5.2-SA.

Design: r trajectories over k=10 factors at p=4 levels (Delta = 2/3 in the
unit cube), each trajectory = k+1 model points, each point run with
N_REPLICATES NetLogo replicates under deterministic seeding
(SeedSequence([sa_seed, point_index])). The four policy decision variables
are held fixed at a reference point. Everything is a pure function of
--sa-seed: same command, same results.

Usage:
    python3 run_morris.py <config.json> [--r 20] [--sa-seed 20260731]
                          [--reference knee|high-cdr] [--out morris_points.csv]

The config is the same nsga2_config file (NETLOGO_PATH, MAX_TICKS,
N_REPLICATES are read from it; MODEL_PATH is overridden to the SA model).
"""
import csv, json, os, random, subprocess, sys
import multiprocessing
from functools import partial

import numpy as np

MODEL_SA = "Urban_Innovation_Model_v5.2-SA.nlogo"

# ---- the ten Morris factors: (netlogo-global, lo, hi, integer?) ----------
# Lower bounds sit strictly above 0 because the model treats 0 as "not set"
# (BehaviorSpace-restored values of exactly 0 would be replaced by defaults),
# except reinforcement-threshold whose default IS 0 and accepts it.
FACTORS = [
    ("coordination-cost-coefficient", 0.05, 1.00, False),
    ("kpf-gamma-cd",                  0.15, 0.55, False),
    ("mutation-prob",                 0.001, 0.02, False),
    ("jump-prob-base",                0.02, 0.50, False),
    ("rescue-threshold",              0.02, 0.50, False),
    ("rescue-jump-max",               0.20, 1.00, False),
    ("initial-cultures",              4,    16,   True),
    ("ws-rewiring-prob",              0.01, 0.50, False),
    ("knowledge-spillover-radius",    2,    6,    True),
    ("reinforcement-threshold",       0.0,  0.50, False),
]
K = len(FACTORS)
P_LEVELS = 4
DELTA = P_LEVELS / (2.0 * (P_LEVELS - 1))       # 2/3

REFERENCES = {                                   # pooled-knee coordinates
    "knee":     {"cultural-diffusion-rate": 0.003, "bridging-capital-weight": 0.41,
                 "innovation-diffusion-rate": 0.09, "policy-effectiveness": 0.72},
    "high-cdr": {"cultural-diffusion-rate": 0.20, "bridging-capital-weight": 0.41,
                 "innovation-diffusion-rate": 0.09, "policy-effectiveness": 0.72},
}

EXPERIMENT_XML = """
<experiments>
  <experiment name="morris_run" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="{ticks}"/>
    <metric>total-innovation-output</metric>
    <metric>cultural-diversity-index</metric>
    <metric>gini-coefficient</metric>
    <enumeratedValueSet variable="random-seed">
      <value value="{seed}"/>
    </enumeratedValueSet>
    {enumerated_values}
  </experiment>
</experiments>
"""


def scale(name, lo, hi, integer, u):
    v = lo + u * (hi - lo)
    return float(round(v)) if integer else v


def derived_globals(pt):
    """Tied parameters the model expects alongside the ten factors."""
    return {
        "kpf-beta-hc":         0.65 - pt["kpf-gamma-cd"],
        "rescue-jump-mid":     (pt["jump-prob-base"] + pt["rescue-jump-max"]) / 2.0,
        "rescue-threshold-lo": pt["rescue-threshold"] / 2.0,
    }


def morris_sample(r, rng):
    """r trajectories; returns (points, moves). points[i] = dict of unit-cube
    coords; moves[i] = (traj, step, factor_index, signed_delta) with step 0 =
    trajectory base point (no factor changed)."""
    points, moves = [], []
    base_levels_up = [0.0, 1.0 / 3.0]            # step +Delta stays in [0,1]
    for t in range(r):
        signs = [rng.choice((1, -1)) for _ in range(K)]
        x = []
        for i in range(K):
            b = rng.choice(base_levels_up)
            x.append(b if signs[i] > 0 else 1.0 - b)
        order = list(range(K))
        rng.shuffle(order)
        points.append(list(x)); moves.append((t, 0, -1, 0.0))
        for step, fi in enumerate(order, start=1):
            x = list(x)
            x[fi] += signs[fi] * DELTA
            assert -1e-9 <= x[fi] <= 1 + 1e-9
            points.append(list(x)); moves.append((t, step, fi, signs[fi] * DELTA))
    return points, moves


def run_one(config, tasks_item):
    point_id, params, netlogo_seed, replicate = tasks_item
    uid = f"{os.getpid()}_{point_id}_{replicate}_{netlogo_seed}"
    lines = "".join(
        f'<enumeratedValueSet variable="{k}"><value value="{v}"/></enumeratedValueSet>\n'
        for k, v in params.items())
    xml = EXPERIMENT_XML.format(ticks=config["MAX_TICKS"], seed=netlogo_seed,
                                enumerated_values=lines)
    xf, cf = f"temp_{uid}.xml", f"temp_{uid}.csv"
    try:
        with open(xf, "w") as f:
            f.write(xml)
        cmd = [config["NETLOGO_PATH"], "--headless", "--model", config["MODEL_PATH"],
               "--setup-file", xf, "--table", cf]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=300)
        rows = list(csv.reader(open(cf)))
        header_i = next(i for i, r in enumerate(rows) if "[run number]" in (r[0] if r else ""))
        header = [c.strip('"') for c in rows[header_i]]
        data = rows[-1]
        get = lambda name: float(data[header.index(name)])
        return (point_id, replicate, netlogo_seed,
                get("total-innovation-output"),
                get("cultural-diversity-index"),
                get("gini-coefficient"))
    except Exception as e:
        sys.stderr.write(f"[fail] point {point_id} rep {replicate}: {e}\n")
        return (point_id, replicate, netlogo_seed, None, None, None)
    finally:
        for f in (xf, cf):
            if os.path.exists(f):
                os.remove(f)


def main():
    cfg_path = sys.argv[1]
    def opt(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    r = int(opt("--r", 20))
    sa_seed = int(opt("--sa-seed", 20260731))
    reference = opt("--reference", "knee")
    out_path = opt("--out", "morris_points.csv")

    config = json.load(open(cfg_path))
    config["MODEL_PATH"] = MODEL_SA
    n_rep = int(config.get("N_REPLICATES", 3))
    fixed = REFERENCES[reference]

    rng = random.Random(sa_seed)
    points, moves = morris_sample(r, rng)
    print(f"Morris: k={K} factors, r={r} trajectories, p={P_LEVELS}, Delta={DELTA:.4f}"
          f" -> {len(points)} points x {n_rep} replicates = {len(points) * n_rep} runs"
          f" | reference={reference} sa_seed={sa_seed}")

    tasks = []
    for pid, unit in enumerate(points):
        pt = {name: scale(name, lo, hi, integer, unit[i])
              for i, (name, lo, hi, integer) in enumerate(FACTORS)}
        full = dict(fixed); full.update(pt); full.update(derived_globals(pt))
        seeds = [int(s % 2147483646) + 1 for s in
                 np.random.SeedSequence([sa_seed, pid]).generate_state(n_rep, dtype=np.uint64)]
        for rep in range(n_rep):
            tasks.append((pid, full, seeds[rep], rep))

    n_cpu = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_cpu) as pool:
        results = pool.map(partial(run_one, config), tasks)

    by_point = {}
    for pid, rep, nseed, inn, div, gin in results:
        by_point.setdefault(pid, []).append((rep, nseed, inn, div, gin))

    fac_names = [f[0] for f in FACTORS]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_id", "trajectory", "step", "factor_changed", "signed_delta"]
                   + fac_names + ["replicate", "netlogo_seed",
                                  "innovation", "diversity", "gini"])
        for pid, unit in enumerate(points):
            t, step, fi, sd = moves[pid]
            fac = fac_names[fi] if fi >= 0 else "base"
            vals = [scale(n, lo, hi, integer, unit[i])
                    for i, (n, lo, hi, integer) in enumerate(FACTORS)]
            for rep, nseed, inn, div, gin in sorted(by_point.get(pid, [])):
                w.writerow([pid, t, step, fac, sd] + vals + [rep, nseed, inn, div, gin])
    n_fail = sum(1 for x in results if x[3] is None)
    print(f"wrote {out_path} ({len(points)} points); failed runs: {n_fail}")
    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
