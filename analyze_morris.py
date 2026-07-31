#!/usr/bin/env python3
"""
analyze_morris.py — elementary effects (mu*, mu, sigma) from morris_points.csv.

For each trajectory, consecutive points differ in exactly one factor by a
known signed step in the unit cube; the elementary effect is the change in
the replicate-mean output divided by that step, so effects are expressed per
full swing of the factor's range and are comparable across factors.

Usage:  python3 analyze_morris.py [morris_points.csv]
"""
import csv, math, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "morris_points.csv"
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit("no rows")

OUTPUTS = ("innovation", "diversity", "gini")

# replicate-mean outputs per point
pts = {}
for r in rows:
    pid = int(r["point_id"])
    d = pts.setdefault(pid, {"traj": int(r["trajectory"]), "step": int(r["step"]),
                             "factor": r["factor_changed"],
                             "delta": float(r["signed_delta"]), "vals": []})
    if r["innovation"] not in ("", "None"):
        d["vals"].append(tuple(float(r[o]) for o in OUTPUTS))

mean = {pid: tuple(sum(v[i] for v in d["vals"]) / len(d["vals"]) for i in range(3))
        for pid, d in pts.items() if d["vals"]}

# elementary effects: walk each trajectory in step order
eff = defaultdict(lambda: defaultdict(list))     # eff[output][factor] = [EE...]
traj = defaultdict(list)
for pid, d in pts.items():
    traj[d["traj"]].append(pid)
for t, pids in traj.items():
    pids.sort(key=lambda p: pts[p]["step"])
    for a, b in zip(pids, pids[1:]):
        if a not in mean or b not in mean:
            continue
        f, delta = pts[b]["factor"], pts[b]["delta"]
        for i, o in enumerate(OUTPUTS):
            eff[o][f].append((mean[b][i] - mean[a][i]) / delta)

for o in OUTPUTS:
    print(f"\n=== {o}: elementary effects per full factor range ===")
    print(f"{'factor':>30} {'n':>3} {'mu*':>12} {'mu':>12} {'sigma':>12}  interpretation")
    stats = []
    for f, es in eff[o].items():
        n = len(es)
        mu = sum(es) / n
        mu_star = sum(abs(e) for e in es) / n
        sd = math.sqrt(sum((e - mu) ** 2 for e in es) / n)
        stats.append((mu_star, mu, sd, n, f))
    top = max(s[0] for s in stats) if stats else 0
    for mu_star, mu, sd, n, f in sorted(stats, reverse=True):
        share = mu_star / top if top else 0
        tag = ("dominant" if share > 0.5 else
               "active" if share > 0.1 else "negligible")
        nl = " nonlinear/interacting" if mu_star and sd > mu_star else ""
        print(f"{f:>30} {n:>3} {mu_star:>12.4g} {mu:>12.4g} {sd:>12.4g}  {tag}{nl}")
print("\nmu* = mean |effect| (importance); mu = signed mean (direction);")
print("sigma > mu* flags nonlinearity or interactions. Effects are per full")
print("range of each factor, holding the four policy variables at the reference.")
