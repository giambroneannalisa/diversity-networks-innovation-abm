#!/usr/bin/env python3
"""
analyze_fronts.py — Cross-seed analysis of NSGA-II Pareto fronts.

Recomputes, for any number of seeds, the statistics reported in Chapter IV
of the thesis, using the exact conventions verified against the original
3-seed data:

  - Pearson correlations, per seed and pooled across fronts
  - CDR regime partition at 0.01 / 0.05 (low / mid / high), pooled means
  - Knee points: per seed, the front solution maximising the
    equally-weighted sum of the three objectives, each min-max normalised
    within that seed's front (diversity max, innovation max, Gini min)
  - Cross-seed dispersion (mean, SD, CV) of the knee decision variables

Usage:
    python3 analyze_fronts.py <directory-with-pareto_seed*.csv>

No dependencies beyond the standard library.
"""
import csv, glob, math, os, re, sys

PARAMS = ['bridging-capital-weight', 'innovation-diffusion-rate',
          'policy-effectiveness', 'cultural-diffusion-rate']


def load(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get('Obj_Gini')]


def col(rows, name):
    return [float(r[name]) for r in rows]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (math.sqrt(sum((x - mx) ** 2 for x in xs))
           * math.sqrt(sum((y - my) ** 2 for y in ys)))
    return num / den if den else float('nan')


def objectives(rows):
    innov = [-v for v in col(rows, 'Obj_Innov_Neg')]
    div = [-v for v in col(rows, 'Obj_Div_Neg')]
    gini = col(rows, 'Obj_Gini')
    return innov, div, gini


def knee(rows):
    """Max equally-weighted sum of min-max normalised objectives (§IV.5)."""
    innov, div, gini = objectives(rows)

    def mm(v, invert=False):
        lo, hi = min(v), max(v)
        if hi == lo:
            return [0.0] * len(v)
        return [(hi - x) / (hi - lo) if invert else (x - lo) / (hi - lo) for x in v]

    score = [a + b + c for a, b, c in zip(mm(innov), mm(div), mm(gini, invert=True))]
    return score.index(max(score))


def main(directory):
    files = sorted(glob.glob(os.path.join(directory, 'pareto_seed*.csv')),
                   key=lambda p: int(re.search(r'seed(\d+)', p).group(1)))
    if not files:
        sys.exit(f"No pareto_seed*.csv files found in {directory}")
    fronts = {int(re.search(r'seed(\d+)', p).group(1)): load(p) for p in files}
    print(f"Loaded {len(fronts)} fronts: " +
          ", ".join(f"seed {s} (n={len(r)})" for s, r in fronts.items()))

    # --- Per-seed and pooled correlations -------------------------------
    pairs = [('CDR', 'cultural-diffusion-rate'), ('BCW', 'bridging-capital-weight'),
             ('IDR', 'innovation-diffusion-rate'), ('PE', 'policy-effectiveness')]
    objs = ['Innov', 'Div', 'Gini']

    print("\n=== Per-seed Pearson correlations (decision variable × objective) ===")
    header = "seed  " + "  ".join(f"{p}-{o:<5s}" for p, _ in pairs for o in objs)
    print(header)
    per_seed = {}
    for s, rows in fronts.items():
        innov, div, gini = objectives(rows)
        vals = []
        for _, pcol in pairs:
            x = col(rows, pcol)
            vals += [pearson(x, innov), pearson(x, div), pearson(x, gini)]
        per_seed[s] = vals
        print(f"{s:<5d} " + "  ".join(f"{v:+.3f}   " for v in vals))

    labels = [f"{p}-{o}" for p, _ in pairs for o in objs]
    print("\nCross-seed mean ± SD:")
    for i, lab in enumerate(labels):
        vs = [per_seed[s][i] for s in fronts]
        m = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs))
        print(f"  {lab:<10s} {m:+.3f} ± {sd:.3f}")

    print("\n=== Objective-objective correlations per seed ===")
    for s, rows in fronts.items():
        innov, div, gini = objectives(rows)
        print(f"  seed {s}: Innov-Gini {pearson(innov, gini):+.3f} | "
              f"Innov-Div {pearson(innov, div):+.3f} | Div-Gini {pearson(div, gini):+.3f}")

    pooled = [r for rows in fronts.values() for r in rows]
    innov, div, gini = objectives(pooled)
    print(f"\n=== Pooled front (n={len(pooled)}) ===")
    for lab, pcol in pairs:
        x = col(pooled, pcol)
        print(f"  {lab}: Innov {pearson(x, innov):+.3f} | Div {pearson(x, div):+.3f} | "
              f"Gini {pearson(x, gini):+.3f}")

    # --- CDR regimes ----------------------------------------------------
    print("\n=== CDR regimes on pooled front (low < 0.01, mid 0.01-0.05, high >= 0.05) ===")
    regs = {'low': [], 'mid': [], 'high': []}
    for r in pooled:
        c = float(r['cultural-diffusion-rate'])
        regs['low' if c < 0.01 else 'mid' if c < 0.05 else 'high'].append(r)
    for name, rows in regs.items():
        if not rows:
            print(f"  {name:<5s} n=0")
            continue
        i, d, g = objectives(rows)
        print(f"  {name:<5s} n={len(rows):<4d} innov {sum(i)/len(i):>12,.0f}   "
              f"div {sum(d)/len(d):.3f}   gini {sum(g)/len(g):.4f}")

    # --- Knee points ----------------------------------------------------
    print("\n=== Knee points (max normalised-sum, §IV.5 convention) ===")
    global_imax = max(-v for rows in fronts.values() for v in col(rows, 'Obj_Innov_Neg'))
    kvals = {p: [] for p in PARAMS}
    for s, rows in fronts.items():
        k = knee(rows)
        innov, div, gini = objectives(rows)
        for p in PARAMS:
            kvals[p].append(col(rows, p)[k])
        print(f"  seed {s}: CDR={col(rows, 'cultural-diffusion-rate')[k]:.4f}  "
              f"BCW={col(rows, 'bridging-capital-weight')[k]:.3f}  "
              f"IDR={col(rows, 'innovation-diffusion-rate')[k]:.3f}  "
              f"PE={col(rows, 'policy-effectiveness')[k]:.3f}  |  "
              f"div {div[k]/max(div)*100:.1f}% of front max  "
              f"innov {innov[k]/max(innov)*100:.1f}% of front max "
              f"({innov[k]/global_imax*100:.1f}% of global max)  "
              f"gini {gini[k]:.4f}")

    print("\n  Cross-seed dispersion of knee decision variables:")
    for p in PARAMS:
        vs = kvals[p]
        m = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs))
        cv = sd / m * 100 if m else float('nan')
        print(f"    {p:<28s} mean {m:.4f}  SD {sd:.4f}  CV {cv:.0f}%")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
