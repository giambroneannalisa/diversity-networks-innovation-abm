#!/usr/bin/env python3
"""
trajectory_channel_check.py — Decompose the CDR-innovation association.

The Pareto fronts show CDR-innovation at +0.376 (13/13 positive). This script
asks how much of that is a gradient in the underlying response surface and
how much is survivorship on the front, and tests the mechanism the KPF
implies: cultural diversity enters the knowledge production function twice,
as CD^0.35 in the numerator and as a quadratic coordination cost
1 + 0.5*CD^2 in the denominator, giving an interior optimum at
CD* = sqrt(0.35/0.825) ~= 0.651. If runs operate above that value, mild
homogenisation should RAISE innovation - the downhill side of the hump.

Usage:
    python3 trajectory_channel_check.py <thesis-dir> <extension-raw-dir>

Reads trajectory_seed*.csv (full runs) and trajectory_seed*_checkpoint.csv
(runs stopped by the CI time limit, used only when no full file exists).
"""
import csv, glob, math, os, sys


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (math.sqrt(sum((a - mx) ** 2 for a in x))
           * math.sqrt(sum((b - my) ** 2 for b in y)))
    return num / den if den else float('nan')


def main(thesis_dir, raw_dir):
    paths = sorted(glob.glob(os.path.join(thesis_dir, 'trajectory_seed*[0-9].csv')))
    paths += sorted(glob.glob(os.path.join(raw_dir, 'trajectory_seed*[0-9].csv')))
    paths += [p for p in
              sorted(glob.glob(os.path.join(raw_dir, 'trajectory_seed*_checkpoint.csv')))
              if not os.path.exists(p.replace('_checkpoint', ''))]
    if not paths:
        sys.exit('No trajectory files found')

    print("=== KPF diversity response f(CD) = CD^0.35 / (1 + 0.5 CD^2) ===")
    for cd in (0.2, 0.4, 0.6, 0.651, 0.7, 0.8, 0.9, 1.0):
        print(f"  CD={cd:.3f}  f={cd ** 0.35 / (1 + 0.5 * cd * cd):.4f}")
    print("  interior optimum at CD* ~= 0.651; above it, less diversity"
          " means more innovation.")

    print("\n=== CDR-innovation on the evaluation cloud, per run ===")
    print(f"{'run':>28} {'n':>5} {'r all':>7} {'r gen<=5':>9} {'r gen>=40':>10}")
    rs, allpts = [], []
    for p in paths:
        rows = [r for r in csv.DictReader(open(p)) if r.get('Obj_Gini')]
        cdr = [float(r['cultural-diffusion-rate']) for r in rows]
        inn = [-float(r['Obj_Innov_Neg']) for r in rows]
        div = [-float(r['Obj_Div_Neg']) for r in rows]
        gen = [int(r['Generation']) for r in rows]
        r_all = pearson(cdr, inn)
        rs.append(r_all)
        e = [i for i, g in enumerate(gen) if g <= 5]
        l = [i for i, g in enumerate(gen) if g >= 40]
        name = os.path.basename(p).replace('trajectory_', '').replace('.csv', '')
        print(f"{name:>28} {len(rows):>5} {r_all:>+7.3f} "
              f"{pearson([cdr[i] for i in e], [inn[i] for i in e]):>+9.3f} "
              f"{pearson([cdr[i] for i in l], [inn[i] for i in l]):>+10.3f}")
        allpts += list(zip(cdr, inn, div))
    m = sum(rs) / len(rs)
    pos = sum(1 for r in rs if r > 0)
    print(f"\n  mean r = {m:+.3f}   positive in {pos}/{len(rs)} runs"
          f"   range [{min(rs):+.3f}, {max(rs):+.3f}]")
    print("  (the front-level value is +0.376: the front amplifies this weak"
          " gradient\n   by survivorship - a high-CDR configuration is"
          " non-dominated only if it\n   delivers innovation no low-CDR"
          " configuration matches)")

    print(f"\n=== Hump test: mean innovation by realised diversity"
          f" ({len(allpts)} evaluations) ===")
    for lo, hi in [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                   (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
        sub = [i for c, i, d in allpts if lo <= d < hi]
        if sub:
            print(f"  div [{lo:.2f},{hi:.2f}): n={len(sub):>5}  "
                  f"mean innov {sum(sub) / len(sub):>10,.0f}")

    print("\n=== Mean innovation by CDR bin (all evaluations) ===")
    for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2),
                   (0.2, 0.35), (0.35, 0.5)]:
        sub = [i for c, i, d in allpts if lo <= c < hi]
        if sub:
            print(f"  CDR [{lo:.2f},{hi:.2f}): n={len(sub):>5}  "
                  f"mean innov {sum(sub) / len(sub):>10,.0f}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
