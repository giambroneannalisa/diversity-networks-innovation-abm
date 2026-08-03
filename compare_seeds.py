#!/usr/bin/env python3
"""
compare_seeds.py — Side-by-side robustness comparison across seed groups.

Compares the thesis seeds against post-defence extension seeds and reports
the pooled picture: per-seed knee points with CDR regime classification,
key correlations, group summaries (mean ± SD, range, CV), and bootstrap
confidence intervals on the pooled means.

Usage:
    python3 compare_seeds.py <thesis-dir> <extension-dir>

Both directories are scanned for pareto_seed*.csv. Stdlib only; the
bootstrap uses a fixed RNG seed so output is deterministic.
"""
import csv, glob, math, os, random, re, sys

LOW, MID = 0.01, 0.05          # thesis §IV.4 regime boundaries
BOOT_N, BOOT_SEED = 10000, 12345


def load(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get('Obj_Gini')]


def col(rows, name):
    return [float(r[name]) for r in rows]


def objectives(rows):
    return ([-v for v in col(rows, 'Obj_Innov_Neg')],
            [-v for v in col(rows, 'Obj_Div_Neg')],
            col(rows, 'Obj_Gini'))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (math.sqrt(sum((x - mx) ** 2 for x in xs))
           * math.sqrt(sum((y - my) ** 2 for y in ys)))
    return num / den if den else float('nan')


def knee(rows):
    """§IV.5: max equally-weighted sum of min-max normalised objectives."""
    innov, div, gini = objectives(rows)

    def mm(v, invert=False):
        lo, hi = min(v), max(v)
        if hi == lo:
            return [0.0] * len(v)
        return [(hi - x) / (hi - lo) if invert else (x - lo) / (hi - lo) for x in v]

    score = [a + b + c for a, b, c in zip(mm(innov), mm(div), mm(gini, invert=True))]
    return score.index(max(score))


def regime(cdr):
    return 'low' if cdr < LOW else 'mid' if cdr < MID else 'high'


def stats(vals):
    n = len(vals)
    if n == 0:
        nan = float('nan')
        return nan, nan, nan, nan, nan
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    cv = sd / m * 100 if m else float('nan')
    return m, sd, min(vals), max(vals), cv


def _mean(v):
    return sum(v) / len(v)


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def boot_ci(vals, level=0.95, stat=_mean):
    """Percentile bootstrap CI of `stat` (fixed RNG → deterministic)."""
    rng = random.Random(BOOT_SEED)
    n = len(vals)
    if n < 2:
        nan = float('nan')
        return nan, nan
    reps = sorted(stat([rng.choice(vals) for _ in range(n)]) for _ in range(BOOT_N))
    lo = reps[int((1 - level) / 2 * BOOT_N)]
    hi = reps[int((1 + level) / 2 * BOOT_N) - 1]
    return lo, hi


def collect(directory, label):
    """Key by (label, seed): the extension re-runs some thesis seeds under the
    v2 driver, so seed alone would let one group overwrite the other."""
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, 'pareto_seed*.csv')),
                    key=lambda q: int(re.search(r'seed(\d+)', q).group(1))):
        s = int(re.search(r'seed(\d+)', p).group(1))
        gen = re.search(r'_gen(\d+)', os.path.basename(p))
        out[(label, s)] = {'rows': load(p), 'group': label,
                           'seed': s, 'gen': int(gen.group(1)) if gen else 50}
    return out


def main(thesis_dir, ext_dir):
    fronts = {}
    fronts.update(collect(thesis_dir, 'thesis'))
    fronts.update(collect(ext_dir, 'extension'))
    if not fronts:
        sys.exit('No pareto_seed*.csv found in either directory')

    groups = {'thesis': [], 'extension': []}
    for k, d in fronts.items():
        groups[d['group']].append(d['seed'])

    print(f"Loaded {len(fronts)} fronts")
    print(f"  thesis    (v1 driver): {sorted(groups['thesis'])}")
    print(f"  extension (v2 driver): {sorted(groups['extension'])}")

    # ---- per-seed table -------------------------------------------------
    print("\n=== Per-seed knee points and key correlations ===")
    print(f"{'seed':>6} {'group':>10} {'gen':>4} {'n':>4} {'kneeCDR':>9} {'regime':>7} "
          f"{'div%':>6} {'innov%':>7} {'CDR-Div':>8} {'CDR-Inn':>8} {'Inn-Gini':>9} {'PE-Gini':>8}")
    per_seed = {}
    for s in sorted(fronts, key=lambda x: (fronts[x]['group'] != 'thesis', fronts[x]['seed'])):
        rows = fronts[s]['rows']
        innov, div, gini = objectives(rows)
        cdr, pe = col(rows, 'cultural-diffusion-rate'), col(rows, 'policy-effectiveness')
        k = knee(rows)
        rec = {
            'group': fronts[s]['group'], 'n': len(rows),
            'seed': fronts[s]['seed'], 'gen': fronts[s]['gen'],
            'knee_cdr': cdr[k],
            'knee_bcw': col(rows, 'bridging-capital-weight')[k],
            'knee_idr': col(rows, 'innovation-diffusion-rate')[k],
            'knee_pe': pe[k],
            'div_share': div[k] / max(div) * 100,
            'innov_share': innov[k] / max(innov) * 100,
            'cdr_div': pearson(cdr, div), 'cdr_innov': pearson(cdr, innov),
            'innov_gini': pearson(innov, gini), 'pe_gini': pearson(pe, gini),
        }
        per_seed[s] = rec
        print(f"{rec['seed']:>6} {rec['group']:>10} {rec['gen']:>4} {rec['n']:>4} {rec['knee_cdr']:>9.4f} "
              f"{regime(rec['knee_cdr']):>7} {rec['div_share']:>5.1f}% {rec['innov_share']:>6.1f}% "
              f"{rec['cdr_div']:>+8.3f} {rec['cdr_innov']:>+8.3f} "
              f"{rec['innov_gini']:>+9.3f} {rec['pe_gini']:>+8.3f}")

    # ---- group summaries ------------------------------------------------
    metrics = [('knee CDR', 'knee_cdr'), ('knee div share %', 'div_share'),
               ('knee innov share %', 'innov_share'),
               ('CDR-Diversity r', 'cdr_div'), ('CDR-Innovation r', 'cdr_innov'),
               ('Innovation-Gini r', 'innov_gini'), ('PE-Gini r', 'pe_gini'),
               ('knee BCW', 'knee_bcw'), ('knee IDR', 'knee_idr'), ('knee PE', 'knee_pe')]

    def subset(name):
        if name == 'pooled':
            return list(per_seed)
        return [s for s in per_seed if per_seed[s]['group'] == name]

    shown = [g for g in ('thesis', 'extension') if subset(g)] + ['pooled']

    print("\n=== Group summaries (mean ± SD [min, max], CV) ===")
    print(f"{'metric':>20} | " + " | ".join(f"{g + ' (n=%d)' % len(subset(g)):^34}" for g in shown))
    for label, key in metrics:
        cells = []
        for g in shown:
            vals = [per_seed[s][key] for s in subset(g)]
            m, sd, lo, hi, cv = stats(vals)
            fmt = '.4f' if 'CDR' in label and 'r' not in label else '.3f'
            cells.append(f"{m:{fmt}} ±{sd:{fmt}} [{lo:{fmt}},{hi:{fmt}}] {cv:>4.0f}%")
        print(f"{label:>20} | " + " | ".join(f"{c:^34}" for c in cells))

    # ---- the EUCC invariant --------------------------------------------
    print("\n=== EUCC invariant: how many knees fall in the low-CDR regime? ===")
    for g in shown:
        ss = subset(g)
        low = [s for s in ss if regime(per_seed[s]['knee_cdr']) == 'low']
        out = [s for s in ss if regime(per_seed[s]['knee_cdr']) != 'low']
        print(f"  {g:>10}: {len(low)}/{len(ss)} in low-CDR"
              + (f"  — outside: " + ", ".join(f"seed {per_seed[s]['seed']} "
                                              f"(CDR={per_seed[s]['knee_cdr']:.4f}, "
                                              f"{regime(per_seed[s]['knee_cdr'])})" for s in out) if out else ""))

    print("\n=== Bootstrap 95% CI of pooled means (10,000 resamples, fixed RNG) ===")
    for label, key in [('knee CDR', 'knee_cdr'), ('CDR-Diversity r', 'cdr_div'),
                       ('CDR-Innovation r', 'cdr_innov'), ('Innovation-Gini r', 'innov_gini'),
                       ('PE-Gini r', 'pe_gini'), ('knee div share %', 'div_share')]:
        vals = [per_seed[s][key] for s in subset('pooled')]
        lo, hi = boot_ci(vals)
        m = sum(vals) / len(vals)
        print(f"  {label:>20}: mean {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    # ---- statistics the robustness claim rests on -----------------------
    pooled = subset('pooled')
    knees = [per_seed[s]['knee_cdr'] for s in pooled]
    lo, hi = boot_ci(knees, stat=_median)
    print(f"\n  {'knee CDR (median)':>20}: {_median(knees):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          "   ← median resists the single mid-regime outlier")
    ind = [1.0 if regime(per_seed[s]['knee_cdr']) == 'low' else 0.0 for s in pooled]
    lo, hi = boot_ci(ind)
    print(f"  {'P(knee in low-CDR)':>20}: {_mean(ind):.3f}    95% CI [{lo:.3f}, {hi:.3f}]")

    # ---- same master seed under both drivers ----------------------------
    shared = sorted({k[1] for k in pooled
                     if k[0] == 'thesis' and ('extension', k[1]) in per_seed})
    if shared:
        print("\n=== Same master seed, v1 vs v2 driver "
              "(v2 runs of these seeds stopped at generation 45) ===")
        print(f"{'seed':>6} {'knee CDR v1':>12} {'knee CDR v2':>12} {'regime v1':>10} "
              f"{'regime v2':>10} {'CDR-Div v1':>11} {'CDR-Div v2':>11}")
        for s in shared:
            a, b = per_seed[('thesis', s)], per_seed[('extension', s)]
            print(f"{s:>6} {a['knee_cdr']:>12.4f} {b['knee_cdr']:>12.4f} "
                  f"{regime(a['knee_cdr']):>10} {regime(b['knee_cdr']):>10} "
                  f"{a['cdr_div']:>+11.3f} {b['cdr_div']:>+11.3f}")

    # ---- pooled-front regime table -------------------------------------
    print("\n=== CDR regimes on the pooled front (all seeds' solutions together) ===")
    allrows = [r for s in per_seed for r in fronts[s]['rows']]
    buckets = {'low': [], 'mid': [], 'high': []}
    for r in allrows:
        buckets[regime(float(r['cultural-diffusion-rate']))].append(r)
    for name in ('low', 'mid', 'high'):
        rows = buckets[name]
        if not rows:
            print(f"  {name:>5} n=0")
            continue
        i, d, g = objectives(rows)
        print(f"  {name:>5} n={len(rows):<4d} innov {sum(i)/len(i):>12,.0f}   "
              f"div {sum(d)/len(d):.3f}   gini {sum(g)/len(g):.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
