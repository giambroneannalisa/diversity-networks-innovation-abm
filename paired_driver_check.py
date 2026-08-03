#!/usr/bin/env python3
"""
paired_driver_check.py — Is a sign inversion driver-induced or seed variation?

Seeds 42, 123 and 456 were run under both drivers: v1 (thesis, generation 50)
and v2 (extension, generation 45). Six further seeds were run only under v2.
That gives three cells:

    A  v1, seeds {42,123,456}      the thesis estimates
    B  v2, seeds {42,123,456}      same seeds, new driver
    C  v2, seeds {789...6789}      new seeds, new driver

If B tracks A, the driver is exonerated and any A-vs-C sign inversion is
genuine across-seed variation. If B tracks C instead, the inversion is an
artefact of the driver change (or of the shorter run) rather than of the seed.

Usage:
    python3 paired_driver_check.py <thesis-dir> <extension-dir>
"""
import csv, glob, math, os, re, sys

BOOT_N = 10000          # bootstrap resamples, fixed RNG seed 12345

SHARED = (42, 123, 456)
PARAMS = ('cultural-diffusion-rate', 'bridging-capital-weight',
          'innovation-diffusion-rate', 'policy-effectiveness')
SHORT = {'cultural-diffusion-rate': 'CDR', 'bridging-capital-weight': 'BCW',
         'innovation-diffusion-rate': 'IDR', 'policy-effectiveness': 'PE'}


def load(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get('Obj_Gini')]


def col(rows, name):
    return [float(r[name]) for r in rows]


def objectives(rows):
    return {'Innov': [-v for v in col(rows, 'Obj_Innov_Neg')],
            'Div': [-v for v in col(rows, 'Obj_Div_Neg')],
            'Gini': col(rows, 'Obj_Gini')}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (math.sqrt(sum((x - mx) ** 2 for x in xs))
           * math.sqrt(sum((y - my) ** 2 for y in ys)))
    return num / den if den else float('nan')


def relations(rows):
    """All 12 decision-variable x objective correlations, plus objective pairs."""
    obj = objectives(rows)
    out = {}
    for p in PARAMS:
        xs = col(rows, p)
        for oname, ys in obj.items():
            out[f'{SHORT[p]}-{oname}'] = pearson(xs, ys)
    out['Innov-Div'] = pearson(obj['Innov'], obj['Div'])
    out['Innov-Gini'] = pearson(obj['Innov'], obj['Gini'])
    out['Div-Gini'] = pearson(obj['Div'], obj['Gini'])
    return out


def fronts(directory):
    out = {}
    for p in glob.glob(os.path.join(directory, 'pareto_seed*.csv')):
        s = int(re.search(r'seed(\d+)', os.path.basename(p)).group(1))
        out[s] = load(p)
    return out


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    return m, sd


def main(thesis_dir, ext_dir):
    v1, v2 = fronts(thesis_dir), fronts(ext_dir)
    missing = [s for s in SHARED if s not in v1 or s not in v2]
    if missing:
        sys.exit(f'Seeds run under only one driver, cannot pair: {missing}')

    A = {s: relations(v1[s]) for s in SHARED}
    B = {s: relations(v2[s]) for s in SHARED}
    C = {s: relations(v2[s]) for s in sorted(v2) if s not in SHARED}
    keys = list(A[SHARED[0]])

    print(f"A = v1, seeds {list(SHARED)} (generation 50)")
    print(f"B = v2, seeds {list(SHARED)} (generation 45)")
    print(f"C = v2, seeds {sorted(C)} (generation 50)")

    # ---- per-seed sign agreement, the actual confounding test -----------
    print("\n=== Same seed, both drivers: does the sign hold? ===")
    print(f"{'relation':>12} " + " ".join(f"{'s%d v1' % s:>8} {'s%d v2' % s:>8}"
                                          for s in SHARED) + "  signs")
    flips = {}
    for k in keys:
        cells, agree = [], 0
        for s in SHARED:
            a, b = A[s][k], B[s][k]
            cells.append(f"{a:>+8.3f} {b:>+8.3f}")
            if (a > 0) == (b > 0):
                agree += 1
        flips[k] = agree
        mark = {3: 'all 3 hold', 2: '1 flips', 1: '2 flip', 0: 'all 3 flip'}[agree]
        print(f"{k:>12} " + " ".join(cells) + f"  {agree}/3 {mark}")

    print(f"\n  sign preserved in {sum(flips.values())}/{3 * len(keys)} "
          f"seed x relation pairs; relations holding in all three seeds: "
          f"{sum(1 for v in flips.values() if v == 3)}/{len(keys)}")

    # ---- does B look like A (same seeds) or like C (same driver)? -------
    print("\n=== Group means: A (thesis) | B (same seeds, v2) | C (new seeds, v2) ===")
    print(f"{'relation':>12} {'A mean':>16} {'B mean':>16} {'C mean':>16}   closer to")
    closer = {'A': 0, 'C': 0}
    for k in keys:
        ma, sa = stats([A[s][k] for s in SHARED])
        mb, _ = stats([B[s][k] for s in SHARED])
        mc, sc = stats([C[s][k] for s in C])
        which = 'A (seed effect)' if abs(mb - ma) <= abs(mb - mc) else 'C (driver/gen)'
        closer['A' if which.startswith('A') else 'C'] += 1
        print(f"{k:>12} {ma:>+9.3f}±{sa:<6.3f} {mb:>+9.3f}{'':7} "
              f"{mc:>+9.3f}±{sc:<6.3f}   {which}")
    print(f"\n  B is closer to A in {closer['A']}/{len(keys)} relations, "
          f"to C in {closer['C']}/{len(keys)}")
    print("  A high count on the left favours across-seed variation as the\n"
          "  explanation; a high count on the right would implicate the driver.")

    # ---- is the relation estimable at all? ------------------------------
    # Two distinct levels. Per front: a Pearson r computed on a single front
    # of n solutions has sampling SE ~ (1-r^2)/sqrt(n-1); if |r| is not large
    # relative to that, the sign of any individual run is close to a coin
    # flip. Pooled: averaging over all 13 runs shrinks the noise, so a
    # relation whose per-run sign is unreliable can still have a resolved
    # average - the bootstrap CI over runs decides.
    import random as _random
    _rng = _random.Random(12345)
    print("\n=== Identification per front, and resolution of the pooled mean ===")
    print(f"{'relation':>12} {'mean |r|':>9} {'|r|/SE':>7} {'per-front':>14} | "
          f"{'pooled mean':>11} {'boot 95% CI':>19} {'pooled sign':>12}")
    sizes = ([len(v1[s]) for s in SHARED] + [len(v2[s]) for s in SHARED]
             + [len(v2[s]) for s in C])
    n_typ = sorted(sizes)[len(sizes) // 2]
    rows_out = []
    for k in keys:      # stable key order: one RNG stream, reproducible CIs
        allr = [A[s][k] for s in SHARED] + [B[s][k] for s in SHARED] + \
               [C[s][k] for s in C]
        m_abs = sum(abs(r) for r in allr) / len(allr)
        se = (1 - m_abs ** 2) / math.sqrt(n_typ - 1)
        ratio = m_abs / se
        verdict = ('identified' if ratio >= 3 else
                   'weak' if ratio >= 1.5 else 'NOT identified')
        m = sum(allr) / len(allr)
        boots = sorted(sum(_rng.choice(allr) for _ in range(len(allr)))
                       / len(allr) for _ in range(BOOT_N))
        lo, hi = boots[int(0.025 * BOOT_N)], boots[int(0.975 * BOOT_N) - 1]
        sign = 'resolved' if lo * hi > 0 else 'straddles 0'
        rows_out.append((ratio, k, m_abs, se, verdict, m, lo, hi, sign))
    for ratio, k, m_abs, se, verdict, m, lo, hi, sign in sorted(rows_out,
                                                               reverse=True):
        print(f"{k:>12} {m_abs:>9.3f} {ratio:>7.1f} {verdict:>14} | "
              f"{m:>+11.3f} [{lo:>+7.3f},{hi:>+7.3f}] {sign:>12}")
    print(f"\n  typical front size n={n_typ}; per-front SE evaluated at each "
          f"relation's own mean |r|.\n  Below ~1.5 the sign of a single run "
          f"is noise; the pooled column says whether the\n  13-run average "
          f"nevertheless has a definite sign.")

    # ---- sensitivity: unique master seeds only --------------------------
    # The v2 re-runs of the thesis seeds share pymoo's initial population
    # with their v1 counterparts, so the 13 runs are not fully independent.
    # Dropping the three re-runs leaves ten runs with distinct master seeds.
    _rng2 = _random.Random(12345)
    uniq = [A[s] for s in SHARED] + [C[s] for s in C]
    print("\n=== Sensitivity: ten runs with distinct master seeds "
          "(thesis v1 + seven new) ===")
    print(f"{'relation':>12} {'mean':>8} {'boot 95% CI':>19}   vs 13-run")
    for k in keys:
        v13 = [A[s][k] for s in SHARED] + [B[s][k] for s in SHARED] + \
              [C[s][k] for s in C]
        m13 = sum(v13) / len(v13)
        v = [u[k] for u in uniq]
        m = sum(v) / len(v)
        boots = sorted(sum(_rng2.choice(v) for _ in range(len(v)))
                       / len(v) for _ in range(BOOT_N))
        lo, hi = boots[int(0.025 * BOOT_N)], boots[int(0.975 * BOOT_N) - 1]
        sign = 'resolved' if lo * hi > 0 else 'straddles 0'
        print(f"{k:>12} {m:>+8.3f} [{lo:>+7.3f},{hi:>+7.3f}] {sign:>12}"
              f"   ({m13:+.3f})")

    # ---- was the thesis sign ever resolved? -----------------------------
    # Using the pooled across-run SD as the per-run uncertainty, is each
    # cell's mean distinguishable from zero? A "sign inversion" between two
    # estimates that both straddle zero is not an inversion of anything.
    print("\n=== Was each sign ever resolved? (pooled across-run SD / sqrt(n)) ===")
    print(f"{'relation':>12} {'A mean':>8} {'SE':>6} {'A vs 0':>9}   "
          f"{'C mean':>8} {'SE':>6} {'C vs 0':>9}")
    sd_pool = {k: stats([A[s][k] for s in SHARED] + [B[s][k] for s in SHARED]
                        + [C[s][k] for s in C])[1] for k in keys}
    for ratio, k, *_ in sorted(rows_out, reverse=True):
        ma, _ = stats([A[s][k] for s in SHARED])
        mc, _ = stats([C[s][k] for s in C])
        sea, sec = sd_pool[k] / math.sqrt(3), sd_pool[k] / math.sqrt(len(C))
        fa = 'resolved' if abs(ma) > 2 * sea else 'straddles 0'
        fc = 'resolved' if abs(mc) > 2 * sec else 'straddles 0'
        print(f"{k:>12} {ma:>+8.3f} {sea:>6.3f} {fa:>9}   "
              f"{mc:>+8.3f} {sec:>6.3f} {fc:>9}")

    # ---- the four reported inversions, resolved -------------------------
    print("\n=== The A-vs-C sign inversions, adjudicated ===")
    for k in keys:
        ma, _ = stats([A[s][k] for s in SHARED])
        mc, _ = stats([C[s][k] for s in C])
        if (ma > 0) == (mc > 0) or abs(ma) < 0.05 or abs(mc) < 0.05:
            continue
        mb, _ = stats([B[s][k] for s in SHARED])
        if (mb > 0) == (ma > 0):
            verdict = ('same seeds keep the v1 sign under v2 -> the inversion '
                       'is across-seed variation, not the driver')
        else:
            verdict = ('same seeds flip sign under v2 -> the driver (or the '
                       'shorter run) is implicated, NOT seed variation')
        print(f"  {k}: A {ma:+.3f} -> C {mc:+.3f}   (B {mb:+.3f})\n      {verdict}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
