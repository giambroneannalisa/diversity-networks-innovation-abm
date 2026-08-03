#!/usr/bin/env python3
"""Byte-level digests of a run's output files, for reproducibility checking.

A re-run of `run_trajectory_v2.py` with the same master seed should produce
byte-identical output files. Comparing whole-file checksums answers only
yes/no; this script emits a *sequence* of digests so a mismatch also says
*where* the two runs diverged.

Usage:

    python3 repro_digest.py trajectory_seed789.csv
    python3 repro_digest.py replicates_seed789.csv
    python3 repro_digest.py pareto_seed789.csv

Output is CSV on stdout: `file,key,n_lines,sha256`.

The blocking is chosen per file kind:

* trajectory (`Generation` column) — one digest per generation, over that
  generation's rows in file order. A divergence localises to a generation.
* replicate log (`eval_index` column) — cumulative digests over every
  evaluation index below a multiple of 100, then `ALL`. Cumulative rather
  than per-block so the first mismatching key is also the first divergence.
* anything else (a Pareto front) — a single `ALL` digest.

Every file also gets a `HEADER` digest, so a column-name or column-order
change is reported separately from a value change.

Lines are hashed as raw text with `\\n` separators and no trailing newline:
the digest is of the bytes the driver wrote, not of a re-parsed and
re-formatted view of them. Both float formatting and row order are therefore
part of what is checked.
"""

import hashlib
import os
import sys


def sha256_lines(lines):
    h = hashlib.sha256()
    h.update("\n".join(lines).encode("utf-8"))
    return h.hexdigest()


def read_csv_lines(path):
    with open(path, "r", newline="") as f:
        text = f.read()
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise SystemExit(f"{path}: empty file")
    return lines[0], lines[1:]


def column_index(header, name):
    cols = header.split(",")
    return cols.index(name) if name in cols else None


def digests(path):
    header, rows = read_csv_lines(path)
    name = os.path.basename(path)
    out = [(name, "HEADER", 1, sha256_lines([header]))]

    gen_col = column_index(header, "Generation")
    eval_col = column_index(header, "eval_index")

    if gen_col is not None:
        blocks = {}
        order = []
        for line in rows:
            key = line.split(",")[gen_col]
            if key not in blocks:
                blocks[key] = []
                order.append(key)
            blocks[key].append(line)
        # numeric order where possible, so gen 10 does not sort before gen 2
        try:
            order = sorted(order, key=lambda k: float(k))
        except ValueError:
            pass
        for key in order:
            out.append((name, f"gen{key}", len(blocks[key]),
                        sha256_lines(blocks[key])))

    elif eval_col is not None:
        # Fine boundaries early, coarse later: a three-generation pilot run
        # reaches only ~70 evaluations, and it must still produce keys the
        # full run's manifest also contains.
        bounds = iter([10 * i for i in range(1, 10)] +
                      [100 * i for i in range(1, 200)])
        limit = next(bounds)
        for i, line in enumerate(rows):
            try:
                ev = int(line.split(",")[eval_col])
            except ValueError:
                continue
            while ev >= limit:
                out.append((name, f"eval<{limit}", i, sha256_lines(rows[:i])))
                limit = next(bounds)

    out.append((name, "ALL", len(rows), sha256_lines(rows)))
    return out


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__.strip())
    print("file,key,n_lines,sha256")
    for path in argv[1:]:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)},MISSING,0,-")
            continue
        for row in digests(path):
            print(",".join(str(c) for c in row))


if __name__ == "__main__":
    main(sys.argv)
