#!/usr/bin/env python3
"""
Merge every per-seed `episodes_seed{N}_unified.csv` found in the eval folder
into a single `episodes_all_seeds_unified.csv` with a leading `seed` column.

Discovery is automatic: any file matching the pattern is included. Rows are
emitted in seed order, then in the source file's order.

Schema of the master file (36 columns) — equivalent to the per-seed unified
header (35 cols) prefixed by `seed`:

  seed, <35 columns of episodes_seed{N}_unified.csv>
"""
import argparse
import csv
import re
from pathlib import Path


PER_SEED_PATTERN = re.compile(r"^episodes_seed(\d+)_unified\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        default=r"c:\ConflictualMAS\results\Evaluation Files",
        help="directory containing per-seed unified files",
    )
    ap.add_argument(
        "--out-name",
        default="episodes_all_seeds_unified.csv",
        help="output filename inside --in-dir",
    )
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    # Discover per-seed unified files; sort by numeric seed.
    discovered: list[tuple[int, Path]] = []
    for p in in_dir.iterdir():
        m = PER_SEED_PATTERN.match(p.name)
        if m:
            discovered.append((int(m.group(1)), p))
    if not discovered:
        print(f"[merge] no per-seed unified files found in {in_dir}")
        return 2
    discovered.sort(key=lambda t: t[0])

    # Read each file; all should share the same 35-col header.
    canonical_header: list[str] | None = None
    out_rows: list[dict] = []
    counts: dict[int, int] = {}
    for seed, path in discovered:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if canonical_header is None:
                canonical_header = reader.fieldnames or []
            elif reader.fieldnames != canonical_header:
                # Mismatch — abort rather than silently produce garbage.
                print(f"[merge] header mismatch in {path.name}:")
                print(f"  expected: {canonical_header}")
                print(f"  got     : {reader.fieldnames}")
                return 3
            n = 0
            for row in reader:
                out = {"seed": seed}
                out.update(row)
                out_rows.append(out)
                n += 1
            counts[seed] = n

    if canonical_header is None:
        return 2  # empty; defensive

    out_path = in_dir / args.out_name
    fieldnames = ["seed"] + canonical_header
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"[merge] wrote {len(out_rows)} rows -> {out_path}")
    for seed in sorted(counts):
        print(f"  seed {seed}: {counts[seed]} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())