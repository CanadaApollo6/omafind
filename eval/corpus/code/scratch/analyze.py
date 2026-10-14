#!/usr/bin/env python3
"""Summarize the bean growth measurements for the science fair."""
import csv
import statistics
import sys
from collections import defaultdict

# columns: day, cup, light, height_cm
DATA = "growth_data.csv"


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "day": int(r["day"]),
                    "cup": r["cup"],
                    "light": r["light"].strip().lower(),
                    "height": float(r["height_cm"]),
                })
            except (KeyError, ValueError) as exc:
                # one bad row shouldn't kill the whole run
                print(f"skipping row {r}: {exc}", file=sys.stderr)
    return rows


def final_heights(rows):
    # keep only the last measurement (day 21) for each cup
    last = {}
    for r in rows:
        if r["cup"] not in last or r["day"] > last[r["cup"]]["day"]:
            last[r["cup"]] = r
    return list(last.values())


def summarize(rows):
    by_light = defaultdict(list)
    for r in final_heights(rows):
        by_light[r["light"]].append(r["height"])
    print(f"{'light':<10}{'n':>3}{'mean':>8}{'sd':>8}")
    for light in sorted(by_light):
        vals = by_light[light]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{light:<10}{len(vals):>3}{statistics.mean(vals):>8.2f}{sd:>8.2f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATA
    summarize(load(path))
