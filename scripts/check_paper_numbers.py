"""
Fail if paper/main.tex contains a decimal number that is not produced by the pipeline.

Results must enter the text through \\input{tables/macros.tex}. Any literal decimal number in
main.tex is flagged unless it is whitelisted below (e.g. physical constants).
Usage: python scripts/check_paper_numbers.py   (exit code 1 on failure)
"""

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WHITELIST = {
    "0.75", "3.5", "5.5", "13.5", "0.3",  # TROPOMI qa threshold, pixel size (km), overpass time, cloud fraction
    "0.05", "0.1", "0.25", "0.45359237", "0.5", "1.0", "0.4",  # learning rate, dropout, grid deg, lb-kg, lr factor, min load, distance
    "2.0", "10.0",  # wind speed threshold (m/s), anemometer height (m)
}


def main() -> int:
    tex = (BASE / "paper" / "main.tex").read_text(encoding="utf-8")
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)                       # drop comments, keep \%
    tex = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", "", tex)
    run_id = (BASE / "experiments" / "LATEST").read_text().strip()
    allowed = set(json.load(open(BASE / "experiments" / run_id / "results_summary.json")).values())
    claimed = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", tex))
    bad = sorted(claimed - allowed - WHITELIST)
    if bad:
        print("Literal numbers in main.tex not produced by the pipeline:", bad)
        return 1
    print(f"OK: no unsupported decimal numbers in main.tex (run {run_id}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
