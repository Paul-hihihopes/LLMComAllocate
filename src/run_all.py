"""Run experiments in dependency order and record actual exit codes and runtimes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from common import BASE

STAGES = {"q1": ["q1_quality.py", "q1_mixture.py"],
          "q2": ["q2_baseline.py", "q2_scaling.py", "q2_validate.py"],
          "q3": ["q3_optimization.py", "q3_comparison_findings.py", "q3_validate.py"],
          "q4": ["q4_evolution.py"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", nargs="+", choices=tuple(STAGES), default=list(STAGES),
                        help="Run selected questions in dependency order; prerequisites must already exist.")
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parent, Path(BASE)/"results/project_checks"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = out/f"run_{stamp}.json"
    for question in STAGES:
        if question not in args.questions:
            continue
        for script in STAGES[question]:
            print(f"Running {script}", flush=True)
            started = perf_counter()
            log = out/f"{stamp}_{Path(script).stem}.log"
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run([sys.executable, "-u", str(root/script)], cwd=BASE,
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            records.append({"script": script, "seconds": perf_counter()-started,
                            "exit_code": process.returncode, "log": str(log.relative_to(BASE))})
            report.write_text(json.dumps({"started": stamp, "python": sys.executable,
                                           "runs": records, "all_passed": all(r["exit_code"] == 0 for r in records)},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{script}: exit={process.returncode}, {records[-1]['seconds']:.1f}s", flush=True)
            if process.returncode:
                print(log.read_text(encoding="utf-8")[-6000:], flush=True)
                raise SystemExit(process.returncode)
    print(f"All requested stages passed. Record: {report}", flush=True)


if __name__ == "__main__":
    main()
