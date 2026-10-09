"""Audit final structure, numerical exports, Q4 preservation and actual run coverage."""
import ast
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from common import BASE
from quality_features import ALL_IND
from run_all import STAGES


def main():
    root = Path(BASE)
    src, results = root/"src", root/"results"
    checks = {}
    def check(name, passed, **details):
        checks[name] = {"passed": bool(passed), **details}
        if not passed:
            raise AssertionError(f"{name}: {details}")

    check("flat_source_layout", not (src/"code").exists()
          and all(p.parent == src for p in src.rglob("*.py")))
    check("no_old_active_scripts", not any(p.name.startswith(("p1", "p2", "p3", "p4", "validate_p1", "validate_p2", "validate_p3", "validate_p4"))
                                           for p in src.glob("*.py")))
    for path in src.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    check("source_syntax", True, files=len(list(src.glob("*.py"))))
    q1 = json.loads((results/"q1/run_manifest.json").read_text(encoding="utf-8"))
    semantic = json.loads((results/"q1/semantic_validation.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(results/"q1/main_domain_quality.csv")
    check("Q1_original_22_features", q1["indicators"] == ALL_IND == semantic["indicators"] and len(ALL_IND) == 22)
    check("Q1_full_data_and_domain_main", q1["n_records"] == semantic["n_records"] == 272505
          and len(quality) == 7 and (quality.baseline == "domain").all()
          and quality.Q_new.between(0, 1).all())
    mapping = pd.read_csv(results/"q1/semantic_pair_mapping.csv")
    check("Q1_missing_pairs_skipped", int(mapping.included.sum()) == semantic["n_valid_pairs"]
          and not mapping.loc[~mapping.included, "reason"].isna().any())
    regressions = json.loads((results/"q1/mixture_models.json").read_text(encoding="utf-8"))
    check("Q1_primary_regressions_and_ridge", {"M1_shared", "M2", "Ridge_original", "Ridge_tuned"} <= set(regressions))
    scores = pd.read_csv(results/"q1/mixture_model_performance.csv")
    check("Q1_nested_CV_and_external_evaluations", {"1M_nested_cv", "1M", "60M", "1B", "10B_est", "70B_est"}
          <= set(scores.scale) and np.isfinite(scores.mean_rmse_raw).all())
    q2 = json.loads((results/"q2/numerical_validation.json").read_text(encoding="utf-8"))
    check("Q2_numerical_checks", all(r["passed"] for r in q2["checks"]), checks=q2["checks_passed"])
    fits = pd.read_csv(results/"q2/model_performance.csv")
    selected = json.loads((results/"q2/selected_model.json").read_text(encoding="utf-8"))
    rank = fits[fits.evaluation == "group_cv_oof"].sort_values(["RMSE", "model"])
    check("Q2_six_models_and_OOF_selection", len(rank) == 6 and rank.iloc[0].model == selected["model"])
    q3 = json.loads((results/"q3/numerical_validation.json").read_text(encoding="utf-8"))
    check("Q3_numerical_checks", q3["all_passed"] and all(r["passed"] for r in q3["checks"].values()), checks=len(q3["checks"]))
    # Compare original Q4 computational functions with the final source AST;
    # only output filename prefixes are normalized, not expressions/constants.
    with zipfile.ZipFile(results/"archives/source_before_reorganization.zip") as archive:
        member = next(n for n in archive.namelist() if n.endswith("/p4_evolution.py"))
        original = archive.read(member).decode("utf-8-sig").replace("p4_", "q4_")
    old_tree = ast.parse(original)
    new_tree = ast.parse((src/"q4_evolution.py").read_text(encoding="utf-8"))
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    check("Q4_computational_functions_unchanged", old_functions == new_functions)
    forecast = pd.read_csv(results/"q4/q4_prediction.csv")
    check("Q4_predictions_finite_ordered", len(forecast) == 4 and np.isfinite(forecast[["median", "p10", "p90"]]).all().all()
          and (forecast.p10 <= forecast["median"]).all() and (forecast["median"] <= forecast.p90).all())
    actual = {}
    for path in sorted((results/"project_checks").glob("run_*.json")):
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            actual[run["script"]] = run
    required = {script for scripts in STAGES.values() for script in scripts}
    check("all_pipeline_stages_actually_executed", required <= set(actual)
          and all(actual[name]["exit_code"] == 0 for name in required), scripts=sorted(required))
    check("README_covers_every_python_file", all(p.name in (root/"README.md").read_text(encoding="utf-8")
                                                for p in src.glob("*.py")))
    output = results/"project_checks/project_validation.json"
    output.write_text(json.dumps({"all_passed": True, "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Project validation passed: {len(checks)} checks. {output}")


if __name__ == "__main__":
    main()
