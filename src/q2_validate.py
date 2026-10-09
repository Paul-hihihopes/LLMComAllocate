"""Independent numerical verification of the new Q2 comparison exports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from common import BASE
from scaling_models import evaluate, parameter_spec, read_json


def independent_prediction(model, frame):
    p = dict(zip(model["names"], model["x"]))
    n, d, q = (frame[k].to_numpy(float) for k in ("N_params_B", "D_tokens_B", "Q_score"))
    intercept = model["E"] + p.get("lambda", 0) * model["H"]
    pred = intercept + p["A"] * n**-p["a"] + p["B"] * d**-p["b"]
    if model["form"] == "quality_power":
        pred += p["eta_Q"] * (1 - q**p["kappa"])
    else:
        pred += p["C"] * (1-q)**p["g"] * d**-p.get("d", 0)
    return pred


def validate(out):
    manifest = read_json(out / "model_results.json")
    data = pd.read_csv(out / "fit_data_and_splits.csv")
    preds = pd.read_csv(out / "predictions.csv")
    perf = pd.read_csv(out / "model_performance.csv")
    folds = pd.read_csv(out / "cv_folds.csv")
    comparisons = pd.read_csv(out / "paired_comparisons.csv")
    stability = pd.read_csv(out / "multistart_stability.csv")
    checks = []

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})
        if not passed:
            raise AssertionError(f"{name}: {details}")

    check("unique_data_and_all_folds", not data.duplicated(["N_params_B", "D_tokens_B", "Q_score", "val_loss"]).any()
          and set(data.fold) == {1, 2, 3, 4, 5} and len(data) == manifest["data_audit"]["unique_rows"])
    check("ND_groups_no_leakage", data.groupby(["N_params_B", "D_tokens_B"]).fold.nunique().max() == 1)
    expected_ids = {
        "fit_unique": set(data.row_id), "group_cv_oof": set(data.row_id),
        "N_holdout": set(data.loc[data.N_params_B == data.N_params_B.max(), "row_id"]),
        "D_holdout": set(data.loc[data.D_tokens_B == data.D_tokens_B.max(), "row_id"])}
    maximum_prediction_error = 0.
    maximum_jac_error = 0.
    for name in perf.model.unique():
        for ev in perf.evaluation.unique():
            sample = preds[(preds.model == name) & (preds.evaluation == ev)]
            if ev in expected_ids:
                check(f"{name}/{ev}/coverage", set(sample.row_id) == expected_ids[ev] and not sample.row_id.duplicated().any())
            if ev == "group_cv_oof":
                for fold in range(1, 6):
                    sample_fold = sample[sample.row_id.isin(data.loc[data.fold == fold, "row_id"])]
                    model = manifest["models"][f"{name}/group_cv_fold/{fold}"]
                    error = np.max(np.abs(independent_prediction(model, sample_fold) - sample_fold.predicted_loss))
                    maximum_prediction_error = max(maximum_prediction_error, float(error))
                    actual_rmse = np.sqrt(np.mean((sample_fold.actual_loss - sample_fold.predicted_loss)**2))
                    saved = folds[(folds.model == name) & (folds.fold == fold)].iloc[0]
                    check(f"{name}/fold{fold}/RMSE", np.isclose(actual_rmse, saved.RMSE, atol=1e-10))
            else:
                model = manifest["models"][f"{name}/{ev}/None"]
                error = np.max(np.abs(independent_prediction(model, sample) - sample.predicted_loss))
                maximum_prediction_error = max(maximum_prediction_error, float(error))
            y, pred = sample.actual_loss.to_numpy(), sample.predicted_loss.to_numpy()
            residual = pred - y
            saved = perf[(perf.model == name) & (perf.evaluation == ev)].iloc[0]
            independent = [1 - np.dot(residual, residual) / np.sum((y-y.mean())**2),
                           np.sqrt(np.mean(residual**2)), np.mean(abs(residual)), np.mean(residual)]
            check(f"{name}/{ev}/metrics", len(y) == saved.n and
                  np.allclose(independent, saved[["R2", "RMSE", "MAE", "bias"]].to_numpy(float), atol=2e-9, rtol=1e-9))
        model = manifest["models"][f"{name}/fit_unique/None"]
        test_frame = pd.DataFrame({"N_params_B": [.1, 1, 5, 10], "D_tokens_B": [10, 50, 300, 600],
                                   "Q_score": [0, .3, .9, 1]})
        x = np.array(model["x"])
        analytical = evaluate(x, model["names"], model["form"], model["mode"], test_frame, model["E"], model["H"], jac=True)
        numerical = approx_derivative(lambda z: evaluate(z, model["names"], model["form"], model["mode"],
                                                        test_frame, model["E"], model["H"]), x, method="3-point")
        maximum_jac_error = max(maximum_jac_error, float(np.max(abs(analytical-numerical))))
        check(f"{name}/Jacobian", np.allclose(analytical, numerical, atol=3e-6, rtol=2e-5))
        p = dict(zip(model["names"], x))
        lo, hi = parameter_spec(model["form"], model["mode"], model["E"], model["H"])[1:]
        check(f"{name}/parameter_bounds", np.all(x >= lo-1e-10) and np.all(x <= hi+1e-10))
        if model["form"] == "quality_power":
            grid = test_frame.iloc[[0]*101].copy()
            grid["Q_score"] = np.linspace(0, 1, 101)
            pred = independent_prediction(model, grid)
            base = model["E"] + p.get("lambda", 0)*model["H"] + p["A"]*.1**-p["a"] + p["B"]*10**-p["b"]
            check(f"{name}/quality_endpoints_monotonic", np.all(np.diff(pred) <= 1e-12) and
                  np.isclose(pred[0]-base, p["eta_Q"], atol=1e-12) and np.isclose(pred[-1], base, atol=1e-12))
        if model["mode"] == "mixture":
            # Exact intercept reparameterization must produce identical predictions.
            ctrl_names = [n for n in model["names"] if n != "lambda"] + ["effective_E"]
            ctrl_x = [p[n] for n in ctrl_names[:-1]] + [model["E"] + p["lambda"]*model["H"]]
            ctrl = evaluate(ctrl_x, ctrl_names, model["form"], "intercept_control", test_frame, model["E"], model["H"])
            check(f"{name}/intercept_equivalence", np.allclose(ctrl, independent_prediction(model, test_frame), atol=1e-12))
    check("all_exported_predictions_reconstructed", maximum_prediction_error < 1e-9, max_absolute_error=maximum_prediction_error)
    for _, row in comparisons.iterrows():
        c = perf[(perf.model == row.candidate) & (perf.evaluation == row.evaluation)].iloc[0]
        r = perf[(perf.model == row.reference) & (perf.evaluation == row.evaluation)].iloc[0]
        check(f"comparison/{row.candidate}/{row.reference}/{row.evaluation}",
              np.isclose(100*(1-c.RMSE/r.RMSE), row.RMSE_reduction_pct, atol=1e-8))
    check("multistart_stability", (stability.SSE_improvement <= 1e-7).all(), max_SSE_improvement=float(stability.SSE_improvement.max()))
    for category in ("sources_sha256", "code_sha256"):
        check(category, all(hashlib.sha256((Path(BASE)/p).read_bytes()).hexdigest() == h
                            for p, h in manifest[category].items()))
    result = {"checks_passed": len(checks), "max_prediction_error": maximum_prediction_error,
              "max_analytic_jacobian_error": maximum_jac_error, "checks": checks}
    (out / "numerical_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Numerical checks passed: {len(checks)}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q2")
    validate(parser.parse_args().output_dir)
