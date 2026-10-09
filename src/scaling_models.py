"""Q2 fixed-E scaling comparisons under an optimal-mixture scenario.

Read A information ONLY through Q1's exported regression, never raw A files.
The constant lambda*H_star is an effective intercept, not an observed p effect.
Shared numerical library; experiments run through q2_baseline/q2_scaling.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from common import B, MIX_DOMAINS, LOSS_DOMAINS


FIELDS = ["N_params_B", "D_tokens_B", "Q_score", "val_loss"]


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def optimal_mixture(path):
    """Global minimum of Q1 M2's equally weighted 13-target prediction.

    Positive mean coefficients get zero mass. On negative coefficients the
    objective is strictly convex; water-filling satisfies its simplex KKT.
    This optimum is conditional on the regression, NOT measured at p_star.
    """
    exported = read_json(path)
    schema = exported["_schema"]
    if schema["input_domains"] != MIX_DOMAINS or schema["target_domains"] != LOSS_DOMAINS:
        raise ValueError("Q1 domain order does not match the current project")
    model = exported["M2"]
    if model["family"] != "M2":
        raise ValueError("An exported M2 model is required")
    coef = np.asarray(model["coef"], float)
    intercept = np.asarray(model["intercept"], float)
    tau = float(model["tau"])
    if coef.shape != (13, 17) or intercept.shape != (13,) or tau <= 0:
        raise ValueError("Invalid exported M2 dimensions or tau")
    b = coef.mean(axis=0)
    if not np.all(np.isfinite(coef)) or not np.any(b < 0):
        raise ValueError("Expected finite coefficients with negative mean effects")
    weights = np.maximum(-b, 0)
    mu = brentq(lambda v: np.maximum(weights / v - tau, 0).sum() - 1,
                1e-12, weights.sum() + 1, xtol=1e-14)
    p_star = np.maximum(weights / mu - tau, 0)
    p_ref = np.full(17, 1 / 17)
    pred_star = intercept + coef @ np.log1p(p_star / tau)
    pred_ref = intercept + coef @ np.log1p(p_ref / tau)
    h_star = float(np.mean(pred_star - pred_ref))
    if not np.isclose(p_star.sum(), 1, atol=1e-10) or h_star >= -1e-10:
        raise ValueError("Nonfeasible optimum or zero H_star; lambda cannot be estimated")
    gradient = b / (tau + p_star)
    active = p_star > 1e-10
    if np.max(np.abs(gradient[active] + mu)) > 1e-8 or np.any(gradient[~active] < -mu - 1e-8):
        raise AssertionError("Optimal mixture does not satisfy KKT")
    return {
        "model": "Q1 M2 (1M)", "tau": tau,
        "reference": "uniform 17-domain mixture", "target_weights": [1 / 13] * 13,
        "p_star": p_star.tolist(), "p_ref": p_ref.tolist(), "H_star": h_star,
        "predicted_mean_loss_star": float(pred_star.mean()),
        "predicted_mean_loss_ref": float(pred_ref.mean()),
        "mean_coefficients": b.tolist(), "KKT_multiplier": float(mu),
        "domain_prediction_star": pred_star.tolist(),
        "domain_prediction_ref": pred_ref.tolist(),
        "interpretation": "Regression optimum at 1M, not a directly observed optimum or proven B recipe",
    }


def load_data():
    b6 = pd.read_csv(Path(B) / "supplementary_NQ_experiment.csv")
    b7 = pd.read_csv(Path(B) / "supplementary_NQ_experiment_expanded.csv")
    b8 = pd.read_csv(Path(B) / "supplementary_NQ_experiment_large.csv")
    combined = pd.concat([b6.assign(source="B6"), b7.assign(source="B7")], ignore_index=True)
    for frame in (combined, b8):
        if not np.isfinite(frame[FIELDS].to_numpy(float)).all():
            raise ValueError("Nonfinite scaling data")
        if (frame[["N_params_B", "D_tokens_B"]] <= 0).any().any():
            raise ValueError("N and D must be positive")
        if not frame["Q_score"].between(0, 1).all():
            raise ValueError("Q must be in [0,1]")
    if (combined.groupby(FIELDS[:3])["val_loss"].nunique() > 1).any():
        raise ValueError("Conflicting duplicate conditions require explicit aggregation")
    unique = combined.drop_duplicates(FIELDS, keep="first").reset_index(drop=True)
    unique["row_id"] = np.arange(len(unique))
    keys6 = set(map(tuple, b6[FIELDS[:3]].to_numpy()))
    new7 = unique[[tuple(x) not in keys6 for x in unique[FIELDS[:3]].to_numpy()]].copy()
    audit = {"B6_rows": len(b6), "B7_rows": len(b7), "original_pooled_rows": len(combined),
             "unique_rows": len(unique), "duplicates_removed": len(combined) - len(unique),
             "B7_new_conditions": len(new7), "B8_rows": len(b8),
             "N_levels": sorted(unique.N_params_B.unique().tolist()),
             "D_levels": sorted(unique.D_tokens_B.unique().tolist()),
             "Q_levels": sorted(unique.Q_score.unique().tolist()),
             "provenance": "B6-B8 are semi-synthetic supplements, not direct experimental observations"}
    return unique, combined, b6, new7, b8, audit


def parameter_spec(form, mode, E, H):
    names = ["A", "a", "B", "b"]
    lower, upper = [1e-6, 1e-4, 1e-6, 1e-4], [1e3, 5, 1e3, 5]
    if form == "quality_power":
        names += ["eta_Q", "kappa"]
        lower += [1e-6, 1e-4]
        upper += [1e3, 10]
    elif form != "classical":
        names += ["C", "g"]
        lower += [1e-6, 1e-4]
        upper += [1e3, 10]
    if form in ("interaction", "interaction_N"):
        names += ["d" if form == "interaction" else "h"]
        lower += [1e-4]
        upper += [5]
    if mode == "mixture":
        # H < 0 and lambda >= 0 represent a gain from the reference mixture.
        # E_eff >= 0 prevents an unphysical negative asymptotic intercept.
        names += ["lambda"]
        lower += [0]
        upper += [E / (-H)]
    elif mode == "intercept_control":
        # Exactly the same feasible predictions as fixed E + lambda*H.
        names += ["effective_E"]
        lower += [0]
        upper += [E]
    return names, np.array(lower, float), np.array(upper, float)


def evaluate(x, names, form, mode, frame, E, H, jac=False):
    p = dict(zip(names, x))
    N, D, Q = (frame[k].to_numpy(float) for k in FIELDS[:3])
    lnN, lnD = np.log(N), np.log(D)
    basisN, basisD = np.exp(-p["a"] * lnN), np.exp(-p["b"] * lnD)
    termN, termD = p["A"] * basisN, p["B"] * basisD
    effective_E = E + p.get("lambda", 0) * H if mode != "intercept_control" else p["effective_E"]
    pred = effective_E + termN + termD
    columns = [basisN, -termN * lnN, basisD, -termD * lnD]
    if form == "quality_power":
        # Screenshot formula: eta_Q * (1 - Q**kappa), NOT (1-Q)**kappa.
        logQ = np.zeros_like(Q)
        np.log(Q, out=logQ, where=Q > 0)
        powerQ = Q ** p["kappa"]
        basisQ = 1 - powerQ
        pred += p["eta_Q"] * basisQ
        # lim Q->0 Q**kappa * log(Q) = 0 for kappa > 0.
        columns += [basisQ, -p["eta_Q"] * powerQ * logQ]
    elif form != "classical":
        one_minus_Q = np.maximum(1 - Q, 0)
        logQ = np.zeros_like(Q)
        np.log(one_minus_Q, out=logQ, where=one_minus_Q > 0)
        scale = np.ones_like(Q)
        if form == "interaction":
            scale = np.exp(-p["d"] * lnD)
        elif form == "interaction_N":
            scale = np.exp(-p["h"] * lnN)
        basisQ = one_minus_Q ** p["g"] * scale
        termQ = p["C"] * basisQ
        pred += termQ
        columns += [basisQ, termQ * logQ]
        if form in ("interaction", "interaction_N"):
            columns += [-termQ * (lnD if form == "interaction" else lnN)]
    if mode == "mixture":
        columns += [np.full_like(N, H)]
    elif mode == "intercept_control":
        columns += [np.ones_like(N)]
    return np.column_stack(columns) if jac else pred


def fit(frame, form, mode, E, H, seeds, nstarts=3):
    names, lower, upper = parameter_spec(form, mode, E, H)
    initial = np.array([seeds.get(k, {"lambda": max(0, (E - 1.5) / -H),
                                     "effective_E": min(E, 1.5)}.get(k, .3)) for k in names])
    rng = np.random.default_rng(20261008)
    starts = [initial]
    for _ in range(nstarts - 1):
        trial = initial.copy()
        for j, name in enumerate(names):
            if name not in ("lambda", "effective_E"):
                trial[j] *= np.exp(rng.normal(0, .5))
        starts.append(trial)
    best = None
    y = frame.val_loss.to_numpy(float)
    for start in starts:
        start = np.clip(start, lower + 1e-10, upper - 1e-10)
        sol = least_squares(lambda x: evaluate(x, names, form, mode, frame, E, H) - y,
                            start, jac=lambda x: evaluate(x, names, form, mode, frame, E, H, jac=True),
                            bounds=(lower, upper), x_scale="jac", max_nfev=12000,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if sol.success and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        raise RuntimeError(f"No converged {form}/{mode} fit")
    params = dict(zip(names, map(float, best.x)))
    params["E_fixed"] = float(E)
    params["effective_E"] = float(params.get("effective_E", E + params.get("lambda", 0) * H))
    if mode == "mixture":
        params["H_star"] = H
        params["mixture_offset"] = params["lambda"] * H
    norm = np.linalg.norm(best.jac, axis=0)
    singular = np.linalg.svd(best.jac / np.maximum(norm, 1e-30), compute_uv=False)
    params["normalized_jacobian_condition"] = float(singular[0] / singular[-1])
    params["jacobian_rank"] = int(np.linalg.matrix_rank(best.jac))
    params["n_parameters"] = len(names)
    params["nfev"] = int(best.nfev)
    params["active_bounds"] = ",".join(k for k, v in zip(names, best.active_mask) if v)
    return {"names": names, "x": best.x, "params": params,
            "form": form, "mode": mode, "E": E, "H": H}


def predict(fitted, frame):
    return evaluate(fitted["x"], fitted["names"], fitted["form"], fitted["mode"],
                    frame, fitted["E"], fitted["H"])


def metrics(y, pred):
    residual = pred - y
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "R2": 1 - sse / sst, "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "MAE": float(np.mean(np.abs(residual))), "bias": float(residual.mean())}


def score_record(model, evaluation, frame, pred, fitted, fold=None):
    return {"model": model, "evaluation": evaluation, "fold": fold,
            **metrics(frame.val_loss.to_numpy(float), pred),
            "lambda": fitted["params"].get("lambda", np.nan),
            "effective_E": fitted["params"]["effective_E"]}


def table_markdown(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = [f"{row[k]:.6f}" if isinstance(row[k], (float, np.floating)) else str(row[k]) for k in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
