"""Independent physical-unit, feasibility, reference and repeatability checks for Q3."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE, MIX_DOMAINS
from q3_optimization import Problem, reference_solution, solve_pso


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q3")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    config = json.loads((out / "model_parameters.json").read_text(encoding="utf-8"))
    mix = config["mixture"]
    runs = pd.read_csv(out / "optimization_runs.csv")
    refs = pd.read_csv(out / "reference_optima.csv")
    sensitivity = pd.read_csv(out / "context_sensitivity.csv")
    history = pd.read_csv(out / "pso_convergence.csv")
    constraint_runs = pd.read_csv(out / "constraint_scenario_runs.csv")
    checks = {}

    def check(name, passed, **evidence):
        checks[name] = {"passed": bool(passed), **evidence}
        if not passed:
            raise AssertionError(f"{name}: {evidence}")

    pc = [f"p_{d}" for d in MIX_DOMAINS]
    all_rows = pd.concat([runs, refs, sensitivity, constraint_runs], ignore_index=True)
    n, d, q = [all_rows[k].to_numpy(float) for k in ("N_B", "D_B", "Q")]
    p = all_rows[pc].to_numpy(float)
    check("finite_inputs", np.isfinite(np.c_[n, d, q, p]).all())
    check("physical_bounds", ((n >= .005 - 1e-12) & (n <= 1e5 + 1e-8)
          & (d >= .2 - 1e-10) & (d <= 1e5 + 1e-8)
          & (q >= .4 - 1e-12) & (q <= 1 + 1e-12)).all())
    simplex_error = float(np.abs(p.sum(axis=1) - 1).max())
    check("simplex", simplex_error < 1e-8 and p.min() >= -1e-10,
          max_sum_error=simplex_error, minimum_fraction=float(p.min()))
    b, tau = np.array(mix["mean_coefficients"]), mix["tau"]
    hp = np.sum(np.log1p(np.maximum(p, 0) / tau) * b, axis=1)
    hp -= np.log1p(np.array(mix["p_ref"]) / tau) @ b
    h_error = float(np.abs(hp - all_rows.H_p).max())
    check("independent_H", h_error < 1e-10, max_absolute_error=h_error)
    check("mixture_lower_bound", hp.min() >= mix["H_star"] - 1e-9,
          minimum_H_minus_Hstar=float(hp.min() - mix["H_star"]))

    # Recalculate costs in RAW parameters/tokens, not the implementation's B units.
    n_raw, tokens = n * 1e9, d * 1e9
    train = 6 * n_raw * tokens
    attn = 2e-4 * n_raw * tokens * all_rows.L_ctx.to_numpy()
    form = all_rows.cost_form.fillna("power") if "cost_form" in all_rows else "power"
    # Sensitivity is explicitly power cost; absent rows above are filled here.
    f_diff = np.select([form == "exp", form == "power", form == "log"],
                       [1e7 * (np.exp(6 * q) - np.exp(6 * .4)),
                        5e9 * (q**4 - .4**4),
                        2e9 * (np.log1p(10 * q) - np.log1p(10 * .4))], default=np.nan)
    quality = tokens * np.maximum(f_diff, 0)
    total = train + attn + quality
    budget = all_rows.budget_FLOPs.to_numpy()
    cost_error = max(float(np.abs(value / all_rows[column].to_numpy() - 1).max())
                     for value, column in [(train, "C_train_FLOPs"), (attn, "C_attn_FLOPs")])
    quality_error = float(np.max(np.abs(quality - all_rows.C_quality_FLOPs) / budget))
    check("raw_unit_cost_recalculation", cost_error < 1e-11 and quality_error < 1e-11,
          max_train_attention_relative_error=cost_error, max_quality_error_over_budget=quality_error)
    check("budget", (total / budget <= 1 + 1e-9).all(), maximum_utilization=float((total / budget).max()))
    check("cost_shares", np.abs((all_rows.s_train + all_rows.s_quality + all_rows.s_attn) - 1).max() < 1e-10)
    ratio_error = float(np.max(np.abs(attn / train - 2e-4 * all_rows.L_ctx / 6)))
    check("attention_ratio_and_critical_context", ratio_error < 1e-10 and 6 / 2e-4 == 30000,
          max_ratio_error=ratio_error, ratio_2048=2e-4 * 2048 / 6, ratio_4096=2e-4 * 4096 / 6)
    loss_errors = []
    for model, rows in all_rows.groupby("model"):
        t = config["parameters"][model]
        ix = rows.index.to_numpy()
        if model.startswith("quality_power"):
            quality_loss = t["eta_Q"] * (1 - q[ix]**t["kappa"])
        else:
            quality_loss = t["C"] * np.maximum(1 - q[ix], 0)**t["g"]
        if model.startswith("interaction"):
            quality_loss *= d[ix]**(-t["d"])
        pred = t["E"] + t["A"] / n[ix]**t["a"] + t["B"] / d[ix]**t["b"] + quality_loss + t["lambda"] * hp[ix]
        loss_errors.extend(np.abs(pred - rows.Loss.to_numpy()).tolist())
    check("independent_loss", max(loss_errors) < 1e-10, max_absolute_error=max(loss_errors))

    grouping = ["model", "cost_form", "budget_FLOPs", "L_ctx", "recipe_mode", "seed"]
    monotone = all(np.all(np.diff(frame.best_Loss) <= 1e-12) for _, frame in history.groupby(grouping))
    check("PSO_best_history_monotone", monotone)
    last = history.sort_values("iteration").groupby(grouping).tail(1)
    pair = runs[runs.algorithm == "PSO"].merge(last, on=grouping, suffixes=("_run", "_hist"))
    expected_evaluations = config["particles"] * (config["iterations"] + 1)
    expected_evaluations += max(2, config["particles"] // 10) * ((config["iterations"] - 1) // 100)
    check("PSO_evaluation_accounting", len(pair) == len(runs) // 2
          and (pair.evaluations_run == expected_evaluations).all()
          and (pair.evaluations_run == pair.evaluations_hist).all()
          and np.abs(pair.Loss - pair.best_Loss).max() < 1e-10,
          evaluations_per_run=expected_evaluations)
    count = len(config["parameters"]) * 3 * 3 * 2 * 2 * 2 * config["repetitions"]
    check("complete_factorial_runs", len(runs) == count and not runs.duplicated(grouping + ["algorithm"]).any(),
          expected=count, observed=len(runs))

    # Check all main scenarios with a twice-denser independent Q scan.
    reference_changes = []
    for row in refs.itertuples():
        problem = Problem(row.model, config["parameters"][row.model], mix, row.cost_form,
                          row.budget_FLOPs, row.L_ctx, "fixed")
        _, refined_loss = reference_solution(problem, grid_size=321)
        reference_changes.append(abs(refined_loss - row.Loss))
    check("reference_grid_refinement", max(reference_changes) < 1e-8,
          scenarios=len(reference_changes), max_loss_change=max(reference_changes))
    check("no_solution_below_reference", runs.gap_to_reference.min() >= -1e-8,
          minimum_gap=float(runs.gap_to_reference.min()))
    # Independent KKT certificate for the existing recipe optimum.
    ps = np.array(mix["p_star"])
    gradient = b / (tau + ps)
    active = ps > 1e-10
    mu = -gradient[active].mean()
    kkt = float(np.max(np.abs(gradient[active] + mu)))
    check("recipe_KKT", kkt < 1e-8 and np.all(gradient[~active] >= -mu - 1e-8),
          max_active_stationarity_error=kkt)
    fixed_rows = runs[runs.recipe_mode == "fixed"]
    check("fixed_recipe_unchanged", np.abs(fixed_rows[pc].to_numpy() - ps).max() < 1e-12)
    main_configs = pd.read_csv(out / "main_optimal_configurations.csv")
    check("main_formula_selected_by_Q2", (main_configs.model == config["main_model"]).all()
          and (main_configs.algorithm == "PSO").all()
          and config["Q2_selection"]["fit"]["form"] == config["main_model"].replace("_fixed_E", ""))
    ablation = pd.read_csv(out / "constraints_before_after.csv")
    check("context_condition_comparison_complete", len(ablation) == len(config["parameters"])*3*3*2*config["repetitions"]
          and set(ablation.after_L_ctx) == {2048, 4096} and (ablation.before_L_ctx == 30000).all()
          and np.allclose(ablation.after_Loss-ablation.before_Loss, ablation.delta_Loss_after_minus_before))
    # Same seed must reproduce a representative complete free-recipe PSO run.
    selected = runs[(runs.algorithm == "PSO") & (runs.recipe_mode == "free")
                    & (runs.model == "interaction") & (runs.cost_form == "power")
                    & (runs.L_ctx == 4096) & (runs.budget_FLOPs == 1e22)].iloc[0]
    problem = Problem("interaction", config["parameters"]["interaction"], mix, "power", 1e22, 4096, "free")
    z, _, _ = solve_pso(problem, int(selected.seed), config["particles"], config["iterations"])
    reproduced = problem.record(z)
    repeat_error = max(abs(reproduced[k] - selected[k]) for k in ["N_B", "D_B", "Q", "Loss"] + pc)
    check("PSO_seed_reproducibility", repeat_error < 1e-9, max_absolute_error=repeat_error)
    for source, expected in config["sources_sha256"].items():
        check(f"source_hash:{source}", hashlib.sha256((Path(BASE) / source).read_bytes()).hexdigest() == expected)
    check("implementation_hash", hashlib.sha256((Path(__file__).parent / "q3_optimization.py").read_bytes()).hexdigest()
          == config["source_code_sha256"])
    result = {"all_passed": True, "checks": checks, "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "numerical_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": True, "checks": len(checks), "runs": len(runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
