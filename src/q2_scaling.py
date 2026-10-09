"""Compare eta_Q*(1-Q**kappa) with Q2 additive/interaction, +/- lambda*H_star.

Keep original Q2/Q3 outputs intact; write results/q2 by default.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import BASE, B, RES
from scaling_models import (FIELDS, csv, fit, load_data, metrics,
                                optimal_mixture, predict, read_json,
                                score_record, table_markdown)


FORMS = ("additive", "interaction", "quality_power")
MODELS = [(f"{form}_{suffix}", form, mode)
          for suffix, mode in [("fixed_E", "fixed"), ("mixture", "mixture")]
          for form in FORMS]
LABELS = {"additive": "Additive", "interaction": "Interaction D",
          "quality_power": "New quality power"}
EVALUATIONS = ["fit_unique", "group_cv_oof", "N_holdout", "D_holdout",
               "B6_to_B7_new_Q", "B8_raw_Q_transfer_diagnostic"]


def paired_comparisons(performance):
    indexed = performance.set_index(["model", "evaluation"])
    pairs = [(f"quality_power_{suffix}", f"{form}_{suffix}", "quality_formula")
             for suffix in ("fixed_E", "mixture") for form in FORMS[:2]]
    pairs += [(f"{f}_mixture", f"{f}_fixed_E", "H_star_offset") for f in FORMS]
    rows = []
    for candidate, reference, kind in pairs:
        for ev in EVALUATIONS:
            c, r = indexed.loc[(candidate, ev)], indexed.loc[(reference, ev)]
            rows.append({"comparison": kind, "candidate": candidate, "reference": reference,
                         "evaluation": ev, "delta_R2": c.R2 - r.R2,
                         "RMSE_candidate": c.RMSE, "RMSE_reference": r.RMSE,
                         "RMSE_reduction_pct": 100 * (1 - c.RMSE / r.RMSE),
                         "MAE_reduction_pct": 100 * (1 - c.MAE / r.MAE)})
    return pd.DataFrame(rows)


def figures(out, performance, predictions, parameters, E, H):
    colors = ["#4878A8", "#DC8B39", "#4D9475"]
    contexts = EVALUATIONS[1:5]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, suffix in zip(axes, ("fixed_E", "mixture")):
        for f, color in zip(FORMS, colors):
            subset = performance[performance.model == f"{f}_{suffix}"].set_index("evaluation")
            ax.plot(range(4), subset.loc[contexts, "RMSE"], "o-", label=LABELS[f], color=color)
        ax.set_xticks(range(4), ["Grouped CV", "Largest N", "Largest D", "New Q"])
        ax.set_ylabel("Held-out RMSE (lower is better)")
        ax.set_title("Without H*" if suffix == "fixed_E" else "With fitted lambda H*")
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle("Same fixed E and splits: semi-synthetic B6/B7")
    fig.savefig(out / "01_validation_comparison.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for ax, (model, form, mode) in zip(axes.flat, MODELS):
        sample = predictions[(predictions.model == model) & (predictions.evaluation == "group_cv_oof")]
        ax.scatter(sample.actual_loss, sample.predicted_loss, s=10, alpha=.5,
                   color=colors[FORMS.index(form)])
        lo, hi = sample[["actual_loss", "predicted_loss"]].to_numpy().min(), sample[["actual_loss", "predicted_loss"]].to_numpy().max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(LABELS[form] + (" + H*" if mode == "mixture" else ""))
        ax.set_xlabel("Observed supplement Loss")
        ax.set_ylabel("OOF predicted Loss")
        ax.grid(alpha=.2)
    fig.savefig(out / "02_oof_predictions.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    q = np.linspace(0, 1, 301)
    # Same illustrative N=1B,D=300B, never included in evaluation scores.
    for ax, suffix in zip(axes, ("fixed_E", "mixture")):
        for f, color in zip(FORMS, colors):
            p = parameters[(parameters.model == f"{f}_{suffix}") & (parameters.evaluation == "fit_unique")].iloc[0]
            term = p.eta_Q * (1 - q**p.kappa) if f == "quality_power" else p.C * (1-q)**p.g
            if f == "interaction":
                term *= 300**-p.d
            ax.plot(q, term, label=LABELS[f], color=color)
        ax.set_xlabel("Q")
        ax.set_ylabel("Quality penalty at N=1B, D=300B")
        ax.set_title("Without H*" if suffix == "fixed_E" else "With H* (offset excluded)")
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.savefig(out / "03_quality_response.png", dpi=190)
    plt.close(fig)


def write_report(out, E, mixture, audit, performance, pairs, parameters, folds, checks):
    summary = performance.pivot(index="model", columns="evaluation", values="R2").reindex([m[0] for m in MODELS]).reset_index()
    rmse = performance[performance.evaluation == "group_cv_oof"][["model", "R2", "RMSE", "MAE"]]
    comparison = pairs[pairs.evaluation == "group_cv_oof"]
    full_params = parameters[parameters.evaluation == "fit_unique"]
    fold_rows = []
    for suffix in ("fixed_E", "mixture"):
        for form in FORMS[:2]:
            new = folds[folds.model == f"quality_power_{suffix}"].set_index("fold")
            old = folds[folds.model == f"{form}_{suffix}"].set_index("fold")
            fold_rows.append({"candidate": f"quality_power_{suffix}", "reference": f"{form}_{suffix}",
                              "folds_lower_RMSE": int((new.RMSE < old.RMSE).sum()), "total_folds": 5})
    verdicts = []
    for _, row in pairs[(pairs.comparison == "quality_formula") & (pairs.evaluation != "fit_unique")].iterrows():
        verdicts.append(f"- {row.candidate} 对比 {row.reference}，{row.evaluation}："
                        f"RMSE{'下降' if row.RMSE_reduction_pct >= 0 else '上升'}"
                        f"{abs(row.RMSE_reduction_pct):.3f}%，R²变化{row.delta_R2:+.6f}。")
    new_parameters = full_params[full_params.model.str.startswith("quality_power")]
    cv_gains = pairs[(pairs.comparison == "quality_formula") & (pairs.evaluation == "group_cv_oof")]
    cv_by_pair = cv_gains.set_index(["candidate", "reference"]).RMSE_reduction_pct
    offset_gain = float(pairs[(pairs.candidate == "quality_power_mixture") &
                             (pairs.comparison == "H_star_offset") &
                             (pairs.evaluation == "group_cv_oof")].RMSE_reduction_pct.iloc[0])
    report = rf"""# 第二问：新质量幂函数的同数据比较

结果：新公式没有在本次验证中表现出相对additive的一致优势。
不加H*时，新公式五折验证RMSE比additive高{-cv_by_pair.loc[('quality_power_fixed_E', 'additive_fixed_E')]:.3f}%，
比interaction高{-cv_by_pair.loc[('quality_power_fixed_E', 'interaction_fixed_E')]:.3f}%。
加H*后，比interaction低{cv_by_pair.loc[('quality_power_mixture', 'interaction_mixture')]:.3f}%，
但比additive高{-cv_by_pair.loc[('quality_power_mixture', 'additive_mixture')]:.3f}%，且最大N/D留出均未优于对应旧公式。
新公式自身加入H*后五折验证RMSE下降{offset_gain:.3f}%。
因此目前可以将它保留为替代质量响应形式，尚无性能依据全面替换原模型。

截图公式按以下形式实现，质量指数作用在Q上。addictive按additive解释。

$$L_{{new}}=E+AN^{{-a}}+BD^{{-b}}+\eta_Q(1-Q^\kappa).$$

加配比项的版本为 $L_{{new,H}}=L_{{new}}+\lambda H^*$。

对照additive的质量项为 $C(1-Q)^g$，interaction的质量项为 $C(1-Q)^gD^{{-d}}$。
每种公式都计算不加H*、加H*两个版本，共六个模型。原主流程的interaction_N是另一种质量与N交互形式，本实验的interaction沿用第二问配比对照中的质量与D交互定义。

## 数据和公平对照

固定E={E:.12g}，沿用已有B1经典拟合的估计值。N和D单位均为十亿。
H*={mixture['H_star']:.12g}，仅读取第一问导出的M2模型求17域p_star，以17域均匀配比为参考，取13个验证损失域等权。
H*=F(p_star)-F(p_ref)，不是Shannon熵。没有再次读取原始A。

B6的{audit['B6_rows']}行与B7的{audit['B7_rows']}行合并后移除{audit['duplicates_removed']}条完全重复记录，共{audit['unique_rows']}行。
六个模型在同样的450行上拟合，使用同样的五折(N,D)分组验证，整组质量条件留在同一折。
OOF为每条样本在未参与该折训练时的预测。另留出最大N（50条）、最大D（90条），以及B6训练、B7新增Q条件测试（90条）。
所有验证只在相应训练部分拟合参数，各模型使用同等多初值最小二乘协议，未用全量拟合参数初始化验证折。
默认5个初值，固定随机种子20261008，解析Jacobian，迭代上限12000、收敛容差1e-11。
η_Q和C同范围[1e-6,1000]，κ和g同范围[1e-4,10]。λ≥0且E+λH*≥0。
B8的{audit['B8_rows']}行用全量模型直接预测，保留原Q含义与数值，仅列为迁移诊断。
B6–B8均为基于真实数据校准的半合成补充集，不能称为直接实验观测。

## 主要成绩

R²越高越好。最大N/D留出属于网格边界外推，不能代替真实大规模独立实验证据。

{table_markdown(summary, ['model', 'fit_unique', 'group_cv_oof', 'N_holdout', 'D_holdout', 'B6_to_B7_new_Q', 'B8_raw_Q_transfer_diagnostic'])}

五折验证汇总误差，RMSE与MAE越低越好：

{table_markdown(rmse, ['model', 'R2', 'RMSE', 'MAE'])}

成对比较：RMSE下降百分比=100×(1-RMSE_candidate/RMSE_reference)，负数为退步。
quality_formula只比较同H设置下的公式，H_star_offset只比较同公式有无常数修正。

{table_markdown(comparison, ['comparison', 'candidate', 'reference', 'delta_R2', 'RMSE_reduction_pct'])}

新公式在多少折获得更低的RMSE：

{table_markdown(pd.DataFrame(fold_rows), ['candidate', 'reference', 'folds_lower_RMSE', 'total_folds'])}

各验证场景的方向与幅度：

{chr(10).join(verdicts)}

## 新公式参数和解释

{table_markdown(new_parameters, ['model', 'A', 'a', 'B', 'b', 'eta_Q', 'kappa', 'lambda', 'effective_E', 'active_bounds'])}

η_Q(1-Q^κ)在Q=0时为η_Q，在Q=1时为0，对κ>0单调下降。
κ=1时与g=1的additive相同。两个非线性公式不互相包含，不能保证替换后一定提升。
本次新公式的κ分别为{new_parameters.iloc[0].kappa:.6f}、{new_parameters.iloc[1].kappa:.6f}，均接近1。
对应additive的g也接近1，因此两者都接近线性的质量惩罚，成绩差异很小。
这些小幅数值差异仅代表本次划分中的实际误差变化，未进行显著性检验。
接近Q=1时，新质量项约为η_Qκ(1-Q)，斜率有限；原additive取g<1时在Q=1附近斜率幅值趋于无穷。
图03给出同N=1B、D=300B下的质量惩罚曲线，解释拟合响应差异，不将示例点计入验证成绩。

λ的值见参数表。由于H*对所有B记录相同，E_eff=E+λH*就是可调的有效截距。
加入H*的收益只能表明固定E下常数偏移的校正有帮助，不能证明学到了配比效应、Q与p耦合或最优配比可迁移。
本实验也没有把B的Q替换为七域质量的加权平均。Q仍是原始Q_score，p不作为有观测支持的逐行特征。
对加H*模型，Q=1只消去质量惩罚，仍有λH*；因此只有不加H*或λ=0时才完整退化为原固定E经典公式。

## 外推和数值检查

B8原Q与B6/B7质量方向、Loss尺度存在差异，因此其R²不能用于单独证明统一标度律外推成功或失败。
本次没有虚构B4/B5/B10缺失的Q与p。要确认新公式在真实观测上的优势，需要独立且质量定义一致的完整N,D,Q实验。
固定E来自B1估计，并非已知的理论不可约损失。

独立复算预测、指标、Jacobian、边界Q=0/1、分组无泄漏与常数截距等价，结果见numerical_validation.json。
共{checks['checks_passed']}项检查通过。全量拟合额外使用12初值复算，SSE改善在容差内，见multistart_stability.csv。
参数条件数、有效秩与边界命中保存在参数表。参数不可辨识程度和验证成绩应分开判断。

运行：`python src/q2_scaling.py`。结果写入`results/q2`。
按分组OOF RMSE最小选择主模型；其全量拟合参数导出到selected_model.json，供第三问读取。
"""
    (out / "analysis.md").write_text(report, encoding="utf-8")


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    q1_path = Path(BASE) / "results/q1/mixture_models.json"
    old_path = Path(BASE) / "results/q2/classical_baseline.json"
    E = float(read_json(old_path)["classical"]["params"]["E"]) if args.fixed_E is None else args.fixed_E
    if not np.isfinite(E) or E <= 0 or args.nstarts < 3:
        raise ValueError("Positive finite E and at least 3 starts required")
    mixture = optimal_mixture(q1_path)
    H = mixture["H_star"]
    unique, _, b6, new7, b8, audit = load_data()
    _, groups = np.unique(unique[FIELDS[:2]].to_numpy(), axis=0, return_inverse=True)
    splits = list(GroupKFold(5).split(unique, groups=groups))
    assignments = np.zeros(len(unique), int)
    for fold, (tr, va) in enumerate(splits, 1):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Group leakage")
        assignments[va] = fold
    csv(unique.assign(group_id=groups, fold=assignments), out / "fit_data_and_splits.csv")
    seeds = {"A": .5, "a": .3, "B": 1.3, "b": .3, "C": .4, "g": 1,
             "eta_Q": .4, "kappa": 1, "d": .05, "lambda": min(.1, E / 2) / -H}
    performance, folds, parameters, predictions, serialized, stability = [], [], [], [], {}, []
    contexts = [("N_holdout", unique[unique.N_params_B < unique.N_params_B.max()],
                 unique[unique.N_params_B == unique.N_params_B.max()]),
                ("D_holdout", unique[unique.D_tokens_B < unique.D_tokens_B.max()],
                 unique[unique.D_tokens_B == unique.D_tokens_B.max()]),
                ("B6_to_B7_new_Q", b6, new7)]

    def save(model, evaluation, frame, pred, fitted, fold=None):
        if evaluation == "group_cv_fold":
            folds.append(score_record(model, evaluation, frame, pred, fitted, fold))
        else:
            performance.append(score_record(model, evaluation, frame, pred, fitted))
        parameters.append({"model": model, "evaluation": evaluation, "fold": fold, **fitted["params"]})
        serialized[f"{model}/{evaluation}/{fold}"] = {k: fitted[k] for k in ("names", "form", "mode", "E", "H", "params")}
        serialized[f"{model}/{evaluation}/{fold}"]["x"] = fitted["x"].tolist()
        if evaluation != "group_cv_fold":
            record_predictions(model, evaluation, frame, pred)

    def record_predictions(model, evaluation, frame, pred):
        ids = frame.row_id.to_numpy() if "row_id" in frame else np.arange(len(frame))
        predictions.extend({"model": model, "evaluation": evaluation, "row_id": int(r),
                            **{k: float(v) for k, v in zip(FIELDS[:3], ndq)},
                            "actual_loss": float(y), "predicted_loss": float(p)}
                           for r, ndq, y, p in zip(ids, frame[FIELDS[:3]].to_numpy(), frame.val_loss, pred))

    print(f"E={E:.12g}, H*={H:.12g}, n={len(unique)}, starts={args.nstarts}", flush=True)
    for model, form, mode in MODELS:
        full = fit(unique, form, mode, E, H, seeds, args.nstarts)
        pred = predict(full, unique)
        # Predeclared extra full-data starts diagnose optimizer stability;
        # never import these full-data parameters into held-out fits.
        stable = fit(unique, form, mode, E, H, seeds, max(12, args.nstarts))
        sse = np.sum((pred - unique.val_loss.to_numpy())**2)
        stable_sse = np.sum((predict(stable, unique) - unique.val_loss.to_numpy())**2)
        stability.append({"model": model, "initial_starts": args.nstarts, "extra_starts": max(12, args.nstarts),
                          "SSE_initial": sse, "SSE_extra": stable_sse, "SSE_improvement": sse - stable_sse})
        if sse - stable_sse > 1e-7:
            raise RuntimeError("Additional starts improve materially; increase --nstarts for all models")
        save(model, "fit_unique", unique, pred, full)
        oof = np.full(len(unique), np.nan)
        for fold, (tr, va) in enumerate(splits, 1):
            fitted = fit(unique.iloc[tr], form, mode, E, H, seeds, args.nstarts)
            oof[va] = predict(fitted, unique.iloc[va])
            save(model, "group_cv_fold", unique.iloc[va], oof[va], fitted, fold)
        if not np.isfinite(oof).all():
            raise AssertionError("Incomplete OOF")
        performance.append({"model": model, "evaluation": "group_cv_oof", **metrics(unique.val_loss.to_numpy(), oof)})
        record_predictions(model, "group_cv_oof", unique, oof)
        for evaluation, train, test in contexts:
            fitted = fit(train, form, mode, E, H, seeds, args.nstarts)
            save(model, evaluation, test, predict(fitted, test), fitted)
        save(model, "B8_raw_Q_transfer_diagnostic", b8, predict(full, b8), full)
        print(f"{model}: fit R2={metrics(unique.val_loss.to_numpy(), pred)['R2']:.6f}; "
              f"OOF R2={metrics(unique.val_loss.to_numpy(), oof)['R2']:.6f}", flush=True)
    perf, par, preds, fold_frame = map(pd.DataFrame, (performance, parameters, predictions, folds))
    ranked = perf[perf.evaluation == "group_cv_oof"].sort_values(["RMSE", "model"])
    winner = ranked.iloc[0]
    selected = {"model": winner.model, "criterion": "minimum group_cv_oof RMSE; full-data refit for Q3",
                "OOF_R2": float(winner.R2), "OOF_RMSE": float(winner.RMSE),
                "fit": serialized[f"{winner.model}/fit_unique/None"]}
    (out / "selected_model.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs = paired_comparisons(perf)
    for name, frame in [("model_performance.csv", perf), ("model_parameters.csv", par),
                        ("predictions.csv", preds), ("cv_folds.csv", fold_frame),
                        ("paired_comparisons.csv", pairs), ("multistart_stability.csv", pd.DataFrame(stability))]:
        csv(frame, out / name)
    sources = [q1_path, old_path] + [Path(B)/f for f in (
        "supplementary_NQ_experiment.csv", "supplementary_NQ_experiment_expanded.csv", "supplementary_NQ_experiment_large.csv")]
    manifest = {"fixed_E": E, "E_source": "CLI override" if args.fixed_E is not None else str(old_path.relative_to(BASE)),
                "mixture": mixture, "data_audit": audit, "nstarts": args.nstarts,
                "optimizer": "bounded scipy least_squares, analytic Jacobian, RNG 20261008",
                "sources_sha256": {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "code_sha256": {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path(__file__).with_name("scaling_models.py"))},
                "models": serialized,
                "limitation": "Constant H_star indistinguishable from effective intercept; B6-B8 semi-synthetic"}
    (out / "model_results.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    from q2_validate import validate
    checks = validate(out)
    write_report(out, E, mixture, audit, perf, pairs, par, fold_frame, checks)
    figures(out, perf, preds, par, E, H)
    print(perf[perf.evaluation == "group_cv_oof"][["model", "R2", "RMSE"]].to_string(index=False), flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-E", type=float)
    parser.add_argument("--nstarts", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q2")
    run(parser.parse_args())
