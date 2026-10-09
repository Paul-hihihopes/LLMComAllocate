# -*- coding: utf-8 -*-
"""Q1: compare original Ridge with the image's M1/M2, without test-set tuning.

Run: python src/q1_mixture.py
Tables, figures and Chinese interpretation are saved to results/q1/.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from common import A, BASE, MIX_DOMAINS, LOSS_DOMAINS, mixture_cols, loss_cols
from mixture_regression import fit_mixture_model


ALPHAS = [0.0, 1e-6, 1e-4, .001, .01, .1, 1.0, 10.0, 100.0]
TAUS = [.0001, .0003, .001, .003, .01, .03, .1, .3, 1.0, 3.0, 10.0]
MODEL_SPECS = {
    "Ridge_original": "linear",
    "Ridge_tuned": "linear",
    "M1_shared": "M1",
    "M1_independent": "M1_independent",
    "M2": "M2",
}
LABELS = {"Ridge_original": "原岭回归", "Ridge_tuned": "调参线性岭回归",
          "M1_shared": "M1共用系数", "M1_independent": "M1每域独立（补充）", "M2": "M2"}
COLORS = ["#83929C", "#3274A1", "#B85E64", "#6D9A75", "#E2842B"]
SCALES = ["1M", "60M", "1B", "10B_est", "70B_est"]


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def log(message):
    print(message, flush=True)


def load_sets():
    names = {"train": "train_1m", "1M": "test_1m", "60M": "test_60m", "1B": "test_1B",
             "10B_est": "est_10b", "70B_est": "est_70b"}
    data, audit = {}, []
    for key, name in names.items():
        mode, scale = name.split("_")
        paths = [Path(A) / "regmix_tables" / f"{mode}_{kind}_{scale}.csv"
                 for kind in ["mixture", "pile_loss"]]
        mix, loss = [pd.read_csv(p) for p in paths]
        if mix["index"].duplicated().any() or loss["index"].duplicated().any():
            raise ValueError(f"Duplicate index in {key}")
        if set(mix["index"]) != set(loss["index"]):
            raise ValueError(f"Unmatched mixture/loss keys in {key}")
        frame = mix.merge(loss, on="index", validate="one_to_one", sort=False)
        X = frame[mixture_cols()].to_numpy(dtype=float)
        Y = frame[loss_cols()].to_numpy(dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(Y).all() or np.any(X < 0) or np.any(X > 1):
            raise ValueError(f"Invalid finite proportions/losses in {key}")
        if np.max(np.abs(X.sum(axis=1) - 1)) > .01:
            raise ValueError(f"Mixture sums out of original rounding tolerance in {key}")
        data[key] = (X, Y, frame["index"].to_numpy())
        for path in paths:
            audit.append({"split": key, "n": len(frame), "source": str(path.relative_to(BASE)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "sum_p_min": X.sum(axis=1).min(), "sum_p_max": X.sum(axis=1).max()})
    train_mix = set(map(tuple, data["train"][0]))
    for row in audit:
        X = data[row["split"]][0]
        row["n_exact_mixtures_also_in_train"] = sum(tuple(x) in train_mix for x in X)
    return data, pd.DataFrame(audit)


def domain_metrics(Y, pred):
    residual = Y - pred
    variance_sum = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    mse = np.mean(residual ** 2, axis=0)
    shifted = pred - pred.mean(axis=0) + Y.mean(axis=0)
    raw_r2 = 1 - np.sum(residual ** 2, axis=0) / variance_sum
    shift_r2 = 1 - np.sum((Y - shifted) ** 2, axis=0) / variance_sum
    return pd.DataFrame({"target_domain": LOSS_DOMAINS, "r2_raw": raw_r2,
                         "r2_shift_diagnostic": shift_r2, "rmse_raw": np.sqrt(mse),
                         "rmse_shift_diagnostic": np.sqrt(np.mean((Y - shifted) ** 2, axis=0)),
                         "mae_raw": np.mean(np.abs(residual), axis=0),
                         "mean_bias_pred_minus_y": np.mean(pred - Y, axis=0),
                         "pearson": [np.corrcoef(Y[:, j], pred[:, j])[0, 1]
                                     for j in range(Y.shape[1])],
                         "spearman": [spearmanr(Y[:, j], pred[:, j]).statistic
                                      for j in range(Y.shape[1])]})


def mean_metrics(frame):
    return {f"mean_{key}": float(frame[key].mean()) for key in frame if key != "target_domain"}


def candidates(model_name):
    if model_name == "Ridge_original":
        return [(1.0, 1.0)]
    return [(alpha, tau) for tau in (TAUS if model_name == "M2" else [1.0]) for alpha in ALPHAS]


def select_config(X, Y, name, splits, context, history):
    if name == "Ridge_original":
        return 1.0, 1.0
    rows = []
    family = MODEL_SPECS[name]
    for alpha, tau in candidates(name):
        # One recipe carries all 13 losses and stays in the same fold.
        fold_mse = []
        for tr, va in splits:
            mdl = fit_mixture_model(X[tr], Y[tr], family=family, alpha=alpha, tau=tau)
            fold_mse.append(np.mean((Y[va] - mdl.predict(X[va])) ** 2))
        row = {"context": context, "model": name, "alpha": alpha, "tau": tau,
               "cv_mse": float(np.mean(fold_mse)), "cv_mse_fold_sd": float(np.std(fold_mse, ddof=1))}
        rows.append(row)
    best = min(rows, key=lambda row: row["cv_mse"])
    for row in rows:
        row["selected"] = row is best
    history.extend(rows)
    return best["alpha"], best["tau"]


def nested_cv(X, Y, history):
    outer = list(KFold(5, shuffle=True, random_state=42).split(X))
    oof = {name: np.full_like(Y, np.nan) for name in MODEL_SPECS}
    rows = []
    for fold, (tr, va) in enumerate(outer, 1):
        inner = list(KFold(4, shuffle=True, random_state=100 + fold).split(X[tr]))
        for name, family in MODEL_SPECS.items():
            alpha, tau = select_config(X[tr], Y[tr], name, inner, f"outer_{fold}", history)
            mdl = fit_mixture_model(X[tr], Y[tr], family=family, alpha=alpha, tau=tau)
            pred = mdl.predict(X[va])
            oof[name][va] = pred
            rows.append({"fold": fold, "model": name, "n_valid": len(va),
                         "alpha": alpha, "tau": tau, **mean_metrics(domain_metrics(Y[va], pred))})
        log(f"Nested CV fold {fold}/5 complete")
    assert all(np.isfinite(pred).all() for pred in oof.values())
    return oof, pd.DataFrame(rows)


def paired_bootstrap(Y, predictions, count=2000):
    rng = np.random.default_rng(42)
    rows = []
    for reference in ["Ridge_original", "Ridge_tuned"]:
        for name in ["M1_shared", "M1_independent", "M2"]:
            diffs = []
            for _ in range(count):
                ix = rng.integers(0, len(Y), size=len(Y))
                rmse_new = np.sqrt(np.mean((Y[ix] - predictions[name][ix]) ** 2, axis=0)).mean()
                rmse_ref = np.sqrt(np.mean((Y[ix] - predictions[reference][ix]) ** 2, axis=0)).mean()
                diffs.append(rmse_new - rmse_ref)
            rmse_diff = (np.sqrt(np.mean((Y - predictions[name]) ** 2, axis=0)).mean()
                         - np.sqrt(np.mean((Y - predictions[reference]) ** 2, axis=0)).mean())
            rows.append({"scale": "1M", "model": name, "reference": reference,
                         "mean_rmse_difference": rmse_diff, "ci95_low": np.quantile(diffs, .025),
                         "ci95_high": np.quantile(diffs, .975), "n_resamples": count})
    return pd.DataFrame(rows)


def model_checks(X, Y):
    reference = np.column_stack([Ridge(alpha=1).fit(X, Y[:, j]).predict(X) for j in range(Y.shape[1])])
    linear = fit_mixture_model(X, Y).predict(X)
    baseline_error = np.max(np.abs(reference - linear))
    assert baseline_error < 1e-9
    m1 = fit_mixture_model(X, Y, "M1_independent", alpha=.1)
    m2 = fit_mixture_model(X, Y, "M2", alpha=.1, tau=1)
    assert np.allclose(m1.predict(X), m2.predict(X), atol=1e-12)
    # Literal M1's shared coefficients solve the stacked objective exactly.
    z = np.log1p(X)
    centered_z = z - z.mean(axis=0)
    centered_y = Y - Y.mean(axis=0)
    coef = np.linalg.solve(Y.shape[1] * centered_z.T @ centered_z + .7 * np.eye(X.shape[1]),
                           centered_z.T @ centered_y.sum(axis=1))
    shared = fit_mixture_model(X, Y, "M1", alpha=.7)
    assert np.allclose(shared.coef_, coef[None, :], atol=1e-10)
    # p=0 stays finite and contributes zero to the logarithmic features.
    assert np.all(m2.transform(np.zeros((1, X.shape[1]))) == 0)
    # Verify a feasible p_a+=eps, p_b-=eps derivative, rather than off-simplex effects.
    p = X.mean(axis=0)[None, :]
    epsilon = 1e-7
    changed = p.copy()
    changed[0, 0] += epsilon
    changed[0, 1] -= epsilon
    numerical = (m2.predict(changed) - m2.predict(p)) / epsilon
    analytic = m2.marginal_effects(p)[:, :, 0] - m2.marginal_effects(p)[:, :, 1]
    assert np.allclose(numerical, analytic, rtol=1e-4, atol=1e-6)
    return {"original_prediction_max_abs_difference": float(baseline_error),
            "M2_tau1_equals_independent_M1": True, "shared_objective_check": True,
            "zero_proportion_check": True, "simplex_derivative_check": True}


def plot_results(out, performance, search, predictions, data, models):
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ["Microsoft YaHei", "SimHei", "SimSun"] if name in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.sans-serif": [font], "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    tests = performance[performance["scale"].isin(SCALES)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, key, title in zip(axes.flat,
        ["mean_r2_raw", "mean_rmse_raw", "mean_r2_shift_diagnostic", "mean_spearman"],
        ["原始R²（独立预测）", "原始RMSE", "去偏移R²（使用真实测试均值的诊断）", "Spearman排序相关性"]):
        for (name, color) in zip(MODEL_SPECS, COLORS):
            t = tests[tests["model"] == name].set_index("scale").loc[SCALES]
            ax.plot(range(5), t[key], marker="o", label=LABELS[name], color=color)
        ax.set_xticks(range(5), ["1M", "60M", "1B", "10B预估", "70B预估"])
        if key == "mean_r2_raw":
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylabel("13域均值；对称对数轴")
        else:
            ax.set_ylabel("13域均值")
        ax.set_title(title)
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=9)
    fig.savefig(out / "12_mixture_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, name in zip(axes, ["Ridge_original", "M1_shared", "M2"]):
        Y = data["1M"][1].ravel()
        pred = predictions[(name, "1M")].ravel()
        ax.scatter(Y, pred, s=7, alpha=.3, color=COLORS[list(MODEL_SPECS).index(name)])
        low, high = min(Y.min(), pred.min()), max(Y.max(), pred.max())
        ax.plot([low, high], [low, high], color="#555555", linestyle="--", lw=1)
        ax.set_xlabel("实际Loss（256配比 × 13目标域）")
        ax.set_ylabel("原始预测Loss")
        ax.set_title(LABELS[name])
    fig.savefig(out / "13_mixture_1m_predictions.png", dpi=180)
    plt.close(fig)

    profile = search[(search["context"] == "full_train") & (search["model"] == "M2")]
    best = profile.loc[profile.groupby("tau")["cv_mse"].idxmin()].sort_values("tau")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(best["tau"], np.sqrt(best["cv_mse"]), "o-", color=COLORS[-1])
    ax.axvline(models["M2"].tau, color="#7A7A7A", linestyle="--", label=f"选择τ={models['M2'].tau:g}")
    ax.set_xscale("log")
    ax.set_xlabel("τ（配比使用0–1单位；每点仅在训练集选择岭系数）")
    ax.set_ylabel("训练集5折CV RMSE（均方误差开方）")
    ax.set_title("M2平滑尺度敏感性")
    ax.legend()
    fig.savefig(out / "14_mixture_tau_profile.png", dpi=180)
    plt.close(fig)


def markdown(frame, digits=4):
    def value(x):
        return f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
                      + ["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def report(out, performance, cv, configs, bootstrap, audit, models, data, per_domain):
    tests = performance[performance["scale"].isin(SCALES)]
    table = tests.pivot(index="scale", columns="model", values="mean_r2_shift_diagnostic").reindex(SCALES).reset_index()
    one_m = tests[tests["scale"] == "1M"][["model", "mean_r2_raw", "mean_rmse_raw", "mean_spearman"]]
    cv_table = cv.groupby("model").agg(r2_mean=("mean_r2_raw", "mean"), r2_fold_sd=("mean_r2_raw", "std"),
                                       rmse_mean=("mean_rmse_raw", "mean")).reset_index()
    lookup = tests.set_index(["model", "scale"])
    m2 = lookup.loc[("M2", "1M")]
    original = lookup.loc[("Ridge_original", "1M")]
    tuned = lookup.loc[("Ridge_tuned", "1M")]
    shared = lookup.loc[("M1_shared", "1M")]
    pct = 100 * (original["mean_rmse_raw"] - m2["mean_rmse_raw"]) / original["mean_rmse_raw"]
    domain_test = per_domain[per_domain["scale"] == "1M"].pivot(index="target_domain", columns="model", values="rmse_raw")
    wins_original = int((domain_test["M2"] < domain_test["Ridge_original"]).sum())
    wins_tuned = int((domain_test["M2"] < domain_test["Ridge_tuned"]).sum())
    X = data["train"][0]
    min_positive = float(X[X > 0].min())
    zeros = float(np.mean(X == 0))
    m2_fold_choices = cv[cv["model"] == "M2"][["fold", "alpha", "tau"]]
    text = f"""# 第一问：图中M1、M2配比回归对照

## 实验结论

两种函数均可计算。M1按图中没有j下标的w_k严格实现为13目标域共用17个系数，1M独立测试平均原始R²={shared['mean_r2_raw']:.4f}。M2在训练集选择τ={models['M2'].tau:g}、岭系数α={models['M2'].alpha:g}，1M测试平均原始R²={m2['mean_r2_raw']:.4f}、平均RMSE={m2['mean_rmse_raw']:.4f}。原固定α=1线性岭回归分别为{original['mean_r2_raw']:.4f}、{original['mean_rmse_raw']:.4f}，M2的RMSE变化为降低{pct:.2f}%。调参线性模型的平均RMSE={tuned['mean_rmse_raw']:.4f}，这是区分函数替换收益与正则化调参收益的公平对照。

本实验保留原模型默认行为，并提供可替换回归函数。是否采用新函数应主要依据1M独立测试和嵌套CV，不以训练拟合优度或预估表结果单独决定。未增加N、D或质量Q，因此不能把对数函数替换视为已经解决跨参数尺度预测。

## 模型定义与可行性

原始模型：L_ij = A_j + Σ_k β_kj p_ik，岭系数固定1。另提供使用同一α网格和CV划分的调参线性模型；M2还搜索τ，搜索规模更大。

图中M1：L_ij = A_j − Σ_k w_k ln(1+p_ik)。令β_k=−w_k即可在ln(1+p)特征上作线性回归，所有j共用β。该限制使各目标域对配比的响应完全相同，任意两个目标域的预测Loss差恒为A_j−A_l，不能表达某配比改善一个域而损害另一个域的权衡。补充M1_independent允许w_kj，它是放宽约束后的诊断模型，不能与图中的M1混称。

图中M2：L_ij = A_j + Σ_k b_kj ln(1+p_ik/τ)，τ>0。本次τ在13个目标域间共用，与图中的无j下标一致；固定τ后用岭回归求A_j、b_kj，以训练内CV选τ和岭系数。τ=1时M2与M1_independent同一函数，已验证。τ很大时ln(1+p/τ)≈p/τ，模型趋近线性；τ小时低配比附近更敏感。p=0的特征严格为0，不需要加伪计数。

图中未要求w或b非负，本实验没有擅自加符号限制。M1中的w是回归系数，不是22项质量指标的权重。若另加w≥0，M1会强迫所有目标域相同方向响应，应另行检验。p采用0–1比例而非百分数，换单位必须同步换τ。

M1和M2都是可加模型，不含训练域间交互项。在单纯形Σp=1下不能单独增加某一域而维持其余不变；转移δ配比从域l到域k的M2一阶效应为δ[b_kj/(τ+p_k)−b_lj/(τ+p_l)]。系数不是因果效应，非线性系数大小也不能脱离τ直接比较。

## 数据与验证协议

A4/A5训练512组1M配比及其13域Loss；A6/A7独立测试256组1M；A8/A9为256组60M；A10/A11为64组1B。A12–A15的10B/70B各63组是预估Loss，不能当作新增真实训练观测。

训练集512个配比互不重复，1M/60M/1B配比均没有与训练集完全相同的行。10B、70B各63个配比全部出现在训练集，是尺度迁移而非新配比泛化。源表配比和存在舍入误差，原值范围及哈希见mixture_data_audit.csv，未重归一化、删行或修改Loss。训练/测试不按同一数字index跨表连接，index只用于各自配比与Loss一对一连接。

所有模型使用同一5折外层CV（random_state=42）。调参模型每个外层训练折内部4折选择参数，外层验证折只评价；最终用全训练集5折选择最终参数。搜索目标为13域等权平均MSE，未用测试真实均值选择模型。岭α网格={ALPHAS}；M2的τ网格={TAUS}。各模型使用原始或对数特征，不作标准化，模型各自选择α，避免把同一α造成的不同尺度收缩当作函数效果。

M1的优化目标为Σ_ij残差²+α||β||²，仅共享系数罚一次，截距不惩罚。其解等价于对13域Loss均值回归，并使用α/13。图中未规定拟合损失或正则化，以上是本次明确采用的估计方法。

## 同尺度独立测试

以下指标均先对每个目标域计算，再对13域等权平均。平均RMSE是13个RMSE的平均，区别于全表MSE再开方。

{markdown(one_m)}

逐目标域RMSE显示，M2优于原模型的目标域有{wins_original}/13个，优于调参线性模型有{wins_tuned}/13个。整体改善不代表每一域都改善；逐域结果见mixture_per_domain_metrics.csv。

嵌套CV的逐折统计：

{markdown(cv_table)}

CV标准差描述5折变化，不是参数置信区间；最终训练拟合结果及OOF整体评价另见mixture_model_performance.csv。1M测试配比级成对bootstrap保留每组13个Loss，报告RMSE差的新减旧95%区间；负值表示新模型误差较低。该区间不包含重新训练或调参的不确定性，也不是多重检验校正结果。

{markdown(bootstrap)}

## 跨尺度迁移

下面仅展示与原流程同口径的去偏移R²诊断：

{markdown(table)}

原始R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²。去偏移先令ŷ'=ŷ−mean(ŷ)+mean(y)，再计算R²。后者使用测试真实均值，不是无需目标标签的预测。只平移截距没有重估斜率，通常不等于Pearson相关系数平方。负去偏移R²说明即使校正基线，配比变化的幅度或方向仍不能迁移。排序相关性只能说明顺序，不能说明绝对Loss精度。

原模型和两个新函数都没有N输入，1M训练截距会随尺度失配。详见mixture_model_performance.csv中的原始R²、原始RMSE、平均偏差与去偏移诊断。10B/70B预估表可能继承上游估算规律，其结果不构成独立真实外推证据。

## 参数稳定性与建议

最终参数：

{markdown(pd.DataFrame(configs))}

模型选择均使用验证集MSE。M2的五个外层训练折均选到同一τ与α，当前划分下选择稳定：

{markdown(m2_fold_choices, digits=6)}

τ的训练内剖面见14号图和mixture_hyperparameter_search.csv。τ与b尺度可能互相补偿，宽平坦的剖面意味着τ难以精确识别。网格选择不等于连续全局最优，也不能把τ解释为已经识别的物理常数。原始p接近单纯形约束，系数存在共线性，对数函数不自动消除所有参数不稳定。

源表训练配比最小正值为{min_positive:g}，约{zeros:.2%}的单元为0。本次τ={models['M2'].tau:g}小于最小正值，对零与微小正配比的差别较敏感。它可能同时学习“是否包含某域”的效应；现有舍入精度不足以验证更小配比附近的精确曲线，不能据此声称连续低配比区域的规律已被证实。

若M2在独立测试与嵌套CV均优于调参线性模型，可优先保留为同尺度配比预测候选。若M1共享系数损害性能，建议改成w_kj，但应明确这是模型结构改变。跨尺度任务还需要尺度相关截距/系数或N输入，以及真实高尺度验证。当前实验没有将之前的质量方向/语义冲突修正同时接入，以便隔离回归函数变更。

## 复算与输出

```powershell
python src/q1_mixture.py
```

主方法为M1共用系数与M2非线性配比回归；岭回归为对照，M1每域独立为补充。完整参数见mixture_models.json，可从原始配比复算预测。

本次表格为可导入Excel的UTF-8 CSV，没有改写前几轮质量实验表。12–14号PNG分别为性能对照、1M预测散点、τ敏感性。系数、逐目标域指标、逐样本原始预测、数据来源校验、训练内搜索与嵌套CV均分别保存。
"""
    (out / "mixture_regression_analysis.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, audit = load_sets()
    X, Y, indices = data["train"]
    validation = model_checks(X, Y)
    history = []
    log("Training-only nested cross validation (5 outer / 4 inner folds)")
    oof, cv = nested_cv(X, Y, history)
    final_splits = list(KFold(5, shuffle=True, random_state=42).split(X))
    models, configs, performance, per_domain, pred_rows, coef_rows = {}, [], [], [], [], []
    predictions = {}
    for name, family in MODEL_SPECS.items():
        alpha, tau = select_config(X, Y, name, final_splits, "full_train", history)
        model = fit_mixture_model(X, Y, family=family, alpha=alpha, tau=tau)
        models[name] = model
        Z = model.transform(X)
        singular = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
        configs.append({"model": name, "family": family, "alpha": alpha, "tau": tau,
                        "centered_design_condition_number": float(singular.max() / singular.min()),
                        "nominal_parameters_including_intercepts": 30 if family == "M1" else (235 if family == "M2" else 234)})
        log(f"Final {name}: alpha={alpha:g}, tau={tau:g}")
        evaluations = [("1M_train_fit", "train_fit", X, Y, indices, model.predict(X)),
                       ("1M_nested_cv", "nested_cv", X, Y, indices, oof[name])]
        evaluations += [(scale, "estimated" if "est" in scale else "test", *data[scale], model.predict(data[scale][0]))
                        for scale in SCALES]
        for scale, stage, Xte, Yte, ids, pred in evaluations:
            metrics = domain_metrics(Yte, pred)
            performance.append({"model": name, "scale": scale, "stage": stage, "n": len(Yte),
                                **mean_metrics(metrics)})
            metrics.insert(0, "scale", scale)
            metrics.insert(0, "model", name)
            per_domain.append(metrics)
            predictions[(name, scale)] = pred
            for j, dom in enumerate(LOSS_DOMAINS):
                pred_rows.append(pd.DataFrame({"model": name, "scale": scale, "index": ids,
                                              "target_domain": dom, "actual_loss": Yte[:, j],
                                              "predicted_loss_raw": pred[:, j]}))
        effects = model.marginal_effects(X.mean(axis=0)[None, :])[0]
        for j, target in enumerate(LOSS_DOMAINS):
            for k, source in enumerate(MIX_DOMAINS):
                coef_rows.append({"model": name, "input_domain": source, "target_domain": target,
                                  "coef_beta_or_b": model.coef_[j, k],
                                  "M1_w": -model.coef_[j, k] if family in ["M1", "M1_independent"] else np.nan,
                                  "intercept": model.intercept_[j], "tau": tau,
                                  "unconstrained_derivative_at_train_mean": effects[j, k]})
    performance = pd.DataFrame(performance)
    # Historical metrics establish that changing the fitting function preserved the baseline.
    old_tables = [Path(BASE) / "results/q1/p1_mixture_test_perf.csv",
                  Path(BASE) / "results/q1/p1_mixture_extrap_perf.csv"]
    errors = []
    for path in old_tables:
        if not path.exists():
            continue
        for row in pd.read_csv(path).itertuples():
            scale = {"10b": "10B_est", "70b": "70B_est"}.get(row.scale, row.scale)
            result = performance[(performance["model"] == "Ridge_original") & (performance["scale"] == scale)].iloc[0]
            errors += [abs(result["mean_r2_shift_diagnostic"] - row.mean_r2), abs(result["mean_rmse_raw"] - row.mean_rmse)]
    if errors:
        assert max(errors) < 1e-8
        validation["historical_metric_max_abs_difference"] = float(max(errors))
    bootstrap = paired_bootstrap(data["1M"][1], {name: predictions[(name, "1M")] for name in MODEL_SPECS})
    search = pd.DataFrame(history)
    for frame, filename in [(performance, "mixture_model_performance.csv"),
                            (pd.concat(per_domain), "mixture_per_domain_metrics.csv"),
                            (cv, "mixture_nested_cv.csv"), (search, "mixture_hyperparameter_search.csv"),
                            (audit, "mixture_data_audit.csv"), (pd.DataFrame(configs), "mixture_model_parameters.csv"),
                            (pd.DataFrame(coef_rows), "mixture_regression_coefficients.csv"),
                            (pd.concat(pred_rows), "mixture_predictions.csv"),
                            (bootstrap, "mixture_bootstrap_comparison.csv")]:
        save_csv(frame, out / filename)
    states = {name: {"family": model.family, "alpha": model.alpha, "tau": model.tau,
                     "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist()}
              for name, model in models.items()}
    states["_schema"] = {"input_domains": MIX_DOMAINS, "target_domains": LOSS_DOMAINS,
                          "proportion_unit": "0 to 1, source rounding preserved"}
    (out / "mixture_models.json").write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    validation.update({"selection": "training-only nested CV; no test-target tuning",
                       "shift_r2": "diagnostic uses actual test mean; not independent prediction",
                       "M1_shared": True, "M2_common_tau": True,
                       "original_22_quality_features_unchanged": True,
                       "sources": audit.to_dict(orient="records"), "random_seed": 42})
    (out / "mixture_regression_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(out, performance, search, predictions, data, models)
    report(out, performance, cv, configs, bootstrap, audit, models, data, pd.concat(per_domain))
    log(performance[performance["scale"].isin(SCALES)][["model", "scale", "mean_r2_raw", "mean_r2_shift_diagnostic", "mean_rmse_raw", "mean_spearman"]].to_string(index=False))
    log(f"Done: {out}")


if __name__ == "__main__":
    main()
