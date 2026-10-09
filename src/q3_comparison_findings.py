"""Produce the three requested Q3 comparisons from saved runs, without rerunning optimizers."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import BASE


def comparison_figures(out, runs, best):
    """Use absolute residuals: a signed mean must not hide reference differences."""
    colors = {"PSO": "#2463a0", "SLSQP": "#b56528"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, model in zip(axes, ("additive", "interaction")):
        for algorithm, marker in [("PSO", "o"), ("SLSQP", "s")]:
            frame = runs[(runs.model == model) & (runs.recipe_mode == "free")
                         & (runs.cost_form == "power") & (runs.L_ctx == 4096)
                         & (runs.algorithm == algorithm)].copy()
            frame["absolute_reference_difference"] = np.abs(frame.gap_to_reference)
            grouped = frame.groupby("budget_FLOPs").absolute_reference_difference.mean()
            ax.plot(grouped.index, np.maximum(grouped.values, 1e-16), marker=marker,
                    label=algorithm, color=colors[algorithm])
        ax.set(xscale="log", yscale="log", xlabel="Budget (FLOPs)",
               ylabel="Mean absolute difference from reference", title=model.capitalize())
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(out / "02_solver_gap.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, column, label in zip(axes.flat, ["N_B", "D_B", "Q", "Loss"],
                                 ["Parameters (B)", "Training tokens (B)", "Quality Q", "Predicted loss"]):
        for model, marker, color in [("additive", "o", "#2463a0"), ("interaction", "s", "#b56528")]:
            frame = best[best.model == model].sort_values("budget_FLOPs")
            ax.plot(frame.budget_FLOPs, frame[column], marker=marker, color=color, label=model)
        ax.set(xscale="log", xlabel="Budget (FLOPs)", ylabel=label)
        if column in ("N_B", "D_B"):
            ax.set_yscale("log")
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Actual PSO optima: power cost, context 4096, fixed recipe")
    fig.savefig(out / "07_resource_model_comparison.png", dpi=180); plt.close(fig)


def markdown(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.7g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def write_comparison_findings(out):
    runs = pd.read_csv(out / "optimization_runs.csv")
    refs = pd.read_csv(out / "reference_optima.csv")
    pair = pd.read_csv(out / "fixed_vs_free_recipe.csv")
    matched = pd.read_csv(out / "matched_evaluations.csv")
    config = json.loads((out / "model_parameters.json").read_text(encoding="utf-8"))
    keys = ["cost_form", "budget_FLOPs", "L_ctx"]
    fields = ["N_B", "D_B", "Q", "Loss", "s_train", "s_quality", "s_attn"]
    comparison = refs[refs.model == "interaction"][keys + fields].merge(
        refs[refs.model == "additive"][keys + fields], on=keys, suffixes=("_interaction", "_additive"))
    comparison["interaction_minus_additive_Loss"] = comparison.Loss_interaction - comparison.Loss_additive
    comparison["N_interaction_over_additive_minus_one"] = comparison.N_B_interaction / comparison.N_B_additive - 1
    comparison["D_interaction_over_additive_minus_one"] = comparison.D_B_interaction / comparison.D_B_additive - 1
    comparison["Q_interaction_minus_additive"] = comparison.Q_interaction - comparison.Q_additive
    comparison.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    algorithms = runs.groupby(["recipe_mode", "algorithm"], as_index=False).agg(
        runs=("Loss", "size"), mean_gap=("gap_to_reference", "mean"), max_gap=("gap_to_reference", "max"),
        hit_rate=("reference_hit", "mean"), median_seconds=("seconds", "median"),
        median_evaluations=("evaluations", "median"), median_successful_starts=("successful_starts", "median"))
    algorithms.to_csv(out / "solver_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    primary = runs[(runs.cost_form == "power") & (runs.L_ctx == 4096) & (runs.recipe_mode == "fixed")
                   & (runs.algorithm == "PSO")]
    best = primary.loc[primary.groupby(["model", "budget_FLOPs"]).Loss.idxmin()]
    best = best.sort_values(["budget_FLOPs", "model"])
    case_table = best[["model", "budget_FLOPs", "N_B", "D_B", "Q", "Loss", "s_train", "s_quality", "s_attn"]]
    comparison_figures(out, runs, best)
    paired_keys = ["model", "cost_form", "budget_FLOPs", "L_ctx", "recipe_mode", "seed"]
    timing = runs[runs.algorithm == "PSO"].merge(runs[runs.algorithm == "SLSQP"], on=paired_keys, suffixes=("_PSO", "_SLSQP"))
    timing["SLSQP_over_PSO_seconds"] = timing.seconds_SLSQP / timing.seconds_PSO
    timing["PSO_over_SLSQP_evaluations"] = timing.evaluations_PSO / timing.evaluations_SLSQP
    ratios = timing.groupby("recipe_mode")[["SLSQP_over_PSO_seconds", "PSO_over_SLSQP_evaluations"]].median()
    time_finding = ("本机自由配比实验中PSO的批量向量计算带来了耗时优势"
                    if ratios.loc['free', 'SLSQP_over_PSO_seconds'] > 1
                    else "本机自由配比实验中PSO没有表现出耗时优势")
    fixed_finding = ("固定配比的低维问题也出现PSO耗时较少"
                     if ratios.loc['fixed', 'SLSQP_over_PSO_seconds'] > 1
                     else "固定配比的低维问题中SLSQP耗时较少")
    available = matched.dropna(subset=["PSO_Loss_at_match"])
    count_win = int((available.PSO_Loss_at_match < available.SLSQP_Loss - 1e-6).sum())
    count_tie = int((np.abs(available.PSO_Loss_at_match - available.SLSQP_Loss) <= 1e-6).sum())
    count_loss = int((available.PSO_Loss_at_match > available.SLSQP_Loss + 1e-6).sum())
    mixture_stats = pair.groupby("algorithm", as_index=False).agg(
        max_abs_free_fixed_loss=("free_minus_fixed_Loss", lambda v: np.abs(v).max()),
        max_H_excess=("H_excess_over_star", "max"), max_p_L1=("p_L1_from_star", "max"),
        max_abs_N_relative_change=("N_relative_change", lambda v: np.abs(v).max()),
        max_abs_D_relative_change=("D_relative_change", lambda v: np.abs(v).max()),
        max_abs_Q_change=("Q_change", lambda v: np.abs(v).max()))
    interaction_wins = int((comparison.interaction_minus_additive_Loss < -1e-6).sum())
    low = comparison[(comparison.cost_form == "power") & (comparison.L_ctx == 4096)
                     & (comparison.budget_FLOPs == 1e19)].iloc[0]
    high = comparison[(comparison.cost_form == "power") & (comparison.L_ctx == 4096)
                      & (comparison.budget_FLOPs == 1e24)].iloc[0]
    fit_data = pd.read_csv(Path(BASE) / "results/q2/fit_data_and_splits.csv")
    fit_bounds = fit_data[["N_params_B", "D_tokens_B", "Q_score"]].agg(["min", "max"])
    # Identify extrapolation without silently adding new constraints.
    for name, column in [("N_B", "N_params_B"), ("D_B", "D_tokens_B"), ("Q", "Q_score")]:
        refs[f"{name}_outside_fit_range"] = ~refs[name].between(fit_bounds.loc["min", column], fit_bounds.loc["max", column])
    refs.to_csv(out / "optimization_range_audit.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    findings = f"""# 第三问实验结论

默认入口已改为粒子群PSO；SLSQP作为同问题的对照求解器保留。本次使用第二问固定E的两个配比扩展模型，并让p在17维单纯形上优化。共{len(runs)}次完整求解（每个算法{len(runs)//2}次），每个场景重复{config['repetitions']}次。

## 1. 两公式加入条件后的优化效果

additive为`E+A/N^a+B/D^b+C(1-Q)^g+lambda*H(p)`；interaction为`E+A/N^a+B/D^b+C(1-Q)^g/D^d+lambda*H(p)`。固定p时H(p)取H_star，自由p时计算完整H(p)。这次不是重新拟合参数。

下面是实际PSO求得的最好配置（幂函数质量成本、上下文4096、固定p_star）。N、D单位均为十亿，s列为训练、质量、注意力成本份额。

{markdown(case_table)}

在此代表场景的低预算下，interaction相对additive的N变化为{low.N_interaction_over_additive_minus_one:.2%}、D变化为{low.D_interaction_over_additive_minus_one:.2%}、Q变化为{low.Q_interaction_minus_additive:.5f}；高预算时分别为{high.N_interaction_over_additive_minus_one:.2%}、{high.D_interaction_over_additive_minus_one:.2%}。两公式均在中高预算选择Q=1。interaction倾向配置更多参数、较少token；此处趋势是当前拟合参数的计算结果，不是任何interaction模型的必然性质。

全部18个成本/预算/上下文组合中，interaction预测最小Loss较低的有{interaction_wins}个，additive较低的有{len(comparison)-interaction_wins}个。两个模型的损失差来自不同响应面及参数；预测值更低不代表真实训练策略更好。当前第二问分组交叉验证RMSE约0.0584/0.0593，远大于部分优化损失差，且没有最优配置的直接训练验证。完整成对差异见model_comparison.csv。

## 2. p重新优化有没有收益

在当前公式内没有理论收益。因为lambda为正，成本不依赖p，而且Q与p没有耦合：

`min_(N,D,Q,p) L = min_(N,D,Q) L_scale + lambda*min_p H(p)`。

第一问已求得H的单纯形最优p_star，因此开放p后仍回到p_star，最优N/D/Q也不变。下表给出自由/固定配比的最大数值差；这些量属于求解精度，不能解释为配比收益。

{markdown(mixture_stats)}

Q仍可以在配比不变时通过域内筛选而变化。若希望最优p随预算、Q变化，必须另行构造并校准Q(p)、域供应约束、任务权重或配比成本；本次没有虚构这些关系。B中没有配比观测，lambda*H_star的拟合实质是常数截距调整，所以开放p的分析依赖“A中的H形状可迁移到B、由lambda缩放”的额外假设。

## 3. PSO相比SLSQP有没有优势

SLSQP指序列最小二乘规划，是本次约束优化对照算法，与第二问回归拟合的最小二乘不同。两算法使用同一套精确可行的资源参数化：由单调性消去D，再优化N/Q和可选的p；PSO没有SLSQP后处理，也没有用已知p_star初始化自由配比。

{markdown(algorithms)}

同场景、同seed的耗时比中位数：固定p时SLSQP/PSO={ratios.loc['fixed','SLSQP_over_PSO_seconds']:.3g}，自由p时={ratios.loc['free','SLSQP_over_PSO_seconds']:.3g}；大于1表示PSO较快。对应目标调用量比PSO/SLSQP分别为{ratios.loc['fixed','PSO_over_SLSQP_evaluations']:.3g}、{ratios.loc['free','PSO_over_SLSQP_evaluations']:.3g}。

{time_finding}；{fixed_finding}。调用量与收敛精度需分别看表，不能把耗时较少理解为需要更少目标评估。当前设置下的1e-6参考命中率已逐组列出，不能据此宣称PSO更准确或必然更稳定。本对照SLSQP使用有限差分估计梯度；若提供解析梯度或先利用p可分离结构求解，速度排名可能变化。实际昂贵训练评估也不会自动获得此处的NumPy批量计算优势。

调用量匹配的补充诊断覆盖{len(available)}/{len(matched)}次：以Loss差1e-6为界，PSO较好{count_win}次、接近{count_tie}次、较差{count_loss}次。它是在SLSQP总调用量内截取PSO完整批次，不等于严格相同墙钟预算，未覆盖的场景是SLSQP调用量少于PSO初始化粒子数。较多调用下的收敛结果不代表少量调用时也占优。

## 条件核查及结果适用范围

- 严格使用`10^18*N_B*D_B*(6+eta*L_ctx)`的训练与注意力成本，质量成本为`10^9*D_B*[f(Q)-f(0.4)]_+`。原代码本来已有正确单位换算。
- 主对照采用截图的2048/4096。截图称30000超过C7支持范围，但本项目完整C7还包含8192、32768、131072；因此30000是未直接记录的理论相等点，并非超出整张C7范围。每个具体架构仍须遵守自身最大窗口，不能将任意连续N的数学解当作已存在的架构。
- 注意力/训练成本比仅由上下文决定：2048时6.8267%，4096时13.6533%；理论相等长度为30000。更长C7窗口仅放在context_sensitivity.csv做模型敏感性分析。
- 第二问实际拟合支持N=0.07–11.97B、D=10–600B、Q=0.1–1。这里低预算解的D低于训练范围，高预算解的N/D高于训练范围，均属于外推；中预算解在这三个变量的范围内，但这仍不是最优配置的真实观测验证。逐场景标记见optimization_range_audit.csv。
- 第二问使用的B6/B7是半合成补充，A配比规律来自第一问导出。计算最优值用于模型内情景比较，不能当作真实训练实验或证明跨来源规律成立。

公式、成本函数、边界、PSO参数与来源详见optimization_analysis.md；独立复算和网格加密检查见numerical_validation.json。运行`python src/q3_optimization.py`重现，单独重新生成本结论用`python src/q3_comparison_findings.py`。条件引入前后情景对照见constraint_comparison.md与constraints_before_after.csv。
"""
    (out / "comparison_findings.md").write_text(findings, encoding="utf-8")
    print(f"Comparison findings: {out / 'comparison_findings.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q3")
    args, _ = parser.parse_known_args()
    write_comparison_findings(args.output_dir.resolve())
    return args.output_dir.resolve()


if __name__ == "__main__":
    main()
