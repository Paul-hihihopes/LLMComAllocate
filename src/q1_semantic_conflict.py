# -*- coding: utf-8 -*-
"""Q1 semantic conflict penalties; retain all existing 22 features.

Run: python src/q1_semantic_conflict.py
The attached document does NOT define an entropy penalty. An independently
specified, nonnegative entropy penalty can be supplied with --entropy-penalty-csv
(record_index, baseline, entropy_penalty), with one row per record and baseline.
Without it, only semantic correction is computed and no sequential result is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib import font_manager

import quality_features as original
import q1_domain_direction as direction
from common import BASE

# Document row 9 contains two pairs, so there are 11 pairs altogether.
# Abbreviated names are mapped only to the identically defined existing fields.
# QuRating components are NOT aliases for the aggregated qurater field.
PAIRS = [
    ("fineweb_edu", "qurater_educational_value", "强"),
    ("qurater_required_expertise", "modernbert_professionalism", "强但专业不等于优质"),
    ("fluency_en", "qurater_writing_style", "较弱"),
    ("fluency_en", "modernbert_readability", "条件性"),
    ("qurater_writing_style", "modernbert_readability", "条件性"),
    ("modernbert_cleanliness", "qurater_writing_style", "条件性"),
    ("rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram", "较强"),
    ("rps_doc_frac_unique_words", "rps_doc_unigram_entropy", "须控制长度"),
    ("rps_doc_frac_chars_top_2gram", "rps_doc_frac_unique_words", "须控制长度和文本类型"),
    ("rps_doc_frac_chars_top_3gram", "rps_doc_unigram_entropy", "须控制长度和文本类型"),
    ("rps_lines_ending_with_terminal_punctution_mark", "fluency_en", "仅限可比较的叙述文本"),
]


def pair_arrays(X, fields, pairs=PAIRS):
    """Document's C and S; globally absent fields and per-row NaNs are skipped."""
    index = {field: i for i, field in enumerate(fields)}
    retained, audit = [], []
    for num, (a, b, note) in enumerate(pairs, 1):
        missing = [field for field in (a, b) if field not in index]
        audit.append({"pair_id": num, "field_j": a, "field_k": b,
                      "document_condition": note, "included": not missing,
                      "reason": "字段均存在" if not missing else "跳过不存在字段：" + ", ".join(missing)})
        if not missing:
            retained.append((num, a, b))
    n, m = len(X), len(retained)
    valid = np.zeros((n, m), dtype=bool)
    C = np.zeros((n, m), dtype=bool)
    S = np.zeros((n, m), dtype=float)
    for col, (_, a, b) in enumerate(retained):
        u, v = X[:, index[a]], X[:, index[b]]
        valid[:, col] = np.isfinite(u) & np.isfinite(v)
        C[:, col] = valid[:, col] & (np.maximum(u, v) >= 0.8) & (np.minimum(u, v) <= 0.2)
        # max with 0 only removes floating-point cancellation at the .6 boundary.
        gap = np.nan_to_num(np.abs(u - v), nan=0.0)
        S[:, col] = np.where(C[:, col], np.maximum(0.0, (gap - 0.6) / 0.4), 0.0)
    return retained, pd.DataFrame(audit), valid, C, S


def conflict_rate(C, valid, domains=None):
    if domains is None:
        return np.divide(C.sum(axis=0), valid.sum(axis=0),
                         out=np.zeros(C.shape[1]), where=valid.sum(axis=0) > 0)
    rates = np.zeros_like(C, dtype=float)
    for domain in np.unique(domains):
        mask = domains == domain
        rates[mask] = conflict_rate(C[mask], valid[mask])
    return rates


def semantic_index(C, S, rates, alpha):
    """The document already weights pairs by 1-(1-alpha)*p_jk.

    |P| is the number of retained schema pairs. Unavailable observations contribute
    zero; the document's fixed denominator is retained, not changed to |P_i|.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0,1]")
    if C.shape[1] == 0:
        return np.zeros(len(C))
    return np.sum(C * S * (1 - (1 - alpha) * rates), axis=1) / C.shape[1]


def sequential_correction(q0, entropy, semantic, lambda_entropy, lambda_semantic):
    if lambda_entropy < 0 or lambda_semantic < 0:
        raise ValueError("Correction coefficients must be nonnegative")
    after_entropy = q0 - lambda_entropy * entropy
    corrected = after_entropy - lambda_semantic * semantic
    return after_entropy, corrected


def coefficient_limit(q, penalty):
    positive = penalty > 0
    return float(np.min(q[positive] / penalty[positive])) if positive.any() else float("inf")


def load_entropy(path, meta, baseline):
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"record_index", "baseline", "entropy_penalty"}
    if not required <= set(frame):
        raise ValueError(f"Entropy file requires {sorted(required)}")
    frame = frame[frame["baseline"] == baseline]
    if frame["record_index"].duplicated().any() or set(frame["record_index"]) != set(meta["record_index"]):
        raise ValueError("Entropy file must contain each record_index exactly once for each baseline")
    entropy = frame.set_index("record_index").loc[meta["record_index"], "entropy_penalty"].to_numpy(dtype=float)
    if not np.isfinite(entropy).all() or np.any(entropy < 0):
        raise ValueError("Entropy penalties must be finite and nonnegative")
    return entropy


def summarize(meta, scores, baseline, penalty_name):
    rows = []
    for dom in direction.DOMAINS:
        mask = meta["domain"].to_numpy() == dom
        s = scores.loc[mask]
        reliability = 1 / (s["old_conflict"].to_numpy() + 0.05)
        q0, old, new = [s[k].to_numpy() for k in ["Q0", "Q_old_trimmed", penalty_name]]
        k = max(1, int(np.ceil(len(s) / 10)))
        old_top = set(np.argsort(old, kind="stable")[-k:])
        new_top = set(np.argsort(new, kind="stable")[-k:])
        row = {"baseline": baseline, "domain": dom, "n": len(s),
               "n_valid_old_aggregation": int(np.isfinite(reliability).sum()),
               "Q0": direction.weighted_mean(q0, reliability),
               "Q_old_trimmed": direction.weighted_mean(old, reliability),
               "Q_new": direction.weighted_mean(new, reliability),
               "Q0_plain": q0.mean(), "Q_old_plain": old.mean(), "Q_new_plain": new.mean(),
               "semantic_mean": s["semantic_index"].mean(),
               "semantic_any_rate": s["conflict_pair_count"].gt(0).mean(),
               "penalized_rate": s["semantic_index"].gt(1e-15).mean(),
               "semantic_deduction_mean": s["semantic_deduction"].mean(),
               "old_increased_rate": np.mean(old > q0 + 1e-12),
               "old_decreased_rate": np.mean(old < q0 - 1e-12),
               "new_vs_old_spearman": spearmanr(old, new).statistic,
               "top10_overlap": len(old_top & new_top) / k,
               "Q_new_min": new.min(), "Q_new_max": new.max()}
        row["delta_new_old"] = row["Q_new"] - row["Q_old_trimmed"]
        rows.append(row)
    table = pd.DataFrame(rows)
    for key in ["Q0", "Q_old_trimmed", "Q_new"]:
        table[key + "_rank"] = table[key].rank(ascending=False, method="min").astype(int)
    return table


def plot_results(out, summary, pair_stats, scores_by_base, meta, sensitivity):
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next(f for f in ["Microsoft YaHei", "SimHei", "SimSun"] if f in available)
    plt.rcParams.update({"font.sans-serif": [font], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#3274A1", "#8499A4", "#E2842B"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    for ax, base in zip(axes, ["original", "domain"]):
        t = summary[summary["baseline"] == base].set_index("domain").loc[direction.DOMAINS]
        x = np.arange(7)
        new_label = "连续熵与语义修正" if "Q_sequential" in scores_by_base[base] else "文档语义修正（未含熵项）"
        for offset, key, label, color in zip([-.26, 0, .26], ["Q0", "Q_old_trimmed", "Q_new"],
                                             ["未修正加权Q", "原截尾修正", new_label], colors):
            ax.bar(x + offset, t[key], .25, label=label, color=color)
        ax.set_xticks(x, direction.DOMAIN_LABELS, rotation=12)
        ax.set_ylim(0, 1)
        ax.set_ylabel("领域Q，固定原冲突倒数聚合")
        ax.set_title("统一指标方向" if base == "original" else "上一轮分领域指标方向")
    axes[0].legend(ncol=3, loc="upper right")
    fig.savefig(out / "08_semantic_quality_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, base in zip(axes, ["original", "domain"]):
        t = pair_stats[(pair_stats["baseline"] == base) & (pair_stats["scope"] == "domain")]
        matrix = t.pivot(index="pair_id", columns="domain", values="conflict_rate").reindex(columns=direction.DOMAINS)
        im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(7), direction.DOMAIN_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix)), [f"对 {i}" for i in matrix.index])
        for i in range(len(matrix)):
            for j in range(7):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=9,
                        color="white" if value > .5 else "#1A334A")
        ax.set_title("统一方向" if base == "original" else "分领域方向")
    fig.colorbar(im, ax=axes, label="冲突发生率", shrink=.8)
    fig.savefig(out / "09_semantic_pair_rates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, base in zip(axes, ["original", "domain"]):
        s = scores_by_base[base]
        ax.hist(s["Q_old_trimmed"] - s["Q0"], bins=70, alpha=.7, color=colors[1], label="原截尾：可加分或扣分")
        ax.hist(s["Q_new"] - s["Q0"], bins=70, alpha=.7, color=colors[2], label="语义修正：只扣分")
        ax.set_yscale("log")
        ax.set_xlabel("修正后Q − 未修正Q")
        ax.set_ylabel("记录数（对数）")
        ax.set_title("统一方向" if base == "original" else "分领域方向")
        ax.legend()
    fig.savefig(out / "10_semantic_deduction_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, base in zip(axes, ["original", "domain"]):
        t = sensitivity[(sensitivity["baseline"] == base) & (sensitivity["rate_scope"] == "pooled")]
        for alpha, g in t.groupby("alpha"):
            ax.plot(g["requested_lambda"], g["mean_semantic_deduction"], marker="o", label=f"α={alpha:g}")
        ax.set_xlabel("请求的λ（超过安全上限时统一缩减）")
        ax.set_ylabel("全量平均语义扣分")
        ax.set_title("统一方向" if base == "original" else "分领域方向")
        ax.legend()
    fig.savefig(out / "11_semantic_parameter_sensitivity.png", dpi=180)
    plt.close(fig)


def length_diagnostics(meta, X):
    """Diagnostic, not a change to the penalty: rank associations with length."""
    rows = []
    for dom in direction.DOMAINS:
        mask = meta["domain"].to_numpy() == dom
        values = np.column_stack([X[mask, original.ALL_IND.index("rps_doc_frac_unique_words")],
                                  X[mask, original.ALL_IND.index("rps_doc_unigram_entropy")],
                                  meta.loc[mask, "rps_doc_word_count"].to_numpy()])
        values = values[np.isfinite(values).all(axis=1)]
        r = np.asarray(spearmanr(values, axis=0).statistic)
        rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
        denominator = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        rows.append({"domain": dom, "n": len(values), "rho_unique_entropy": rxy,
                     "rho_unique_word_count": rxz, "rho_entropy_word_count": ryz,
                     "partial_rho_given_word_count": (rxy - rxz * ryz) / denominator if denominator > 1e-12 else np.nan,
                     "unique_normalized_median": np.median(values[:, 0]),
                     "entropy_normalized_median": np.median(values[:, 1]),
                     "word_count_median": np.median(values[:, 2])})
    return pd.DataFrame(rows)


def write_report(out, summary, pair_stats, audit, manifest, sensitivity, lengths):
    cols = ["domain", "Q0", "Q_old_trimmed", "Q_new", "delta_new_old", "Q_old_trimmed_rank", "Q_new_rank", "penalized_rate"]
    sections = []
    for base, title in [("original", "原统一方向"), ("domain", "上一轮分领域方向")]:
        sections.append(f"### {title}\n\n" + direction.markdown_table(summary[summary["baseline"] == base], cols, 6))
    included = audit[audit["included"]]
    skipped = audit[~audit["included"]]
    affected = summary[["baseline", "domain", "semantic_any_rate", "semantic_deduction_mean", "old_increased_rate", "new_vs_old_spearman", "top10_overlap"]]
    top_pairs = pair_stats[(pair_stats["scope"] == "pooled")].sort_values("mean_strength", ascending=False)
    dom_result = summary[summary['baseline'] == 'domain'].set_index('domain')
    arxiv, book, github = [dom_result.loc[d] for d in ['arxiv', 'book', 'github']]
    rates8 = pair_stats[(pair_stats['baseline'] == 'domain') & (pair_stats['pair_id'] == 8) & (pair_stats['scope'] == 'domain')].set_index('domain')
    text = f"""# 第一问 文档语义冲突扣分与原截尾修正对照

## 完成范围与结论

已实现文档语义冲突公式，并在A1–A3同一批272,505条记录、同一22项特征上计算。主结果使用λ={manifest['parameters']['lambda_semantic_requested']}、α={manifest['parameters']['alpha']}，这些参数为演示选择而非训练拟合。分别在原统一方向和上一轮分领域方向中对照，避免把方向变化误认为冲突修正效果。原截尾算法、22特征、归一化、列表均值、指标权重均保持一致。

附件只有语义修正公式，没有熵修正扣分公式。项目原熵权法决定指标权重，不是额外的熵惩罚。{('本次未提供熵项，数值结果仅是语义修正；尚不能称为完成熵与语义的连续修正。' if not manifest['entropy_available'] else '本次熵项来自用户指定CSV；逐行先减去加权熵项，再减去加权语义项。')}

新方法作为“不一致性扣分”比截尾合成更直接：只有指定指标对发生极端冲突才扣分，能逐对解释扣分来源。它尚未被证明是更准确的真实质量评价。评分更低、范围合格或排名稳定，都不是训练收益证据。

## 文档公式与加权扣减

u均为当前流程归一化且统一方向后的特征，P仅包含实际存在的指标对。

```text
C_i,jk = 1[max(u_ij,u_ik) >= 0.8 且 min(u_ij,u_ik) <= 0.2]
S_i,jk = C_i,jk * (|u_ij-u_ik| - 0.6) / 0.4
p_jk = sum_i(C_i,jk) / n_jk
T_i = (1/|P|) * sum_(j,k in P)(C_i,jk*S_i,jk*[1-(1-alpha)*p_jk])
Q_after_entropy = Q0 - lambda_entropy * H_i
Q_sequential = Q_after_entropy - lambda_semantic * T_i
```

H_i须由独立定义的熵修正公式提供。没有H_i时只计算Q_semantic_only=Q0−λT_i，不把未知熵项写成已完成的零惩罚。语义项按文档中的发生率因子加权，指标加权已经体现在Q0中，不另外引入文档未给出的w_j*w_k。λ是语义扣分系数，α控制常见冲突被折减的程度。α=1不折减；α=0按1−p折减，冲突越常见扣分越轻。该设定可能保护领域固有差异，也可能掩盖普遍性错误，不能理解为“冲突越常见越应罚”。

极端边界0.8/0.2触发C=1但S=0，故“冲突发生”与“实际扣分”比例可能不同。主方法的p在全量合并样本计算，严格对应文档；另计算域内p的敏感性。未引入相关系数、判别强度等级或人工质量标签。

## 字段映射与跳过规则

文档10行中一行包含两对，共11对；保留6对，跳过5对。只将文档缩写映射到语义完全一致的现有字段。QuRating教育、专业、写作风格子维度不是当前单列qurater，因此没有拆分、增加或代替特征。

{direction.markdown_table(included, ['pair_id','field_j','field_k','document_condition'])}

跳过项：

{direction.markdown_table(skipped, ['pair_id','field_j','field_k','reason'])}

逐样本若任一端非有限则跳过该对，该观测不参与n_jk。|P|保持文档的全局有效字段对数量，不随样本改变；跳过的观测贡献为0。某对无有效观测时p记0且不产生惩罚。本次六对字段无缺失，19个ModernBERT专业/推理缺失不进入有效冲突对；未因此插补或删除记录。

中性方向取0.5，天然不能满足高低极端条件。同向关系需要在当前方向体系下成立：GitHub标点项为负向而流畅度为正向，文档“仅限可比较叙述文本”的条件没有可用文本类型标记，本次保留该对并标注适用性风险，未擅自变更公式。ad_en、fluency_en仍是原列表均值，不能解释为正类概率。

## 修正前后领域结果

这里Q_old_trimmed是原代码对每条样本删除最低3项和最高3项、重归一化剩余指标权重后的分数，区别于原主输出的未截尾Q_weighted。旧截尾过程没有用|Q_C−Q_F|决定是否修正，它作用于全部样本。新方法直接从同一Q0扣分，不在旧截尾Q上再次扣除，以便公平对照。

领域聚合全部固定使用各自方向体系下原来的1/(|Q_C−Q_F|+0.05)，包括相同有效记录集合。未同时替换领域聚合权重，否则会把样本扣分与聚合变化混在一起。另输出普通均值覆盖全部记录。新语义修正本身能覆盖全部记录，但旧聚合仍跳过旧冲突缺失项。

{chr(10).join(sections)}

## 样本排序与修正机制

{direction.markdown_table(affected, digits=6)}

old_increased_rate表明原截尾会为部分样本加分。语义方法及非负熵项的连续扣减均不会高于Q0。截尾法是稳健聚合，不是单向惩罚，不能单凭这一区别说原方法计算错误。

全量冲突对统计（按平均强度排序）：

{direction.markdown_table(top_pairs, ['baseline','pair_id','n_valid','n_conflict','conflict_rate','mean_strength'], 6)}

## 是否更加合理

### 本次数据提供的直接证据

上一轮分领域方向下，GitHub原截尾Q为{github['Q_old_trimmed']:.6f}，语义修正为{github['Q_new']:.6f}；ArXiv为{arxiv['Q_old_trimmed']:.6f}和{arxiv['Q_new']:.6f}；Book为{book['Q_old_trimmed']:.6f}和{book['Q_new']:.6f}。语义扣分后的七域排序与未修正加权Q一致，原截尾则把Book和GitHub的前两名次对换。这个名次差异来自评价定义，不能决定谁更优。

语义修正给ArXiv的{arxiv['penalized_rate']:.2%}和Book的{book['penalized_rate']:.2%}记录扣分，远高于GitHub的{github['penalized_rate']:.2%}。主要来源是“不同单词占比—词频熵”和“反向重复度—不同单词占比”。仅第一对在ArXiv和Book的发生率已达{rates8.loc['arxiv','conflict_rate']:.2%}和{rates8.loc['book','conflict_rate']:.2%}，这些指标在长文档中不必同尺度变化，不能把高发生率解释为大批论文或书籍质量差。

长度诊断如下，全部使用当前特征；这些诊断没有用于改变本次扣分公式：

{direction.markdown_table(lengths, digits=4)}

rho为Spearman相关；partial_rho_given_word_count是对词数秩做线性控制后的偏秩相关，不能消除全部长度/文本类型混杂，也不是因果识别。不同单词占比是比例，词频熵衡量分布的信息量，两者不是同一个刻度的两次测量。若随长度呈相反变化，则采用0.8/0.2直接判冲突可能误罚；文档“须控制长度”的条件在主公式中尚未落实。因此目前更合理的结论是：新方法的扣分机制更可解释，但当前冲突对和阈值还不足以支持直接替代原方法。

优势：每次扣分都能对应一个预先声明的同向指标对，能区分无冲突与严重不一致；不会仅因单项特别高或低而删除有价值指标；连续两项扣分可分别审计，λ/α的含义清晰。旧截尾简单、无系数且能抑制极值，但无论冲突是否发生都去掉六项，可能同时删掉真实高质量信号，并为部分低分样本加分。

局限：6对仅覆盖少量特征，缺少QuRating维度使教育/专业/写作冲突无法检测，且没有覆盖内容质量与广告等所有潜在冲突。0.8/0.2依赖1%–99%归一化，语义对应指标的刻度不一定校准，低差距不能证明一致。若条件性同向关系在代码、数学、表格上不成立，扣分可能惩罚合理差异。重复覆盖的三对“重复性/多样性”关系共享指标，存在多次惩罚同一现象的可能。

发生率由样本分布决定：GitHub占78.44%，全量p易受代码域支配。因此额外输出域内p敏感性，但主结果不擅自换成域内估计。未建立跨域统一语义刻度，不应把领域名次当作训练配比的直接依据。

熵项定义未知，无法判断其与语义项是否重复计罚。若H_i本来衡量的也是指标离散或不一致，两项相加可能重复惩罚；应先明确H_i及系数，再用消融实验比较只熵、只语义、连续扣减和截尾。没有人工标签或训练验证损失，本次只能比较数值行为与结构合理性，不能证明新方法更准确。

建议：若目的是可解释的“不一致性扣分”，新方法适合保留为候选；若目的是总体语料质量，先盲评各域的高扣分/零扣分样本、验证每对的适用条件，并控制N、D、p及预算比较等量语料训练收益。λ和α应由这种验证选择，而不是为了得到希望的排名。

本次改变的是质量分，没有修改原始特征，因此并不使同一指标对的C或S减小，不能称为“消除了指标矛盾”。

## 参数范围与检查

所有系数非负；若请求的λ导致负分，会按所有样本共同的安全上限缩减至90%，不逐样本截断、不对结果再做Min–Max。熵扣分后若已为负则直接报错，要求调整熵系数。没有惩罚时λ上限为无穷。该缩减只保证数值范围，不是质量最优参数。分数允许边界[0,1]，不会人为挪动真实0/1来满足严格开区间。

{direction.markdown_table(pd.DataFrame(manifest['coefficient_checks']), digits=6)}

λ在0、0.05、0.1、0.2、0.5，α在0、0.5、1，以及合并/域内发生率均已计算敏感性。有效惩罚不增分、非冲突不惩罚、严重度范围、字段跳过、缺失跳过、边界、连续两次加权扣减已用有意义的数值样例检查。原始领域结果与既有p1_domain_quality.csv误差见manifest。

## 复算与文件

```powershell
python src/q1_semantic_conflict.py
# 确认熵公式并生成逐记录H后，才运行连续修正：
python src/q1_semantic_conflict.py --entropy-penalty-csv path/to/entropy.csv --lambda-entropy 1 --lambda-semantic 0.1 --alpha 0.5
```

外部熵文件须含record_index、baseline（original/domain）、entropy_penalty三列，每个baseline均完整覆盖所有record_index，不能以重复id连接。脚本不从该文件猜测熵公式；manifest记录输入SHA256及系数。Excel汇总未覆盖本次追加实验，使用以下新增CSV和报告。

- semantic_domain_comparison.csv：同一方向内未修正、原截尾、文档修正领域对照。
- semantic_pair_mapping.csv：11对映射及5对跳过原因。
- semantic_pair_statistics.csv：全量及分领域冲突率、严重度、系数。
- semantic_parameter_sensitivity.csv：λ、α、发生率分组敏感性。
- semantic_length_diagnostics.csv：多样性指标与词数的相关性、偏秩相关及中位数。
- semantic_sample_scores.csv.gz：每条记录全部中间量及修正分数。
- semantic_validation.json：特征、参数、缺失/熵状态、原结果对照及复算来源。
- 08–11 PNG：领域对照、各对发生率、加减分分布、参数敏感性。
"""
    (out / "semantic_analysis.md").write_text(text, encoding="utf-8")


def numerical_checks():
    X = np.array([[1., 0.], [.8, .2], [.79, .1], [np.nan, 0.], [.1, .9]])
    retained, audit, valid, C, S = pair_arrays(X, ["a", "b"], [("a", "b", "test"), ("a", "missing", "test")])
    assert len(retained) == 1 and not audit.loc[1, "included"]
    assert C[:, 0].tolist() == [True, True, False, False, True]
    assert np.allclose(S[:, 0], [1, 0, 0, 0, .5], atol=1e-12)
    assert np.isclose(conflict_rate(C, valid)[0], .75)
    assert np.allclose(semantic_index(C, S, np.array([.75]), .5), S[:, 0] * .625)
    assert not pair_arrays(X, ["a", "b"], [("c", "d", "test")])[0]
    after, new = sequential_correction(np.array([.8]), np.array([.1]), np.array([.2]), .5, .25)
    assert np.allclose(after, [.75]) and np.allclose(new, [.7])
    assert np.isclose(coefficient_limit(np.array([.1, .5]), np.array([.2, 0.])), .5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q1")
    parser.add_argument("--lambda-semantic", type=float, default=.1)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--entropy-penalty-csv", type=Path)
    parser.add_argument("--lambda-entropy", type=float, default=1.)
    args = parser.parse_args()
    if args.lambda_semantic < 0 or args.lambda_entropy < 0 or not 0 <= args.alpha <= 1:
        parser.error("lambda must be nonnegative and alpha must be within [0,1]")
    numerical_checks()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta, provenance = direction.load_data()
    raw = meta[original.ALL_IND].to_numpy(dtype=float)
    U = original.winsorize_minmax(raw)
    X0 = original.winsorize_minmax(raw * np.array([1 if original.DIRECTION[k] else -1 for k in original.ALL_IND]))
    domains = meta["domain"].to_numpy()
    X1 = direction.apply_domain_directions(U, domains)
    lengths = length_diagnostics(meta, X1)
    results, tables, pair_rows, sensitivity_rows, checks, validations = {}, [], [], [], [], {}
    export = meta[["record_index", "id", "domain", "dataset", "set", "source_row"]].copy()
    entropy_available = args.entropy_penalty_csv is not None
    penalty_name = "Q_sequential" if entropy_available else "Q_semantic_only"
    for baseline, X in [("original", X0), ("domain", X1)]:
        direction.log(f"Calculate semantic conflict: {baseline}")
        w, _ = direction.combined_weights(X)
        old = direction.score(X, w)
        if baseline == "original":
            reference_path = Path(BASE) / "results/q1/p1_domain_quality.csv"
            if reference_path.is_file():
                ref = pd.read_csv(reference_path).set_index("domain").loc[direction.DOMAINS]
                actual = direction.aggregate(meta, old, baseline).set_index("domain").loc[direction.DOMAINS]
                error = np.max(np.abs(ref[["Q_weighted", "Q_resolved"]].to_numpy() - actual[["Q_weighted", "Q_resolved"]].to_numpy()))
                assert error < 1e-8
                validations["baseline_max_abs_difference"] = float(error)
        retained, audit, valid, C, S = pair_arrays(X, original.ALL_IND)
        rates = conflict_rate(C, valid)
        domain_rates = conflict_rate(C, valid, domains)
        semantic = semantic_index(C, S, rates, args.alpha)
        entropy = load_entropy(args.entropy_penalty_csv, meta, baseline)
        q0 = old["Q_weighted"].to_numpy()
        calculation_entropy = np.zeros(len(meta)) if entropy is None else entropy
        after_entropy, _ = sequential_correction(q0, calculation_entropy, semantic, args.lambda_entropy, 0)
        if np.any(after_entropy < 0):
            raise ValueError("Entropy deduction already produces negative Q; reduce lambda-entropy")
        limit = coefficient_limit(after_entropy, semantic)
        effective_lambda = args.lambda_semantic if args.lambda_semantic < limit else .9 * limit
        _, corrected = sequential_correction(q0, calculation_entropy, semantic, args.lambda_entropy, effective_lambda)
        assert np.isfinite(corrected).all() and np.all(corrected >= 0) and np.all(corrected <= 1)
        assert np.all(corrected <= after_entropy + 1e-12)
        assert np.allclose(corrected[semantic == 0], after_entropy[semantic == 0])
        checks.append({"baseline": baseline, "lambda_requested": args.lambda_semantic,
                       "lambda_effective": effective_lambda, "lambda_safe_supremum": limit,
                       "alpha": args.alpha, "Q_new_min": float(corrected.min()), "Q_new_max": float(corrected.max())})
        scores = pd.DataFrame({"Q0": q0, "Q_old_trimmed": old["Q_resolved"],
                               "old_conflict": old["conflict"],
                               "entropy_penalty": np.nan if entropy is None else entropy,
                               "entropy_deduction": np.nan if entropy is None else args.lambda_entropy * entropy,
                               "Q_after_entropy": np.nan if entropy is None else after_entropy,
                               "semantic_index": semantic, "semantic_deduction": effective_lambda * semantic,
                               "conflict_pair_count": C.sum(axis=1), "valid_pair_count": valid.sum(axis=1),
                               penalty_name: corrected, "Q_new": corrected})
        results[baseline] = scores
        tables.append(summarize(meta, scores, baseline, penalty_name))
        for key in scores:
            export[f"{baseline}_{key}"] = scores[key]
        for j, (pair_id, a, b) in enumerate(retained):
            for dom in ["ALL"] + direction.DOMAINS:
                mask = np.ones(len(meta), dtype=bool) if dom == "ALL" else domains == dom
                nv = int(valid[mask, j].sum())
                nc = int(C[mask, j].sum())
                pair_rows.append({"baseline": baseline, "scope": "pooled" if dom == "ALL" else "domain",
                                  "domain": dom, "pair_id": pair_id, "field_j": a, "field_k": b,
                                  "n_valid": nv, "n_conflict": nc, "conflict_rate": nc / nv if nv else 0,
                                  "mean_strength": float(S[mask, j].mean()), "pooled_rate_used": rates[j],
                                  "pooled_weight": 1 - (1 - args.alpha) * rates[j],
                                  "mean_weighted_penalty_contribution": float((effective_lambda / len(retained) * S[mask, j] * (1 - (1 - args.alpha) * rates[j])).mean())})
            export[f"{baseline}_pair{pair_id}_C"] = C[:, j].astype(int)
            export[f"{baseline}_pair{pair_id}_S"] = S[:, j]
        contribution_sum = sum(r["mean_weighted_penalty_contribution"] for r in pair_rows if r["baseline"] == baseline and r["domain"] == "ALL")
        assert np.isclose(contribution_sum, scores["semantic_deduction"].mean(), atol=1e-12)
        for scope, scoped_rates in [("pooled", rates), ("domain", domain_rates)]:
            for alpha in [0, .5, 1]:
                si = semantic_index(C, S, scoped_rates, alpha)
                safe_limit = coefficient_limit(after_entropy, si)
                for requested in [0, .05, .1, .2, .5]:
                    effective = requested if requested < safe_limit else .9 * safe_limit
                    new = after_entropy - effective * si
                    sensitivity_rows.append({"baseline": baseline, "rate_scope": scope, "alpha": alpha,
                                             "requested_lambda": requested, "effective_lambda": effective,
                                             "mean_semantic_deduction": float((effective * si).mean()),
                                             "corrected_mean": float(new.mean()), "corrected_min": float(new.min()),
                                             "corrected_max": float(new.max()),
                                             "new_vs_old_spearman": spearmanr(old["Q_resolved"], new).statistic})
    summary, pair_stats, sensitivity = pd.concat(tables, ignore_index=True), pd.DataFrame(pair_rows), pd.DataFrame(sensitivity_rows)
    primary = summary[summary.baseline == "domain"].copy()
    primary["quality_method"] = "domain directions + " + penalty_name
    direction.save_csv(primary, out / "main_domain_quality.csv")
    for table, name in [(summary, "semantic_domain_comparison.csv"), (audit, "semantic_pair_mapping.csv"),
                        (pair_stats, "semantic_pair_statistics.csv"), (sensitivity, "semantic_parameter_sensitivity.csv"),
                        (lengths, "semantic_length_diagnostics.csv")]:
        direction.save_csv(table, out / name)
    manifest = {"sources": provenance, "n_records": len(meta), "indicators": original.ALL_IND,
                "n_indicators": len(original.ALL_IND), "n_document_pairs": len(PAIRS),
                "n_valid_pairs": len(retained), "entropy_available": entropy_available,
                "entropy_status": "external penalties supplied" if entropy_available else "formula absent in attached document; semantic-only results",
                "result_column": penalty_name, "parameters": {"lambda_semantic_requested": args.lambda_semantic,
                "lambda_entropy": args.lambda_entropy, "alpha": args.alpha}, "coefficient_checks": checks,
                "validation": validations, "numerical_contract_checks": "passed",
                "pooled_rate": True, "aggregation": "unchanged old conflict inverse; same eligible records",
                "missing_pair_policy": "skip absent columns; finite observations only in n_jk; fixed retained |P| denominator",
                "duplicates_preserved": True}
    if entropy_available:
        with args.entropy_penalty_csv.open("rb") as stream:
            manifest["entropy_input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (out / "semantic_validation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    direction.log("Export semantic sample results")
    export.to_csv(out / "semantic_sample_scores.csv.gz", index=False, encoding="utf-8-sig", float_format="%.10g", compression="gzip")
    plot_results(out, summary, pair_stats, results, meta, sensitivity)
    write_report(out, summary, pair_stats, audit, manifest, sensitivity, lengths)
    direction.log(summary[["baseline", "domain", "Q0", "Q_old_trimmed", "Q_new", "penalized_rate"]].to_string(index=False))
    direction.log(f"Done: {out}; entropy_available={entropy_available}")


if __name__ == "__main__":
    main()
