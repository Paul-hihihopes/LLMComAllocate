# -*- coding: utf-8 -*-
"""Reproduce Q1 and compare domain-specific directions on the same 22 fields.

Run: python src/q1_domain_direction.py
Outputs: <project>/results/q1. The original pipeline and its outputs are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import font_manager

import quality_features as original
from common import A, BASE, RES

DOMAINS = ["arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"]
DOMAIN_LABELS = ["ArXiv", "Book", "C4", "CommonCrawl", "GitHub", "StackExchange", "Wikipedia"]
INDICATORS = original.ALL_IND
# Column order: ArXiv, Book, C4, CommonCrawl, GitHub, StackExchange, Wikipedia.
# 0 means a neutral value of 0.5 AFTER normalization, not a raw numeric zero.
RULES = {
    "rps_doc_num_sentences": "PP00NPP",
    "rps_doc_mean_word_length": "P000N00",
    "rps_lines_ending_with_terminal_punctution_mark": "PP00NPP",
    "rps_lines_uppercase_letter_fraction": "0NNNNNN",
    "rps_lines_numerical_chars_fraction": "P0NN0P0",
    "rps_doc_frac_unique_words": "PPPPPPP",
    "rps_doc_unigram_entropy": "PPPPPPP",
    "rps_doc_frac_no_alph_words": "0NNNP0N",
    "rps_doc_word_count": "PP00NPP",
    "rps_doc_frac_chars_top_2gram": "NNNNNNN",
    "rps_doc_frac_chars_top_3gram": "NNNNNNN",
    "dsir_books": "PPNN0PP",
    "dsir_wiki": "PPNN0PP",
    "dsir_math": "PPNN0PP",
    "ad_en": "NNPPNNN",
    "fluency_en": "PPPPPPP",
    "fineweb_edu": "PPNN0PP",
    # The current field is the unchanged mean of the entire list. The image
    # gives four distinct QuRating directions; only unanimous signs transfer.
    "qurater": "P0000P0",
    "modernbert_professionalism": "P0NNPP0",
    "modernbert_readability": "PPPP0PP",
    "modernbert_reasoning": "P0NNPP0",
    "modernbert_cleanliness": "PPPPPPP",
}
IMAGE_ROWS = {
    "rps_doc_num_sentences": "1", "rps_doc_mean_word_length": "2",
    "rps_lines_ending_with_terminal_punctution_mark": "3",
    "rps_lines_uppercase_letter_fraction": "4", "rps_lines_numerical_chars_fraction": "5",
    "rps_doc_frac_unique_words": "6", "rps_doc_unigram_entropy": "7",
    "rps_doc_frac_no_alph_words": "8", "rps_doc_word_count": "9",
    "rps_doc_frac_chars_top_2gram": "10", "rps_doc_frac_chars_top_3gram": "11",
    "dsir_books": "12", "dsir_wiki": "13", "dsir_math": "14", "ad_en": "15",
    "fluency_en": "16", "fineweb_edu": "17", "qurater": "18-21",
    "modernbert_professionalism": "22", "modernbert_readability": "23",
    "modernbert_reasoning": "24", "modernbert_cleanliness": "25",
}
NAMES = {
    "dsir_books": "书籍语料分布相似度", "dsir_wiki": "Wiki语料分布相似度",
    "dsir_math": "数学语料分布相似度", "rps_doc_word_count": "文档词数",
    "rps_doc_num_sentences": "文档句子数", "rps_doc_unigram_entropy": "词频信息熵",
    "rps_doc_frac_unique_words": "不同单词占比", "rps_doc_frac_no_alph_words": "无字母单词占比",
    "rps_doc_frac_chars_top_2gram": "最频繁二元组覆盖字符比",
    "rps_doc_frac_chars_top_3gram": "最频繁三元组覆盖字符比",
    "rps_lines_uppercase_letter_fraction": "大写字母占比",
    "rps_lines_ending_with_terminal_punctution_mark": "正常标点结尾行比例",
    "rps_lines_numerical_chars_fraction": "数字字符占比",
    "rps_doc_mean_word_length": "单词平均字符长度", "fineweb_edu": "FineWeb教育价值",
    "fluency_en": "流畅度", "modernbert_cleanliness": "ModernBERT整洁性",
    "modernbert_readability": "ModernBERT可读性", "modernbert_reasoning": "ModernBERT推理复杂度",
    "modernbert_professionalism": "ModernBERT专业性", "qurater": "QuRating列表均值",
    "ad_en": "广告检测列表均值",
}
METHOD_LABELS = {"original": "原始统一方向", "domain": "分领域方向",
                 "fixed_weights": "分领域方向（固定原权重）",
                 "qurater_positive": "分领域方向（QuRating保持正向）"}
SOURCES = [
    ("A1", "sample", Path(A) / "slimpajama_quality_signal_sample.jsonl.xz", None),
    ("A2", "arxiv_ext", Path(A) / "slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz", "arxiv"),
    ("A3", "github_ext", Path(A) / "slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz", "github"),
]
assert list(INDICATORS) == original.SCALAR_IND + original.LIST_IND
assert len(INDICATORS) == 22 and set(RULES) == set(INDICATORS)
assert all(len(v) == 7 and set(v) <= {"P", "N", "0"} for v in RULES.values())


def log(message):
    print(message, flush=True)


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def load_data():
    """Stream records, discard text immediately, preserve original row order."""
    frames, provenance = [], []
    for dataset, set_name, path, hint in SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)
        log(f"Loading {dataset}: {path.name}")
        rows = []
        with lzma.open(path, "rt", encoding="utf-8") as stream:
            for source_row, line in enumerate(stream, 1):
                r = json.loads(line)
                row = {"id": r.get("id"), "domain": r.get("_source_domain", hint),
                       "set": set_name, "dataset": dataset, "source_row": source_row,
                       "content_len": len(r["content"]) if "content" in r else np.nan}
                for key in original.SCALAR_IND:
                    value = r.get(key)
                    row[key] = value if isinstance(value, (int, float)) else np.nan
                for key in original.LIST_IND:
                    row[key] = original.compress_list(r.get(key))
                rows.append(row)
        frame = pd.DataFrame(rows)
        frames.append(frame)
        # SHA256 of the input compressed bytes makes the result reproducible.
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance.append({"dataset": dataset, "set": set_name,
                           "path": str(path.relative_to(BASE)), "n": len(frame),
                           "sha256": digest})
        log(f"  {len(frame):,} rows")
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "record_index", np.arange(len(df)))
    unknown = set(df["domain"].dropna()) - set(DOMAINS)
    if unknown or df["domain"].isna().any():
        raise ValueError(f"Unmapped domains: {unknown}")
    raw = df[INDICATORS].to_numpy(dtype=float)
    if np.isinf(raw).any():
        raise ValueError("Infinite indicators found.")
    log(f"  Missing indicator cells: {np.isnan(raw).sum()}; affected rows: {np.isnan(raw).any(axis=1).sum()}")
    return df, provenance


def combined_weights(X):
    """Same entropy and modified CRITIC formulas, computed in one rank pass."""
    ent, entropy = original.entropy_weight(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = np.asarray(spearmanr(X, axis=0).statistic)
    if R.shape != (len(INDICATORS), len(INDICATORS)):
        raise ValueError("Cannot calculate the full indicator correlation matrix.")
    # Equivalent to the original pairwise fallback for constant columns.
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    std = np.nanstd(X, axis=0, ddof=1)
    conflict = np.sum(1 - np.abs(R), axis=1)
    critic = std * conflict
    critic /= critic.sum()
    combo = 0.5 * ent + 0.5 * critic
    assert np.isfinite(combo).all() and np.isclose(combo.sum(), 1)
    return combo, pd.DataFrame({"indicator": INDICATORS, "indicator_cn": [NAMES[k] for k in INDICATORS],
                               "entropy_w": ent, "critic_w": critic, "combo_w": combo,
                               "entropy": entropy, "std": std, "critic_conflict": conflict})


def apply_domain_directions(U, domains, qurater_positive=False):
    X = U.copy()
    for domain_index, domain in enumerate(DOMAINS):
        mask = domains == domain
        for j, key in enumerate(INDICATORS):
            sign = "P" if key == "qurater" and qurater_positive else RULES[key][domain_index]
            if sign == "N":
                X[mask, j] = 1 - U[mask, j]
            elif sign == "0":
                X[mask, j] = 0.5
    return X


def score(X, weights):
    ci, content, form = original.conflict_index(X)
    return pd.DataFrame({"Q_weighted": np.nansum(X * weights, axis=1),
                         "Q_topsis": original.topsis(X, weights),
                         "Q_resolved": original.robust_resolve(X, weights, k=6),
                         "conflict": ci, "Q_content": content, "Q_format": form})


def weighted_mean(values, reliability):
    values, reliability = np.asarray(values), np.asarray(reliability)
    valid = np.isfinite(values) & np.isfinite(reliability)
    return np.average(values[valid], weights=reliability[valid]) if valid.any() else np.nan


def aggregate(meta, scores, method):
    rows = []
    threshold = scores["conflict"].quantile(0.9)
    for domain in DOMAINS:
        mask = meta["domain"] == domain
        s = scores.loc[mask]
        reliability = 1 / (s["conflict"] + 0.05)
        rows.append({"method": method, "domain": domain, "n": len(s),
                     "n_valid_conflict": int(s["conflict"].notna().sum()),
                     **{key: weighted_mean(s[key], reliability) for key in
                        ["Q_weighted", "Q_resolved", "Q_topsis"]},
                     "Q_weighted_plain": s["Q_weighted"].mean(),
                     "Q_std": s["Q_weighted"].std(ddof=1),
                     "conflict_mean": s["conflict"].mean(),
                     "conflict_p90": s["conflict"].quantile(0.9),
                     "high_conflict_fraction": (s["conflict"] >= threshold).mean(),
                     "P90_global": threshold})
    table = pd.DataFrame(rows)
    table["rank"] = table["Q_weighted"].rank(ascending=False, method="min").astype(int)
    return table


def markdown_table(df, columns=None, digits=4):
    if columns:
        df = df[columns]
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        values = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def rank_comparison(meta, old, new):
    rows = []
    for domain in ["ALL"] + DOMAINS:
        mask = np.ones(len(meta), dtype=bool) if domain == "ALL" else (meta["domain"] == domain).to_numpy()
        a, b = old.loc[mask, "Q_weighted"].to_numpy(), new.loc[mask, "Q_weighted"].to_numpy()
        k = max(1, int(np.ceil(len(a) * 0.1)))
        old_top = set(np.argsort(a, kind="stable")[-k:])
        new_top = set(np.argsort(b, kind="stable")[-k:])
        common = len(old_top & new_top)
        rows.append({"domain": domain, "n": len(a), "spearman": spearmanr(a, b).statistic,
                     "kendall_tau": kendalltau(a, b).statistic,
                     "top10_overlap": common / k, "top10_jaccard": common / len(old_top | new_top),
                     "mean_abs_change": np.mean(np.abs(b - a)),
                     "p90_abs_change": np.quantile(np.abs(b - a), 0.9),
                     "old_std": np.std(a, ddof=1), "new_std": np.std(b, ddof=1)})
    return pd.DataFrame(rows)


def make_plots(out, directions, comparison, weights, meta, results, stability, decomposition):
    available = {f.name for f in font_manager.fontManager.ttflist}
    chinese_font = next((f for f in ["Microsoft YaHei", "SimHei", "SimSun"] if f in available), None)
    if chinese_font is None:
        raise RuntimeError("A Chinese font is needed to render the analysis figures.")
    plt.rcParams.update({"font.sans-serif": [chinese_font], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    blue, orange = "#3274A1", "#E1812C"
    def finish(fig, name):
        fig.savefig(out / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    matrix = np.array([[{"N": -1, "0": 0, "P": 1}[v] for v in RULES[k]] for k in INDICATORS])
    ax.imshow(matrix, cmap=ListedColormap(["#BED7EA", "#F7F7F7", "#F3BEC3"]),
              norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3), aspect="auto")
    ax.set_xticks(range(7), DOMAIN_LABELS, rotation=18, ha="right")
    ax.set_yticks(range(22), [NAMES[k] for k in INDICATORS])
    for i, key in enumerate(INDICATORS):
        for j, value in enumerate(RULES[key]):
            ax.text(j, i, value, ha="center", va="center")
    ax.set_title("现有22项指标的分领域方向\nP：正向；N：负向；0：中性0.5；QuRating方向分歧时取中性")
    finish(fig, "01_direction_matrix.png")

    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.bar(x - 0.19, comparison["original_Q"], 0.36, label="原始统一方向", color=blue)
    ax.bar(x + 0.19, comparison["domain_Q"], 0.36, label="分领域方向", color=orange)
    ax.scatter(x + 0.19, comparison["qurater_positive_Q"], marker="D", s=40, color="#3C8D62",
               label="敏感性：QuRating保持正向", zorder=4)
    for i, row in comparison.iterrows():
        ax.text(i - 0.19, row["original_Q"] + 0.012, f'{row["original_Q"]:.3f}', ha="center", fontsize=9)
        ax.text(i + 0.19, max(row["domain_Q"], row["qurater_positive_Q"]) + 0.022,
                f'{row["domain_Q"]:.3f}', ha="center", fontsize=9)
    ax.set_xticks(x, DOMAIN_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("领域质量Q（按冲突倒数聚合）")
    ax.set_title("领域质量分对照：同一批A1+A2+A3记录")
    ax.legend(loc="upper right")
    finish(fig, "02_domain_quality_comparison.png")

    original_w = weights[weights["method"] == "original"].set_index("indicator").loc[INDICATORS]
    domain_w = weights[weights["method"] == "domain"].set_index("indicator").loc[INDICATORS]
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    y = np.arange(22)
    ax.barh(y - 0.18, original_w["combo_w"], 0.35, label="原始统一方向", color=blue)
    ax.barh(y + 0.18, domain_w["combo_w"], 0.35, label="分领域方向", color=orange)
    ax.set_yticks(y, [NAMES[k] for k in INDICATORS])
    ax.invert_yaxis()
    ax.set_xlabel("0.5×熵权 + 0.5×CRITIC权重")
    ax.set_title("赋权公式保持相同，方向变化后重新计算权重")
    ax.legend()
    finish(fig, "03_indicator_weights_comparison.png")

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), layout="constrained")
    for i, domain in enumerate(DOMAINS):
        ax = axes.flat[i]
        indices = np.flatnonzero(meta["domain"].to_numpy() == domain)
        indices = rng.choice(indices, min(2500, len(indices)), replace=False)
        ax.scatter(results["original"].loc[indices, "Q_weighted"], results["domain"].loc[indices, "Q_weighted"],
                   s=5, alpha=0.25, color=blue, rasterized=True)
        ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls="--")
        rho = stability.set_index("domain").loc[domain, "spearman"]
        ax.set(title=f"{DOMAIN_LABELS[i]}（ρ={rho:.3f}）", xlim=(0, 1), ylim=(0, 1),
               xlabel="原始样本Q", ylabel="分领域样本Q")
    axes.flat[-1].axis("off")
    fig.suptitle("样本质量与域内排序变化（每域最多绘制2500条，统计量使用全量）")
    finish(fig, "04_sample_score_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(x - 0.19, comparison["original_conflict"], 0.36, label="原始统一方向", color=blue)
    ax.bar(x + 0.19, comparison["domain_conflict"], 0.36, label="分领域方向", color=orange)
    ax.set_xticks(x, DOMAIN_LABELS, rotation=15, ha="right")
    ax.set_ylabel("平均冲突指数 |Q内容 − Q格式|")
    ax.set_title("冲突公式保持相同，输入方向变化后冲突值重新计算")
    ax.legend()
    finish(fig, "05_conflict_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    pos, neg = np.zeros(7), np.zeros(7)
    for key, label, color in [("direction_component", "方向与中性处理", blue),
                              ("indicator_weight_component", "指标重新赋权", orange),
                              ("conflict_aggregation_component", "冲突聚合权重变化", "#3C8D62")]:
        values = decomposition[key].to_numpy()
        bottoms = np.where(values >= 0, pos, neg)
        ax.bar(x, values, bottom=bottoms, label=label, color=color)
        pos += np.maximum(values, 0)
        neg += np.minimum(values, 0)
    ax.scatter(x, decomposition["total_delta"], color="black", s=28, label="总变化", zorder=5)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x, DOMAIN_LABELS, rotation=15, ha="right")
    ax.set_ylabel("领域质量分变化 ΔQ")
    ax.set_title("领域质量变化的逐步分解（分解顺序：方向、指标权重、冲突聚合）")
    ax.legend()
    finish(fig, "06_score_change_decomposition.png")

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    s = stability[stability["domain"] != "ALL"].set_index("domain").loc[DOMAINS]
    ax.bar(x - 0.19, s["spearman"], 0.36, label="域内Spearman相关", color=blue)
    ax.bar(x + 0.19, s["top10_overlap"], 0.36, label="前10%样本保留比例", color=orange)
    ax.set_xticks(x, DOMAIN_LABELS, rotation=15, ha="right")
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("排序一致性")
    ax.set_title("新旧评分是否选择相同的高质量样本")
    ax.legend()
    finish(fig, "07_ranking_stability.png")


def write_report(out, comparison, stability, decomposition, weights, provenance, validation, counts):
    old_rank = comparison.sort_values("original_rank")["domain"].tolist()
    new_rank = comparison.sort_values("domain_rank")["domain"].tolist()
    sensitivity = np.max(np.abs(comparison["qurater_positive_Q"] - comparison["domain_Q"]))
    total = sum(p["n"] for p in provenance)
    github_share = counts.loc[counts["domain"] == "github", "n"].sum() / total
    w0 = weights[weights["method"] == "original"].set_index("indicator")["combo_w"]
    w1 = weights[weights["method"] == "domain"].set_index("indicator")["combo_w"]
    largest_weights = (w1 - w0).abs().sort_values(ascending=False).head(5)
    c = comparison.set_index('domain')
    s = stability.set_index('domain')
    d = decomposition.set_index('domain')
    text = f"""# 第一问：分领域指标方向与原方法对照

## 结论与实验范围

可以按领域定义方向，但所得Q应解释为“领域适用性与质量的综合评分”，尚不能宣称它比原方法更能预测训练收益。本次只研究质量评价，不重新训练语言模型、不改动配比模型及问题二、三的参数。

使用A1、A2、A3共{total:,}条记录，完整保留原有22项指标字段、列表均值压缩、样本范围和重复记录。GitHub占合并记录的{github_share:.2%}，Book仅{int(comparison.loc[comparison['domain'] == 'book', 'n'].iloc[0])}条。因此合并赋权明显受样本构成影响，Book结论尤其需要更多数据检验。

原始领域排序：{' > '.join(old_rank)}。

分领域方向排序：{' > '.join(new_rank)}。

### 领域质量对照

{markdown_table(comparison, ['domain', 'n', 'original_Q', 'domain_Q', 'delta_Q', 'original_rank', 'domain_rank', 'qurater_positive_Q'])}

Q是样本加权分再按冲突倒数聚合的领域分，差值为新方法减原方法。敏感性列仅将聚合后的qurater保持全领域正向。

### 结果如何解释

- GitHub的Q增加{c.loc['github', 'delta_Q']:.4f}，排名从{int(c.loc['github', 'original_rank'])}升到{int(c.loc['github', 'domain_rank'])}。按本次分解，方向与中性化贡献{d.loc['github', 'direction_component']:.4f}，重新赋权贡献{d.loc['github', 'indicator_weight_component']:.4f}。原方法偏重词数和句数，代码样本的原归一化值较低，翻为负向后获得高分；这反映了奖励短文本的新规则，不能直接解释为代码本身更优。
- 词数权重从{w0['rps_doc_word_count']:.4f}降至{w1['rps_doc_word_count']:.4f}，句数从{w0['rps_doc_num_sentences']:.4f}降至{w1['rps_doc_num_sentences']:.4f}。三个DSIR权重合计从{sum(w0[k] for k in ['dsir_books','dsir_wiki','dsir_math']):.4f}升至{sum(w1[k] for k in ['dsir_books','dsir_wiki','dsir_math']):.4f}。中性化并不等于把对应指标的全局权重设为0。
- Wikipedia与StackExchange的直接方向贡献仅{d.loc['wikipedia', 'direction_component']:.4f}与{d.loc['stackexchange', 'direction_component']:.4f}，重新赋权贡献分别为{d.loc['wikipedia', 'indicator_weight_component']:.4f}与{d.loc['stackexchange', 'indicator_weight_component']:.4f}。即使某领域很少改变方向，其他领域改变分布后产生的全局新权重也会影响它。
- Book的直接方向贡献为{d.loc['book', 'direction_component']:.4f}，但重新赋权贡献为{d.loc['book', 'indicator_weight_component']:.4f}，最终Q下降。因此不能仅用某个字段翻转来解释总评分。
- C4的域内Spearman为{s.loc['c4', 'spearman']:.4f}，前10%重叠为{s.loc['c4', 'top10_overlap']:.1%}，新旧排序几乎不相关。GitHub前10%重叠为{s.loc['github', 'top10_overlap']:.1%}；Wikipedia与StackExchange分别为{s.loc['wikipedia', 'top10_overlap']:.1%}和{s.loc['stackexchange', 'top10_overlap']:.1%}，相对更稳定。
- 七域中六域平均冲突降低，但CommonCrawl从{c.loc['commoncrawl', 'original_conflict']:.4f}升至{c.loc['commoncrawl', 'domain_conflict']:.4f}。保持冲突公式不等于保证冲突改善；其他领域的降低也可能来自中性项拉近两组均值。
- QuRating全P敏感性中，GitHub仍第1、Book仍第2、Wikipedia仍第3，但ArXiv由主方法的第{int(c.loc['arxiv', 'domain_rank'])}变为第{int(c.loc['arxiv', 'qurater_positive_rank'])}。聚合字段映射会影响接近的领域排名，不能忽略。

## 22项指标与图片25行的映射

21个现有字段直接对应图片中的21行。图片18–21行的四种QuRating维度，在现有流程中只有qurater一列（整个列表取均值），没有四个独立的质量特征。本实验保持这种压缩方式，不拆分、不替换、不添加字段，也不声称列表分量顺序已得到确认。

对聚合后的qurater使用保守的“方向一致”规则：四行在某领域全为P时用P，全为N时用N，否则用0。故ArXiv、StackExchange为P，其余五域为0。它是映射假设，不是图片对聚合字段给出的直接规定。另计算qurater在七域都取P的敏感性方法，两种映射的最大领域Q差为{sensitivity:.4f}。

ad_en和fluency_en同样保持原代码的列表均值，未改为正类logit、概率或logit差。因此这里只转移方向规则，不把当前压缩字段等同于图片中的独立二分类logit。

完整映射见direction_mapping.csv；图片规则图见01_direction_matrix.png。

## 计算口径：仅按领域改变方向

1. 读取A1/A2/A3，14项标量原样提取，8项列表对有限值取均值。
2. 每个指标使用合并样本的原始1%和99%分位数，裁剪并归一化为u，保持原来的跨领域归一化范围。
3. 对领域d的指标j，P取u，N取1-u，0取0.5。0表示中性、不提供该领域的样本区分信号，不是缺失值，也不是把原始值乘0后当成最低质量。
4. 在整个有向矩阵上重新计算熵权与修改版CRITIC（Spearman绝对相关），各占50%。权重公式相同，权重数值会变化。
5. 用加权和与TOPSIS评分。TOPSIS的理想点仍在合并矩阵上计算。
6. 沿用原内容15项/格式7项分组、组内均值、I=|Q_C-Q_F|、每条样本去最低3项和最高3项、按剩余权重重归一化。
7. 领域聚合沿用1/(I+0.05)。冲突算法没有变，但输入改变，I和聚合权重重新计算。没有固定旧冲突值。

本实验未做“各领域分别Min–Max、分别赋权、分别选TOPSIS理想点”。那会进一步改变流程，并使不同领域的Q主要表示各自领域内的相对位置，不能直接作统一质量排名。

### 缺失值沿用原流程

实际有19条记录存在缺失指标：ModernBERT推理复杂度13个缺失，专业性6个缺失。未插补或删除记录。原流程用nanpercentile/nanstd/nansum跳过缺失，Spearman对含缺失列得到的非有限相关回退为0；TOPSIS的普通max/min使含缺失指标整列距离项被nansum忽略；内容组普通均值使这些记录的原冲突为缺失，领域聚合跳过非有限权重。本次保留这些行为以准确复现基线，它们不等于一套理想的缺失值处理方法。

分领域中性项取0.5（包括该指标原来缺失的记录），因此Wikipedia的15条记录重新获得有效冲突值，CommonCrawl的4条记录仍无有效冲突。有效聚合条数见domain_scores.csv的n_valid_conflict。分解的最后一项包含冲突权重和有效聚合样本集合变化；本次没有另行调整该算法。

## 四种对照方法

| method | 含义 | 目的 |
| --- | --- | --- |
| original | 原统一方向，全量重新计算 | 基线 |
| domain | 图片领域方向＋保守QuRating映射，重新赋权 | 主实验 |
| fixed_weights | 使用主实验方向矩阵，固定原指标权重 | 判断重新赋权的影响；冲突仍按新矩阵计算 |
| qurater_positive | 图片其他领域方向不变，QuRating全P，重新赋权 | 聚合字段映射的敏感性 |

以上方法均采用相同22列、相同数据和相同冲突公式。fixed_weights只用于诊断，不是主方法。

## 样本排序变化

{markdown_table(stability, ['domain', 'n', 'spearman', 'kendall_tau', 'top10_overlap', 'mean_abs_change', 'old_std', 'new_std'])}

Spearman和Kendall衡量新旧排序的一致性。top10_overlap是旧前10%与新前10%共同样本数除以前10%样本数，越低意味着筛选结果变化越大。ALL使用所有记录，受领域占比及域间分数变化影响，应优先看域内结果。并列分数的前10%边界使用稳定行号处理。

排序相似或分数上升都不是质量评价准确性证据。当前没有人工质量真值，也没有新筛选语料的训练收益实测。

## 分数变化的来源

{markdown_table(decomposition)}

分解采用固定顺序：先改变方向（暂用旧指标权重、旧冲突聚合权重及其有效样本集合），再换指标权重（暂用旧冲突聚合权重），最后换新冲突聚合权重及其有效样本集合。三项严格加总为总变化，但不是因果贡献，也不是与分解顺序无关的唯一分解。

权重变化最大的5项为：{', '.join(NAMES[k] + '（|Δw|=' + format(v, '.4f') + '）' for k, v in largest_weights.items())}。详见indicator_weights.csv。

## 优势、局限与建议

### 分领域方向的优势

- 对GitHub的标点、词长、符号比例等不再强行沿用自然语言偏好，能表达领域结构差异。
- 中性指标不再用原始数值区分该领域样本，减少明显不适用信号的直接影响。
- 每个领域的规则可追溯，便于检查筛选标准为何改变。

### 分领域方向的局限

- 方向来自人为设定，并非通过训练损失或人工质量标签识别。更符合领域直觉不等于评价更准确。
- C4/CommonCrawl的广告字段设为正向、教育和推理字段设为负向，是图片本身的设定。若目标是教育性、推理能力或去广告的通用预训练质量，这些方向需要实证支持；不能仅因为网页域就认为广告越多越好。
- GitHub词数/句数设为负向等于奖励更短文本，不是“适度最好”，可能误伤完整函数和较长优质代码。若认为存在最佳区间，下一步应单独研究适度型转换。
- 原始列表均值本身未验证语义。例如二分类logit取均值与正类概率不是同一个量，本实验按要求沿用，不能靠正负向切换解决这一问题。
- 跨领域方向切换会把领域身份写入质量分及合并权重。不同领域的高分可能代表不同偏好，跨领域Q不再具有完全相同的“越优质”含义。
- 中性0.5使该指标在对应领域内失去区分能力，并对加权和形成常数基线。总分的离散度还受重新赋权影响，不能保证所有领域总分方差都降低。中性项数和权重不同会影响域间均值；保留冲突分组和截尾方法时，中性项也会影响分组均值与被保留指标。
- 当前样本极不均衡，GitHub扩展集主导合并赋权。各领域样本更多并不代表领域质量更高，但会影响权重估计。
- 保留原冲突公式并不保证其原解释仍成立：新格式组混合了领域适用性方向，尤其广告正向之后，Q_F更适合称为“格式及领域适用性分”。冲突变小也可能由中性化造成，不能直接认定更可靠。

### 原始统一方向的优势与局限

统一方向有利于规则一致、跨域解释简单、复算稳定，但自然语言偏好可能误判代码和数学文本。它的领域Q同样是人为定义指标和数据驱动权重的结果，尚无独立训练收益验证。

### 推荐用途

本次分领域结果适合先用于各领域内部的候选文本排序，原始结果保留为对照。暂不因某领域Q上升就提高它在配方p中的占比。要判断哪种方法更优，建议对每域进行盲评，或控制N/D/p和预算后，用新旧规则选出的等量语料训练、比较目标验证损失。若需要跨域配置，宜增加跨域校准和任务收益验证。

## 验证与复算

22字段集合严格等于原代码；缺失行为按原流程保留，P/N/0映射、权重和为1、Q范围、分解加总均已检查。原方法重算与现存p1_domain_quality.csv对照：{json.dumps(validation, ensure_ascii=False)}。

SHA256、样本来源和指标分位数保存在run_manifest.json、data_inventory.csv、normalization_ranges.csv。A1与扩展集可能有相同id，本次按“其他步骤不变”保留，重复统计只提示、不去重。不存在重新训练模型、置信区间或优劣显著性检验。

复算命令：`python src/q1_domain_direction.py`。

## 输出索引

- domain_comparison.csv：七领域的新旧Q、名次、冲突及两个诊断结果。
- domain_scores.csv：四种方法的加权Q、消解Q、TOPSIS、冲突等完整领域统计。
- direction_mapping.csv：现有22列与图片规则映射。
- indicator_weights.csv：三种重新赋权方法的熵权、CRITIC及组合权重。
- ranking_stability.csv：全量和各域样本排序、前10%重叠和分数离散度。
- score_change_decomposition.csv：方向、赋权、冲突聚合三部分变化。
- indicator_direction_contributions.csv：各指标在旧冲突权重下的直接方向贡献。
- conflict_by_set.csv：抽样与两个扩展集的冲突及质量统计。
- sample_vs_full.csv：使用同一全量校准下，抽样记录和全部记录的领域Q。
- sample_scores.csv.gz：全量记录的新旧分数和冲突（gzip压缩CSV）。
- normalized_domain_indicators.csv.gz：主实验的全量22列归一化方向矩阵及行号。
- data_inventory.csv、normalization_ranges.csv、run_manifest.json：数据和复算信息。
- 01–07 PNG：方向、领域分数、权重、样本分数、冲突、变化分解、排序稳定性。
"""
    (out / "analysis.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta, provenance = load_data()
    log(f"Total {len(meta):,} records; preserve original missing-value behavior")
    counts = meta.groupby(["dataset", "set", "domain"], sort=False).agg(
        n=("record_index", "size"), unique_nonmissing_ids=("id", "nunique")).reset_index()
    counts["duplicate_id_rows_within_set"] = counts["n"] - counts["unique_nonmissing_ids"]
    counts["share_of_all"] = counts["n"] / len(meta)
    save_csv(counts, out / "data_inventory.csv")
    directions = pd.DataFrame([{"indicator": k, "indicator_cn": NAMES[k],
                                "original_direction": "P" if original.DIRECTION[k] else "N",
                                "image_rows": IMAGE_ROWS[k],
                                **dict(zip(DOMAINS, RULES[k])),
                                "mapping_note": "四维方向一致才转移，否则中性；另做全P敏感性" if k == "qurater" else
                                "保持原列表均值，非正类logit" if k in ["ad_en", "fluency_en"] else "直接对应"}
                               for k in INDICATORS])
    save_csv(directions, out / "direction_mapping.csv")
    raw = meta[INDICATORS].to_numpy(dtype=float)
    q01, q99 = np.nanquantile(raw, [0.01, 0.99], axis=0)
    save_csv(pd.DataFrame({"indicator": INDICATORS, "raw_q01": q01, "raw_q99": q99,
                           "missing_n": (~np.isfinite(raw)).sum(axis=0)}), out / "normalization_ranges.csv")
    # Applying N as 1-u is exactly equivalent to negative-before-MinMax when
    # the same raw pooled quantile range is used; do not pool signed raw
    # values of differing domains, which would mix incompatible scales.
    U = original.winsorize_minmax(raw)
    X0 = original.winsorize_minmax(raw * np.array([1 if original.DIRECTION[k] else -1 for k in INDICATORS]))
    expected0 = np.column_stack([U[:, j] if original.DIRECTION[k] else 1 - U[:, j]
                                 for j, k in enumerate(INDICATORS)])
    assert np.allclose(X0, expected0, atol=1e-12, equal_nan=True)
    domains = meta["domain"].to_numpy()
    X1 = apply_domain_directions(U, domains)
    Xs = apply_domain_directions(U, domains, qurater_positive=True)
    weights, matrices, result = {}, {"original": X0, "domain": X1, "qurater_positive": Xs}, {}
    weight_tables = []
    for method, X in matrices.items():
        log(f"Calculate weights and scores: {method}")
        weights[method], wt = combined_weights(X)
        wt.insert(0, "method", method)
        weight_tables.append(wt)
        result[method] = score(X, weights[method])
        assert np.isfinite(result[method][["Q_weighted", "Q_resolved", "Q_topsis", "Q_format"]].to_numpy()).all()
        assert result[method][["Q_weighted", "Q_resolved", "Q_topsis"]].to_numpy().min() >= -1e-12
        assert result[method][["Q_weighted", "Q_resolved", "Q_topsis"]].to_numpy().max() <= 1 + 1e-12
    result["fixed_weights"] = score(X1, weights["original"])
    weight_df = pd.concat(weight_tables, ignore_index=True)
    save_csv(weight_df, out / "indicator_weights.csv")
    tables = {method: aggregate(meta, scores, method) for method, scores in result.items()}
    domain_scores = pd.concat(tables.values(), ignore_index=True)
    save_csv(domain_scores, out / "domain_scores.csv")
    comparison = pd.DataFrame({"domain": DOMAINS})
    for method, table in tables.items():
        lookup = table.set_index("domain")
        comparison[f"{method}_Q"] = comparison["domain"].map(lookup["Q_weighted"])
        comparison[f"{method}_rank"] = comparison["domain"].map(lookup["rank"])
        comparison[f"{method}_conflict"] = comparison["domain"].map(lookup["conflict_mean"])
    comparison.insert(1, "n", comparison["domain"].map(tables["original"].set_index("domain")["n"]))
    comparison["delta_Q"] = comparison["domain_Q"] - comparison["original_Q"]
    comparison["delta_conflict"] = comparison["domain_conflict"] - comparison["original_conflict"]
    save_csv(comparison, out / "domain_comparison.csv")

    log("Calculate ranking and exact score-change decomposition")
    stability = rank_comparison(meta, result["original"], result["domain"])
    save_csv(stability, out / "ranking_stability.csv")
    decomposition, contributions = [], []
    for domain in DOMAINS:
        mask = domains == domain
        reliability0 = 1 / (result["original"].loc[mask, "conflict"].to_numpy() + 0.05)
        oldq = weighted_mean(result["original"].loc[mask, "Q_weighted"], reliability0)
        directionq = weighted_mean(result["fixed_weights"].loc[mask, "Q_weighted"], reliability0)
        newweightq = weighted_mean(result["domain"].loc[mask, "Q_weighted"], reliability0)
        newq = tables["domain"].set_index("domain").loc[domain, "Q_weighted"]
        row = {"domain": domain, "direction_component": directionq - oldq,
               "indicator_weight_component": newweightq - directionq,
               "conflict_aggregation_component": newq - newweightq, "total_delta": newq - oldq}
        assert np.isclose(sum(row[k] for k in ["direction_component", "indicator_weight_component", "conflict_aggregation_component"]), row["total_delta"], atol=1e-12)
        decomposition.append(row)
        valid = np.isfinite(reliability0)
        changes = np.average((X1[mask] - X0[mask])[valid], axis=0, weights=reliability0[valid]) * weights["original"]
        assert np.isclose(changes.sum(), directionq - oldq, atol=1e-12)
        for k, value in zip(INDICATORS, changes):
            contributions.append({"domain": domain, "indicator": k, "indicator_cn": NAMES[k],
                                  "direction_contribution": value})
    decomposition = pd.DataFrame(decomposition)
    save_csv(decomposition, out / "score_change_decomposition.csv")
    save_csv(pd.DataFrame(contributions), out / "indicator_direction_contributions.csv")

    set_rows, sample_rows = [], []
    for method, scores in result.items():
        threshold = scores["conflict"].quantile(0.9)
        for name in meta["set"].unique():
            sub = scores.loc[meta["set"] == name]
            set_rows.append({"method": method, "set": name, "n": len(sub),
                             "Q_mean": sub["Q_weighted"].mean(), "Q_resolved_mean": sub["Q_resolved"].mean(),
                             "conflict_mean": sub["conflict"].mean(), "conflict_p90": sub["conflict"].quantile(0.9),
                             "high_conflict_fraction": (sub["conflict"] >= threshold).mean(), "P90_global": threshold})
        for domain in DOMAINS:
            mask = (meta["domain"] == domain) & (meta["set"] == "sample")
            sub = scores.loc[mask]
            q_sample = weighted_mean(sub["Q_weighted"], 1 / (sub["conflict"] + 0.05))
            q_full = tables[method].set_index("domain").loc[domain, "Q_weighted"]
            sample_rows.append({"method": method, "domain": domain, "sample_n": len(sub),
                                "sample_only_Q": q_sample, "all_records_Q": q_full, "delta_full_minus_sample": q_full - q_sample})
    save_csv(pd.DataFrame(set_rows), out / "conflict_by_set.csv")
    save_csv(pd.DataFrame(sample_rows), out / "sample_vs_full.csv")

    validation = {"baseline_reference_available": False}
    old_file = Path(BASE) / "results/q1/historical_domain_quality.csv"
    if old_file.is_file():
        reference = pd.read_csv(old_file).set_index("domain").loc[DOMAINS]
        actual = tables["original"].set_index("domain").loc[DOMAINS]
        fields = ["Q_weighted", "Q_resolved", "Q_topsis", "Q_weighted_plain", "conflict_mean"]
        max_error = np.max(np.abs(reference[fields].to_numpy() - actual[fields].to_numpy()))
        validation.update({"baseline_reference_available": True, "max_abs_difference": float(max_error),
                           "same_counts": bool((reference["n"] == actual["n"]).all())})
        if max_error > 1e-8 or not validation["same_counts"]:
            raise ValueError(f"Baseline does not reproduce the existing outputs: {validation}")
    log(f"Baseline validation: {validation}")
    # Check efficient CRITIC against the original pairwise implementation on
    # a deterministic subset, then compare full baseline weights to saved ones.
    check = X0[np.linspace(0, len(X0) - 1, min(3000, len(X0)), dtype=int)]
    _, check_w = combined_weights(check)
    pairwise, _, _ = original.critic_weight(check)
    assert np.allclose(pairwise, check_w["critic_w"], atol=1e-12)
    wfile = Path(RES) / "quality_features_weights.csv"
    if wfile.is_file():
        oldw = pd.read_csv(wfile).set_index("indicator").loc[INDICATORS, "combo_w"].to_numpy()
        validation["max_weight_abs_difference"] = float(np.max(np.abs(oldw - weights["original"])))
        assert np.allclose(oldw, weights["original"], atol=1e-10)
    manifest = {"sources": provenance, "n_records": len(meta), "indicators": INDICATORS,
                "methods": METHOD_LABELS, "normalization": "pooled raw 1%-99% range, then domain P=u/N=1-u/0=0.5",
                "qurater_mapping": "unanimous signs from image rows 18-21; otherwise neutral",
                "duplicates_preserved": True, "nonmissing_unique_ids_all": int(meta["id"].nunique()),
                "all_id_missing_n": int(meta["id"].isna().sum()),
                "extra_repeated_id_rows_all": int(meta["id"].notna().sum() - meta["id"].nunique()),
                "missing_by_indicator": dict(zip(INDICATORS, np.isnan(raw).sum(axis=0).tolist())),
                "missing_affected_rows": int(np.isnan(raw).any(axis=1).sum()),
                "missing_policy": "unchanged original nan handling; neutral directions set to 0.5 including originally missing entries",
                "validation": validation}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("Export full sample results (compressed CSV)")
    export = meta[["record_index", "id", "domain", "dataset", "set", "source_row"]].copy()
    for method, scores in result.items():
        for key in ["Q_weighted", "Q_resolved", "Q_topsis", "conflict"]:
            export[f"{method}_{key}"] = scores[key]
    export.to_csv(out / "sample_scores.csv.gz", index=False, encoding="utf-8-sig", float_format="%.10g", compression="gzip")
    norm_export = pd.DataFrame(X1, columns=INDICATORS)
    norm_export.insert(0, "record_index", meta["record_index"])
    norm_export.to_csv(out / "normalized_domain_indicators.csv.gz", index=False, encoding="utf-8-sig", float_format="%.10g", compression="gzip")
    log("Render comparison figures and analysis")
    make_plots(out, directions, comparison, weight_df, meta, result, stability, decomposition)
    write_report(out, comparison, stability, decomposition, weight_df, provenance, validation, counts)
    # Typed matrices for the optional readable Excel summary builder.
    compact = {"comparison": json.loads(comparison.to_json(orient="split", double_precision=15)),
               "directions": json.loads(directions.to_json(orient="split", double_precision=15)),
               "stability": json.loads(stability.to_json(orient="split", double_precision=15)),
               "weights": json.loads(weight_df.to_json(orient="split", double_precision=15)),
               "decomposition": json.loads(decomposition.to_json(orient="split", double_precision=15))}
    (out / "summary_data.json").write_text(json.dumps(compact, ensure_ascii=False), encoding="utf-8")
    log(comparison[["domain", "original_Q", "domain_Q", "delta_Q", "original_rank", "domain_rank"]].to_string(index=False))
    log(f"Done: {out}")


if __name__ == "__main__":
    main()
