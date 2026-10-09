# -*- coding: utf-8 -*-
"""Q1共用22指标、归一化、客观赋权、TOPSIS及旧冲突修正对照。

领域方向与数据读取由q1_domain_direction负责，语义修正由
q1_semantic_conflict负责。本模块不读附件、不绘图、不导出实验结果。
"""

import numpy as np
from scipy.stats import spearmanr

# ---------------- 常量 ----------------
SCALAR_IND = ["dsir_books", "dsir_wiki", "dsir_math",
              "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
              "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
              "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
              "rps_lines_uppercase_letter_fraction", "rps_lines_ending_with_terminal_punctution_mark",
              "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"]
LIST_IND = ["fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
            "modernbert_reasoning", "modernbert_professionalism", "qurater", "ad_en"]
ALL_IND = SCALAR_IND + LIST_IND

# 方向: True=越高越好, False=越低越好(负向)
DIRECTION = {
    "fineweb_edu": True, "fluency_en": True,
    "modernbert_cleanliness": True, "modernbert_readability": True,
    "modernbert_reasoning": True, "modernbert_professionalism": True,
    "dsir_books": True, "dsir_wiki": True, "dsir_math": True,
    "qurater": True,
    "ad_en": False,                       # 广告含量: 越低越好
    "rps_doc_word_count": True, "rps_doc_num_sentences": True,
    "rps_doc_unigram_entropy": True, "rps_doc_frac_unique_words": True,
    "rps_doc_frac_no_alph_words": False,  # 无字母词占比: 越低越好
    "rps_doc_frac_chars_top_2gram": False, "rps_doc_frac_chars_top_3gram": False,  # 重复性: 越低越好
    "rps_lines_uppercase_letter_fraction": False,  # 全大写噪音: 越低越好
    "rps_lines_ending_with_terminal_punctution_mark": True,
    "rps_lines_numerical_chars_fraction": False,   # 数字占比过高=表格/乱码
    "rps_doc_mean_word_length": True,     # 词长适中偏长为佳(信息密度)
}
assert set(DIRECTION) == set(ALL_IND)

def compress_list(v):
    """列表型指标压缩为标量: 均值; 空列表->nan"""
    if isinstance(v, list) and len(v) > 0:
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            return float(np.mean(arr))
    return np.nan

def winsorize_minmax(X, lo=0.01, hi=0.99):
    """分位数裁剪 + Min-Max 归一化, 返回 0-1 矩阵(越高越好)"""
    Xn = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        q_lo, q_hi = np.nanpercentile(col, [lo * 100, hi * 100])
        c = np.clip(col, q_lo, q_hi)
        span = q_hi - q_lo
        Xn[:, j] = (c - q_lo) / span if span > 1e-12 else 0.5
    return Xn

def entropy_weight(X):
    """熵权法: 输入已归一化(0-1, 非负)"""
    n, m = X.shape
    P = X / np.nansum(X, axis=0, keepdims=True)
    P = np.where(P > 0, P, 1e-12)
    e = -np.nansum(P * np.log(P), axis=0) / np.log(n)
    d = 1 - e
    w = d / d.sum()
    return w, e

def critic_weight(X):
    """CRITIC 赋权: 对比强度(标准差) x 冲突性(1-|r|)"""
    n, m = X.shape
    std = np.nanstd(X, axis=0, ddof=1)
    # 用 Spearman 秩相关减小异常值影响
    R = np.ones((m, m))
    cols = [X[:, j] for j in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            with np.errstate(invalid="ignore"):
                rho = spearmanr(cols[i], cols[j]).statistic
            R[i, j] = R[j, i] = rho if np.isfinite(rho) else 0.0
    conflict = np.sum(1 - np.abs(R), axis=1)
    C = std * conflict
    w = C / C.sum()
    return w, std, conflict

def topsis(X, w):
    """TOPSIS 贴近度(越高越好), 输入归一化矩阵与权重"""
    V = X * w
    ideal = np.max(V, axis=0)
    nadir = np.min(V, axis=0)
    d_pos = np.sqrt(np.nansum((V - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.nansum((V - nadir) ** 2, axis=1))
    return d_neg / (d_pos + d_neg)

# 质量指标分族: "内容质量族" C 与 "格式洁净族" F (方向统一后均越高越好)
FAMILY_C = ["fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
            "modernbert_reasoning", "modernbert_professionalism", "dsir_books", "dsir_wiki",
            "dsir_math", "qurater", "rps_doc_word_count", "rps_doc_num_sentences",
            "rps_doc_unigram_entropy", "rps_doc_frac_unique_words", "rps_doc_mean_word_length"]
FAMILY_F = ["ad_en", "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
            "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
            "rps_lines_numerical_chars_fraction", "rps_lines_ending_with_terminal_punctution_mark"]
assert set(FAMILY_C + FAMILY_F) == set(ALL_IND)

def family_scores(X):
    """两族分数: Q_C=内容质量族均值, Q_F=格式洁净族均值 (方向统一后均越高越好, 0-1)"""
    jc = [ALL_IND.index(k) for k in FAMILY_C]
    jf = [ALL_IND.index(k) for k in FAMILY_F]
    return X[:, jc].mean(axis=1), X[:, jf].mean(axis=1)


def conflict_index(X):
    """冲突指数 (样本级): |Q_C - Q_F| ∈ [0,1]
    Q_C=内容质量族均值, Q_F=格式洁净族均值.
    冲突高 = "内容好而格式脏" (如高教育价值+高广告) 或 "格式净而内容贫"."""
    qc, qf = family_scores(X)
    return np.abs(qc - qf), qc, qf


def robust_resolve(X, w, k=6):
    """冲突消解: 对称 trimmed 聚合 (每个样本剔除 k/2 个最低与 k/2 个最高分指标后加权)
    消解规则: 冲突样本中, 极端高/低分指标视为该样本"内部不一致"的来源,
    对称剔除后按剩余指标重归一化权重合成, 得到消解后的打分 Q*."""
    n, m = X.shape
    k = min(k, m - 2)
    k2 = k // 2
    Qr = np.empty(n)
    for i in range(n):
        x = X[i]
        order = np.argsort(x)
        keep = order[k2: m - k2]        # 对称去极值
        ww = w[keep]
        Qr[i] = np.nansum(x[keep] * ww) / ww.sum()
    return Qr
