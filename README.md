# 算力约束下提升大语言模型能力的资源配置建模

2026 华为杯 F 题。所有 Python 直接放在 `src/`，原始附件在 `real_attachments/`，表、图和分析统一输出到 `results/q1` 至 `results/q4`。

题目：[算力约束下提升大语言模型能力的资源配置建模.docx](./算力约束下提升大语言模型能力的资源配置建模.docx)。字段口径：[数据说明.pdf](./数据说明.pdf)。不修改原始附件。

## 运行

Python 3.10+，依赖见 `requirements.txt`。在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python src/run_all.py
```

运行器按 Q1 → Q2 → Q3 → Q4 执行，失败立即停止。实际退出码、耗时和完整日志保存到 `results/project_checks/`。可用 `python src/run_all.py --questions q2 q3` 只运行选定题目，但前置 Q1 导出必须已存在。运行器不会自动安装依赖。

独立执行的依赖顺序：

```powershell
python src/q1_quality.py
python src/q1_mixture.py
python src/q2_baseline.py
python src/q2_scaling.py
python src/q2_validate.py
python src/q3_optimization.py
python src/q3_comparison_findings.py
python src/q3_validate.py
python src/q4_evolution.py
python src/validate_project.py
```

## 每个 Python 文件的实验内容

| 文件 | 实验职责 |
| --- | --- |
| `run_all.py` | 全流程入口：依赖排序、错误退出、实际运行记录和日志。 |
| `validate_project.py` | 全流程运行后的项目审计：目录/命名、22指标、各题结果、Q4函数保留、实际执行覆盖及README完整性。 |
| `q1_quality.py` | Q1.1–1.2 主入口，串行运行领域方向对照与冲突修正。 |
| `q1_domain_direction.py` | A1–A3 同一批272,505记录、同一22指标上比较七领域方向与原统一方向；保留权重、排名、分数变化分解、抽样/扩展对照与QuRating敏感性。 |
| `q1_semantic_conflict.py` | 领域/原方向的未修正、旧截尾、文档语义修正对照；保留连续熵/语义加权扣减接口、参数敏感性和长度诊断，缺失配对跳过；导出主质量表。 |
| `q1_mixture.py` | Q1.3：M1共用系数、M2为主，固定/调参岭回归为对照，M1每域独立为补充；嵌套CV、跨尺度验证、成对bootstrap及模型导出。 |
| `q2_baseline.py` | B1经典标度律拟合、导出固定E；Cerebras、跨族收敛点及文献迁移诊断，保留原始/去偏移R²。 |
| `q2_scaling.py` | Q2主入口：additive、interaction、quality_power各自无/有λH*的六项拟合、分组验证、边界外推和B8迁移诊断；按OOF RMSE选优给Q3。 |
| `q2_validate.py` | 独立复算预测/指标，检查解析Jacobian、Q边界、分组无泄漏、截距等价、初值稳定性和来源哈希。 |
| `q3_optimization.py` | Q3主入口：纯PSO主求解、SLSQP对照；两个公式、固定/自由配比、三预算、三质量成本、两主窗口、五次重复；窗口条件引入前后及上下文敏感性。主配置用Q2选优公式。 |
| `q3_comparison_findings.py` | 根据真实输出生成公式、配比、算法精度/耗时/调用量的比较及拟合范围核查，可独立重建报告和补充图。 |
| `q3_validate.py` | 独立单位/成本/损失复算，检查预算/单纯形、全因子完整性、参考加密、PSO seed复现、主模型选择和条件对照。 |
| `q4_evolution.py` | 保持Q4原方法：开源筛选、能力前沿分位数回归、规模/技术分解、Loss–能力桥接、12/24月情景预测、C8六维任务聚合；仅改路径和名称。 |
| `quality_features.py` | Q1公共库：22指标、旧方向、列表均值、归一化、熵权/CRITIC、TOPSIS及旧冲突对照函数，无重复旧主入口。 |
| `mixture_regression.py` | M1/M2/岭回归的共用拟合、预测和边际效应。 |
| `scaling_models.py` | Q2公共库：B6–B8读取/去重、Q1 M2导出的p*与H*求解、有界最小二乘及解析Jacobian；无旧比较入口。 |
| `common.py` | 项目路径、附件根目录、17配方域/13损失域的固定顺序。 |
| `plotstyle.py` | 共用科研绘图样式，无独立实验。 |

公共库由实验脚本导入，直接执行不会生成实验结果。`--output-dir`仅修改当前实验输出；全流程默认读取标准目录，不会自动发现自定义路径。

## 第一问：主方法与对照

质量建模保留当前22指标，没有额外加入截图的QuRating子维度。七领域分别设置正、负、中性方向；仍使用全体样本1%/99%分位数归一化，再应用领域方向。保留熵权/CRITIC组合赋权、TOPSIS与原领域聚合口径。方向、权重、冲突聚合对分数变化的贡献分别导出。分数变高不等于预测更准确；没有外部真值时，报告合理性、排名和稳定性变化。

两段式接口为 `Q_after_entropy=Q0-lambda_entropy*entropy_penalty`，再减 `lambda_semantic*semantic_index`。文档没有给出可计算的熵扣分定义，项目也缺少对应逐样本罚分，因此默认真实计算语义修正并标记为 `Q_semantic_only`，不虚称完成熵扣分。熵权赋权不等于熵扣分。提供外部定义及完整罚分后可运行：

```powershell
python src/q1_quality.py --entropy-penalty-csv path/to/entropy.csv --lambda-entropy 1 --lambda-semantic 0.1 --alpha 0.5
```

罚分CSV须含 `record_index,baseline,entropy_penalty`；baseline分别为original与domain，每个基线完整覆盖所有记录、无重复、罚分有限非负。缺失冲突字段跳过，不用聚合QuRating冒充子维度。检查扣分范围，不用截尾掩盖负质量分。主质量结果：`main_domain_quality.csv`。

Q1.3：M1为 `L_j=c_j-Σw_k log(1+p_k)`（共用w），M2为 `L_j=c_j+Σb_jk log(1+p_k/tau)`（共用tau）。用A4/A5的512组配方训练，参数只在训练内选择；A6–A15跨尺度验证，10B/70B为估计数据而非实测。七质量域、17配方域和13损失域用途不同，未虚构七到十七域质量映射。

报告与主导出在 `results/q1/`：`analysis.md`、`semantic_analysis.md`、`mixture_regression_analysis.md`、`main_domain_quality.csv`、`mixture_models.json`。图、压缩逐样本CSV和内部校验同目录保留。

## 第二问：六项比较和第三问衔接

基础项为 `E+A*N^(-a)+B*D^(-b)`，固定E从B1重新估计。三种质量项分别无/有 `lambda*H_star`：

| 形式 | 质量项 |
| --- | --- |
| additive | `C*(1-Q)^g` |
| interaction | `C*(1-Q)^g*D^(-d)` |
| quality_power（新公式） | `eta_Q*(1-Q^kappa)` |

新公式为 `1-Q^kappa`，不是 `(1-Q)^kappa`。H*是Q1 M2预测的13损失域等权平均相对均匀配比的最优差值，**不是熵**。Q2仅通过Q1导出接入A，不重复读取原始A；B的Q仍为原始Q_score，未由p计算。

B6/B7去重后450条唯一数据，按(N,D)整组五折验证，另做最大N、最大D留出及B6→B7新Q验证。B8保留原Q含义，仅作迁移诊断。B6–B8为半合成补充，不能表述为直接实验观测。

`results/q2/model_performance.csv`保存六项成绩，`analysis.md`解释结果。`selected_model.json`按分组OOF RMSE最小选优，保存全量重拟合参数给Q3。当前为additive_mixture：OOF R²≈0.970582、RMSE≈0.058426，每次重跑以实际导出为准。

由于所有B行的H*相同，`E+lambda*H*`等价于可调有效截距。其提升不能证明学到了逐样本配比效应或Q–p耦合。Q3开放p需额外假设“A中的H(p)形状可迁移到B，由lambda缩放”。

## 第三问：优化、隐含条件和比较

`results/q3/main_optimal_configurations.csv`使用Q2验证选优公式的实际PSO解，固定/自由配比均列出；additive与interaction比较保留。默认128粒子、800轮、5次重复，SLSQP默认24初值；同一资源参数化、同seed，不给自由配比PSO喂入已知p*，不做SLSQP后处理。

N、D以十亿计：训练加注意力成本 `10^18*N*D*(6+2e-4*L_ctx)`，质量成本 `10^9*D*[f(Q)-f(0.4)]_+`。保留原三种质量成本、资源边界、预算与配比单纯形条件；由D单调性精确消去D，另作独立参考扫描及加密检查。

原实现已经正确换算单位，因此不把单位错误的解用作合法“引入前”基线。新增 `constraints_before_after.csv`明确以前为理论窗口30000、后为保守主情景2048/4096，在同公式、预算、PSO参数、seed下比较；详见 `constraint_comparison.md`与08号图。窗口缩短改变任务上下文，损失差是情景差异，不能当作同任务性能提升。C7实际另有8192、32768、131072等窗口，4096不是整张表上限；30000是理论注意力/训练等成本点，未在C7直接记录。连续N无法唯一对应架构，数学配置不等于已证明可在某现有架构实现。

在正lambda、Q与p可分离、无配比成本的当前模型下，开放p的理论最优仍为此前p*，数值偏差属于求解误差。PSO优势按实际精度、耗时、波动、调用量分别报告，不预设必胜SLSQP；超过B拟合范围的场景在 `optimization_range_audit.csv`标记。

主要结果：`optimization_analysis.md`、`comparison_findings.md`、`constraint_comparison.md`、`optimization_runs.csv`、`solver_comparison.csv`、`fixed_vs_free_recipe.csv`和 `numerical_validation.json`。

## 清理及复现记录

移除src/code层和p编号活动文件；旧重复质量主入口、旧岭回归主入口、旧七模型比较入口及旧独立优化入口已移除，必要对照整合进新实验/公共库。重构前源码可从 `results/archives/source_before_reorganization.zip`恢复。旧Q2/Q3结果已移至 `results/archives/q2_before_reorganization`、`q3_before_reorganization`，避免混作本次结果。保留原始数据、题目、数据说明和第一问相关输出。

实际运行记录在 `results/project_checks/run_*.json`及对应日志，各题内部数值检查在各自结果目录。全流程运行后执行 `python src/validate_project.py`，项目审计保存为 `results/project_checks/project_validation.json`。具体修改及验证范围见 [重构与验证记录](results/project_checks/reorganization_report.md)。Q4只改名及输出路径，模型和统计计算方法不变，保留原情景预测及其适用限制。
