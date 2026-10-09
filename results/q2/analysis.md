# 第二问：新质量幂函数的同数据比较

结果：新公式没有在本次验证中表现出相对additive的一致优势。
不加H*时，新公式五折验证RMSE比additive高0.072%，
比interaction高0.255%。
加H*后，比interaction低1.450%，
但比additive高0.024%，且最大N/D留出均未优于对应旧公式。
新公式自身加入H*后五折验证RMSE下降2.557%。
因此目前可以将它保留为替代质量响应形式，尚无性能依据全面替换原模型。

截图公式按以下形式实现，质量指数作用在Q上。addictive按additive解释。

$$L_{new}=E+AN^{-a}+BD^{-b}+\eta_Q(1-Q^\kappa).$$

加配比项的版本为 $L_{new,H}=L_{new}+\lambda H^*$。

对照additive的质量项为 $C(1-Q)^g$，interaction的质量项为 $C(1-Q)^gD^{-d}$。
每种公式都计算不加H*、加H*两个版本，共六个模型。原主流程的interaction_N是另一种质量与N交互形式，本实验的interaction沿用第二问配比对照中的质量与D交互定义。

## 数据和公平对照

固定E=1.68979756292，沿用已有B1经典拟合的估计值。N和D单位均为十亿。
H*=-0.0896732440612，仅读取第一问导出的M2模型求17域p_star，以17域均匀配比为参考，取13个验证损失域等权。
H*=F(p_star)-F(p_ref)，不是Shannon熵。没有再次读取原始A。

B6的360行与B7的450行合并后移除360条完全重复记录，共450行。
六个模型在同样的450行上拟合，使用同样的五折(N,D)分组验证，整组质量条件留在同一折。
OOF为每条样本在未参与该折训练时的预测。另留出最大N（50条）、最大D（90条），以及B6训练、B7新增Q条件测试（90条）。
所有验证只在相应训练部分拟合参数，各模型使用同等多初值最小二乘协议，未用全量拟合参数初始化验证折。
默认5个初值，固定随机种子20261008，解析Jacobian，迭代上限12000、收敛容差1e-11。
η_Q和C同范围[1e-6,1000]，κ和g同范围[1e-4,10]。λ≥0且E+λH*≥0。
B8的1704行用全量模型直接预测，保留原Q含义与数值，仅列为迁移诊断。
B6–B8均为基于真实数据校准的半合成补充集，不能称为直接实验观测。

## 主要成绩

R²越高越好。最大N/D留出属于网格边界外推，不能代替真实大规模独立实验证据。

| model | fit_unique | group_cv_oof | N_holdout | D_holdout | B6_to_B7_new_Q | B8_raw_Q_transfer_diagnostic |
| --- | --- | --- | --- | --- | --- | --- |
| additive_fixed_E | 0.971570 | 0.969047 | 0.891289 | 0.965740 | 0.983858 | -2.831755 |
| interaction_fixed_E | 0.972470 | 0.969160 | 0.894091 | 0.970196 | 0.983493 | -2.809423 |
| quality_power_fixed_E | 0.971554 | 0.969003 | 0.890608 | 0.965593 | 0.983847 | -2.832815 |
| additive_mixture | 0.972686 | 0.970582 | 0.891876 | 0.971673 | 0.983289 | -2.745026 |
| interaction_mixture | 0.973054 | 0.969696 | 0.892380 | 0.973158 | 0.983028 | -2.747627 |
| quality_power_mixture | 0.972689 | 0.970568 | 0.891371 | 0.971630 | 0.983282 | -2.743905 |

五折验证汇总误差，RMSE与MAE越低越好：

| model | R2 | RMSE | MAE |
| --- | --- | --- | --- |
| additive_fixed_E | 0.969047 | 0.059930 | 0.046310 |
| interaction_fixed_E | 0.969160 | 0.059821 | 0.046210 |
| quality_power_fixed_E | 0.969003 | 0.059973 | 0.046345 |
| additive_mixture | 0.970582 | 0.058426 | 0.045350 |
| interaction_mixture | 0.969696 | 0.059299 | 0.045960 |
| quality_power_mixture | 0.970568 | 0.058440 | 0.045387 |

成对比较：RMSE下降百分比=100×(1-RMSE_candidate/RMSE_reference)，负数为退步。
quality_formula只比较同H设置下的公式，H_star_offset只比较同公式有无常数修正。

| comparison | candidate | reference | delta_R2 | RMSE_reduction_pct |
| --- | --- | --- | --- | --- |
| quality_formula | quality_power_fixed_E | additive_fixed_E | -0.000044 | -0.071746 |
| quality_formula | quality_power_fixed_E | interaction_fixed_E | -0.000157 | -0.254859 |
| quality_formula | quality_power_mixture | additive_mixture | -0.000014 | -0.023855 |
| quality_formula | quality_power_mixture | interaction_mixture | 0.000872 | 1.449746 |
| H_star_offset | additive_mixture | additive_fixed_E | 0.001535 | 2.510311 |
| H_star_offset | interaction_mixture | interaction_fixed_E | 0.000535 | 0.871512 |
| H_star_offset | quality_power_mixture | quality_power_fixed_E | 0.001565 | 2.556965 |

新公式在多少折获得更低的RMSE：

| candidate | reference | folds_lower_RMSE | total_folds |
| --- | --- | --- | --- |
| quality_power_fixed_E | additive_fixed_E | 2 | 5 |
| quality_power_fixed_E | interaction_fixed_E | 2 | 5 |
| quality_power_mixture | additive_mixture | 3 | 5 |
| quality_power_mixture | interaction_mixture | 2 | 5 |

各验证场景的方向与幅度：

- quality_power_fixed_E 对比 additive_fixed_E，group_cv_oof：RMSE上升0.072%，R²变化-0.000044。
- quality_power_fixed_E 对比 additive_fixed_E，N_holdout：RMSE上升0.312%，R²变化-0.000680。
- quality_power_fixed_E 对比 additive_fixed_E，D_holdout：RMSE上升0.213%，R²变化-0.000146。
- quality_power_fixed_E 对比 additive_fixed_E，B6_to_B7_new_Q：RMSE上升0.032%，R²变化-0.000010。
- quality_power_fixed_E 对比 additive_fixed_E，B8_raw_Q_transfer_diagnostic：RMSE上升0.014%，R²变化-0.001061。
- quality_power_fixed_E 对比 interaction_fixed_E，group_cv_oof：RMSE上升0.255%，R²变化-0.000157。
- quality_power_fixed_E 对比 interaction_fixed_E，N_holdout：RMSE上升1.631%，R²变化-0.003483。
- quality_power_fixed_E 对比 interaction_fixed_E，D_holdout：RMSE上升7.445%，R²变化-0.004603。
- quality_power_fixed_E 对比 interaction_fixed_E，B6_to_B7_new_Q：RMSE下降1.078%，R²变化+0.000354。
- quality_power_fixed_E 对比 interaction_fixed_E，B8_raw_Q_transfer_diagnostic：RMSE上升0.307%，R²变化-0.023393。
- quality_power_mixture 对比 additive_mixture，group_cv_oof：RMSE上升0.024%，R²变化-0.000014。
- quality_power_mixture 对比 additive_mixture，N_holdout：RMSE上升0.233%，R²变化-0.000505。
- quality_power_mixture 对比 additive_mixture，D_holdout：RMSE上升0.076%，R²变化-0.000043。
- quality_power_mixture 对比 additive_mixture，B6_to_B7_new_Q：RMSE上升0.021%，R²变化-0.000007。
- quality_power_mixture 对比 additive_mixture，B8_raw_Q_transfer_diagnostic：RMSE下降0.015%，R²变化+0.001121。
- quality_power_mixture 对比 interaction_mixture，group_cv_oof：RMSE下降1.450%，R²变化+0.000872。
- quality_power_mixture 对比 interaction_mixture，N_holdout：RMSE上升0.468%，R²变化-0.001009。
- quality_power_mixture 对比 interaction_mixture，D_holdout：RMSE上升2.807%，R²变化-0.001528。
- quality_power_mixture 对比 interaction_mixture，B6_to_B7_new_Q：RMSE下降0.753%，R²变化+0.000255。
- quality_power_mixture 对比 interaction_mixture，B8_raw_Q_transfer_diagnostic：RMSE下降0.050%，R²变化+0.003722。

## 新公式参数和解释

| model | A | a | B | b | eta_Q | kappa | lambda | effective_E | active_bounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality_power_fixed_E | 0.447606 | 0.323332 | 1.425145 | 0.390589 | 0.359293 | 0.978454 | nan | 1.689798 |  |
| quality_power_mixture | 0.529124 | 0.283218 | 1.328565 | 0.299575 | 0.359277 | 1.025877 | 1.886355 | 1.520642 |  |

η_Q(1-Q^κ)在Q=0时为η_Q，在Q=1时为0，对κ>0单调下降。
κ=1时与g=1的additive相同。两个非线性公式不互相包含，不能保证替换后一定提升。
本次新公式的κ分别为0.978454、1.025877，均接近1。
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
共155项检查通过。全量拟合额外使用12初值复算，SSE改善在容差内，见multistart_stability.csv。
参数条件数、有效秩与边界命中保存在参数表。参数不可辨识程度和验证成绩应分开判断。

运行：`python src/q2_scaling.py`。结果写入`results/q2`。
按分组OOF RMSE最小选择主模型；其全量拟合参数导出到selected_model.json，供第三问读取。
