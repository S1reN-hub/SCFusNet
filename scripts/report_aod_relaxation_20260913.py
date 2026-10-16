"""Write the full gate-sensitivity result, including adverse test cases."""
from pathlib import Path
import csv
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_aod_hardgate_relaxation_20260913 import OUT


def main():
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    cases = json.loads((OUT/'test_cases_with_ground_qc.json').read_text(encoding='utf-8'))
    chosen = json.loads((OUT/'selection_frozen.json').read_text(encoding='utf-8'))['selected']
    base = summary[0]
    selected = next(r for r in summary if r['candidate'] == chosen)
    rmin = next(r for r in cases if r['candidate'] == 'd15_c80')
    sc = [r for r in cases if r['candidate'] == chosen]
    ts = json.loads((OUT/chosen/'test_summary.json').read_text(encoding='utf-8'))
    am = {side:{k:float(np.mean([s['admitted'][side][k] for s in ts['metrics_by_seed'].values()])) for k in ('rmse','nrmse_pct','mae','nmbe_pct')} for side in ('before','after')}
    lines = ['# AOD 硬门槛放宽：当前七站测试集探索性结果', '',
        '结论：可以。仅将 AOD 异常门槛从 0.20 降至 0.15，保留晴空概率门槛 0.80，当前测试集即可纳入一个 IZA 案例的 5 个时效，并在三个种子下均产生改善。更宽的门槛可纳入 10 个测试案例，但也出现过度修正。', '',
        '## 实验口径', '',
        '固定现有 DynamicMLP、17 站训练和 7 站验证/测试划分，不新增数据。AOD 异常定义为预测 AOD550 减去训练集估计的站点月份参考值；这里调整的是异常门槛，不是 AOD 绝对值门槛。两路专家晴空概率的较小值用作共同晴空概率。', '',
        '预先固定 3×3 网格：异常门槛 {0.20, 0.15, 0.10}，晴空门槛 {0.80, 0.75, 0.70}。日照比例、太阳高度约束、门内连续支持系数、修正公式、优化器和训练轮数均沿用原配置。每组使用 42、43、44 三个种子重新训练修正网络。各组检查点按原验证准入时段 MAE 选择；跨门槛选择统一比较完整验证集的 MSE，避免不同准入子集之间直接比较指标。', '',
        f'验证集选择结果为 `{chosen}`（异常≥0.10，晴空概率≥0.70），选择文件在计算本轮测试预测误差前保存。它与 0.10/0.75 的验证 nRMSE 差异约为 0.000002 个百分点，二者表现接近，不据此声称阈值具有普适最优性。原门槛的重跑预测与既有结果完全一致。', '',
        '此前已多次查看该测试集，本轮还用测试输入检查了门槛覆盖情况，因此本报告属于既有测试集上的探索性敏感性分析。测试标签未参与本轮检查点或网格方案选择，所有预定组合和变差案例均保留。', '',
        '指标先按种子分别计算，再取三种子均值；案例数与小时数均为独立案例/预测时效记录数，不乘以三个种子。全测试集 nRMSE 使用相同的 601 个有效预测时效记录。', '',
        '## 完整门槛比较', '',
        '| AOD 异常下限 | 晴空概率下限 | 测试准入案例/小时 | 验证 nRMSE（%） | 测试 nRMSE（%） |',
        '|---:|---:|---:|---:|---:|',
        f'| 不修正 | — | — | {base["validation_before_nrmse"]:.5f} | {base["test_before_nrmse"]:.5f} |']
    for r in summary:
        lines.append(f'| {r["anomaly_min"]:.2f} | {r["common_clear_min"]:.2f} | {r["admitted_cases"]}/{r["admitted_hours"]} | {r["validation_after_nrmse"]:.5f} | {r["test_after_nrmse"]:.5f} |')
    lines += ['', '## 最小幅度调整：0.15 / 0.80', '',
        'IZA，起报时刻 2021-11-02 12:00 UTC，准入 1–5 h。预测 AOD550 为 0.240–0.243，异常为 0.163–0.165，共同晴空概率最低 0.909。该案例原本主要被 AOD 异常 0.20 门槛排除。', '',
        f'- 准入 5 h 的 nRMSE：{rmin["admitted_before_nrmse_pct"]:.2f}% → {rmin["admitted_after_nrmse_pct"]:.2f}%，对应 RMSE 相对下降 {rmin["admitted_relative_rmse_reduction_pct"]:.2f}%。',
        f'- 整个案例 6 h 的 nRMSE：{rmin["whole_case_before_nrmse_pct"]:.2f}% → {rmin["whole_case_after_nrmse_pct"]:.2f}%。',
        f'- 准入时段 MAE：{rmin["admitted_before_mae"]:.2f} → {rmin["admitted_after_mae"]:.2f} W/m²；平均下调 {rmin["mean_adjustment"]:.2f} W/m²，种子各自最大单小时下调量的均值为 {rmin["max_adjustment"]:.2f} W/m²。',
        '- 三个种子的案例 RMSE 均改善。地面逐分钟核对中，3–5 h 满足平稳且太阳直射可见条件；1–2 h 的波动略高于预设近晴空判据。因此可以讨论条件修正效果，不宜把全部 5 h 描述为严格晴空。', '',
        '这是“只小幅降低一道门槛”的可行性结果。它不等同于本轮网格的验证最优方案，正文采用时需保持配置与对应结果一致。', '',
        '## 验证选择方案：0.10 / 0.70', '',
        f'测试集纳入 10 个案例、38 个时效记录，其中 8 个案例的准入 RMSE 在三个种子下均改善，2 个案例在三个种子下均变差。准入子集 nRMSE 从 {am["before"]["nrmse_pct"]:.2f}% 降至 {am["after"]["nrmse_pct"]:.2f}%；完整测试集 nRMSE 从 {selected["test_before_nrmse"]:.4f}% 降至 {selected["test_after_nrmse"]:.4f}%。', '',
        '| 测试站点及起报时间（UTC） | 准入时效 | 准入 nRMSE，前→后（%） | RMSE 相对下降 | 近晴空时效/准入时效 |',
        '|---|---|---:|---:|---:|']
    for r in sorted(sc, key=lambda x:(x['station_id'], x['issue_time_utc'])):
        lines.append(f'| {r["station_id"]} {r["issue_time_utc"][:16].replace("T", " ")} | {r["admitted_leads"]} h | {r["admitted_before_nrmse_pct"]:.2f} → {r["admitted_after_nrmse_pct"]:.2f} | {r["admitted_relative_rmse_reduction_pct"]:+.2f}% | {r["admitted_observed_near_clear_hours"]}/{r["admitted_hours"]} |')
    lines += ['',
        'SEL（2024-02-09）的准入 1–5 h 全部满足地面近晴空判据，nRMSE 5.12%→2.80%，是较清楚的机制展示案例。DRA（2025-07-15）的准入 1–3 h 也全部满足该判据，nRMSE 4.75%→4.00%。GOB（2023-09-04）准入 3–6 h，nRMSE 15.39%→11.91%，其中 3–5 h 满足近晴空判据。', '',
        'TBL（2021-09-04）提供了明确的局限性：准入时段也满足近晴空判据，但基准预测已经偏低，下调后 nRMSE 3.45%→5.88%。AOD 正异常与较高晴空概率并不保证当前预测需要向下修正。DRA（2025-10-10）也出现轻度恶化。', '',
        '这些 AOD 值约为 0.15–0.29，建议表述为“AOD 正异常情景”，不要统一称为强沙尘或高 AOD 极端事件。近晴空判据仅用于结果解释，不进入推理门控；预测 AOD 未在本轮与同时刻观测 AOD 验证，案例改善本身也不能单独证明误差由气溶胶造成。', '',
        '## 对论文的建议', '',
        '当前已有真实测试案例可替代上一稿的训练案例。若保留原机制并强调最小改动，可展示 0.15/0.80 下的 IZA 案例；若采用本轮验证选择的 0.10/0.70，可用 SEL、DRA 和 GOB 展示条件修正，同时简短讨论 TBL 的过度修正。两个版本的数值不可交叉混用。全测试集改善较小，主张应集中在条件性修正能力。', '',
        '## 文件与核验', '',
        '- `protocol.json`：网格及验证选择规则；`selection_frozen.json`：测试评估前的选择记录。',
        '- `summary.csv`：全部网格结果；`test_cases_with_ground_qc.csv`：全部准入测试案例及地面条件。',
        '- 各候选目录：三种子检查点、训练历史、完整验证/测试逐时预测。',
        '- `audit.json`：原门槛重现误差为零；全部 57 个案例有效时效按原聚合方法复算真值误差为零。近晴空 QC 的逐分钟太阳角度筛选使 3 个门外低太阳高度时段均值有所不同，已单独记录；全部准入时效无此差异。',
        '- 原 DynamicMLP、旧修正检查点、数据划分及正文文件均保持原样，本报告为独立实验产物。', '']
    path = OUT/'硬门槛放宽_测试案例分析.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(report=str(path), selected=chosen, admitted_aggregate=am), ensure_ascii=True))


if __name__ == '__main__':
    main()
