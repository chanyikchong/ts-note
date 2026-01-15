# 实用时间序列建模

<div class="interview-summary">
<strong>面试要点：</strong> 实用建模涉及：正确的训练/测试分割（时间分割，永不随机）、滚动窗口回测、处理部署（模型更新、监控）。关键问题：数据泄漏、概念漂移、不确定性量化。生产建议：从简单开始（朴素基线）、记录假设、持续监控预测准确度、制定备选策略。
</div>

## 核心定义

**回测：** 历史模拟模型的表现。

**前向验证：** 模拟生产的扩展窗口验证。

**概念漂移：** 特征与目标之间的关系随时间变化。

**模型重训练：** 定期使用新数据更新模型。

**预测协调：** 确保不同聚合级别的预测一致。

## 数学与推导

### 滚动原点回测

对于原点 $T_1, T_2, \ldots, T_m$ 和预测范围h：
$$\text{RMSE}(h) = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_{T_i+h} - \hat{y}_{T_i+h|T_i})^2}$$

### 预测偏差

$$\text{Bias} = \frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)$$

- 正偏差：系统性低估
- 负偏差：系统性高估

### 跟踪信号（用于监控）

$$TS_t = \frac{\sum_{i=1}^{t}e_i}{\text{MAD}}$$

如果 |TS| > 4，模型可能有偏差，需要重训练。

### 预测区间覆盖率

$$\text{Coverage} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}(y_t \in PI_t)$$

95%预测区间应该有~95%覆盖率；显著低于表示校准不良。

## 算法/模型概述

**生产预测流程：**

```python
def production_pipeline(data, config):
    """
    生产用的完整预测流程。
    """
    # 1. 数据验证
    validate_data(data)

    # 2. 特征工程
    features = create_features(data)

    # 3. 训练-测试分割（时间）
    train, holdout = temporal_split(features, config['holdout_size'])

    # 4. 通过交叉验证进行模型选择
    best_model = None
    best_score = float('inf')

    for model_class in config['candidate_models']:
        score = time_series_cv(train, model_class, config['cv_folds'])
        if score < best_score:
            best_model = model_class
            best_score = score

    # 5. 在完整训练集上最终训练
    model = best_model.fit(train)

    # 6. 保留集评估
    holdout_metrics = evaluate(model, holdout)

    # 7. 在所有数据上重训练用于部署
    final_model = best_model.fit(features)

    # 8. 生成带区间的预测
    forecasts = final_model.forecast(config['horizon'])
    intervals = final_model.prediction_intervals(config['horizon'])

    return {
        'model': final_model,
        'forecasts': forecasts,
        'intervals': intervals,
        'metrics': holdout_metrics
    }
```

**监控仪表板指标：**

| 指标 | 良好 | 警告 | 行动 |
|--------|------|---------|--------|
| MAPE | < 10% | 10-20% | > 20%：调查 |
| 偏差 | ≈ 0 | |bias| > 1σ | |bias| > 2σ：重训练 |
| PI覆盖率 | 90-100% | 80-90% | < 80%：重新校准 |
| 跟踪信号 | |TS| < 4 | 4-6 | > 6：重训练 |

## 常见陷阱

1. **随机训练-测试分割：** 导致数据泄漏。始终使用时间分割。

2. **优化错误的指标：** 最小化业务相关损失（如非对称成本），而不仅仅是RMSE。

3. **没有基线比较：** 声称"模型有效"却没有与朴素/季节性朴素比较。

4. **静态模型：** 新数据到达时不重训练。监控性能并定期重训练。

5. **忽略预测区间：** 没有不确定性的点预测会误导决策者。

6. **对保留集过拟合：** 如果多次在保留集上调整，它就变成了训练数据。使用嵌套交叉验证。

## 小型示例

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def tracking_signal(errors):
    cumsum = np.cumsum(errors)
    mad = np.mean(np.abs(errors))
    return cumsum[-1] / mad if mad > 0 else 0

# 模拟生产监控
np.random.seed(42)
n_periods = 12  # 12个月的监控

actuals = 100 + np.random.randn(n_periods) * 10
forecasts = actuals + np.random.randn(n_periods) * 5 + 2  # 轻微偏差

errors = actuals - forecasts

# 计算监控指标
print("=== 预测监控报告 ===\n")
print(f"MAE: {mean_absolute_error(actuals, forecasts):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actuals, forecasts)):.2f}")
print(f"MAPE: {mape(actuals, forecasts):.2f}%")
print(f"偏差: {np.mean(errors):.2f}")
print(f"跟踪信号: {tracking_signal(errors):.2f}")

# PI覆盖率检查（模拟95%区间）
pi_width = 1.96 * np.std(errors)
lower = forecasts - pi_width
upper = forecasts + pi_width
coverage = np.mean((actuals >= lower) & (actuals <= upper))
print(f"PI覆盖率: {coverage*100:.1f}%")

# 警报检查
print("\n=== 警报 ===")
if abs(np.mean(errors)) > 2 * np.std(errors):
    print("警告：检测到显著的预测偏差！")
if abs(tracking_signal(errors)) > 4:
    print("警告：跟踪信号超过阈值 - 考虑重训练")
if coverage < 0.80:
    print("警告：预测区间覆盖率过低")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么时间序列中随机训练-测试分割是错误的？</summary>

<div class="answer">
<strong>答案：</strong> 随机分割导致数据泄漏：
- 未来观测出现在训练集中
- 过去观测出现在测试集中
- 模型在训练期间"看到"了未来信息

**后果：**
- 过于乐观的评估指标
- 模型在生产中失败（因为未来不可用）
- 时间模式学习不正确

**正确方法：**
```
训练: [1, ..., T]    测试: [T+1, ..., T+h]
```
始终在过去训练，在未来测试。

<div class="pitfall">
<strong>常见陷阱：</strong> 直接在时间序列上使用sklearn的train_test_split()或KFold。使用TimeSeriesSplit或手动时间分割。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 什么是概念漂移，如何检测它？</summary>

<div class="answer">
<strong>答案：</strong> 概念漂移 = 特征与目标之间的关系随时间变化。

**类型：**
- **突变：** 突然变化（如COVID影响）
- **渐变：** 缓慢变化（如客户行为演变）
- **季节性：** 周期性模式变化
- **循环：** 在状态之间振荡

**检测方法：**
1. **性能监控：** 误差随时间增加
2. **统计检验：** 比较最近与历史分布
3. **控制图：** 跟踪预测误差，标记失控
4. **跟踪信号：** 累积偏差指示漂移

**响应：**
- 在最近数据上重训练
- 使用自适应模型（指数平滑）
- 减少回望窗口
- 添加状态指示变量

<div class="pitfall">
<strong>常见陷阱：</strong> 假设模型永远准确。安排定期监控和重训练。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 如何计算和解释跟踪信号？</summary>

<div class="answer">
<strong>答案：</strong>
$$TS_t = \frac{\text{RSFE}_t}{\text{MAD}_t} = \frac{\sum_{i=1}^{t}e_i}{\frac{1}{t}\sum_{i=1}^{t}|e_i|}$$

**解释：**
- TS ≈ 0：无系统性偏差
- TS > 0：系统性低估
- TS < 0：系统性高估
- |TS| > 4：可能存在显著偏差（需要行动）

**为什么使用它：**
- 通过MAD标准化以便比较
- 随时间累积证据
- 区分随机误差和系统性偏差

**更新频率：**
按月或按季度检查；每日TS太嘈杂。

<div class="pitfall">
<strong>常见陷阱：</strong> 对每次TS波动都做出反应。等待持续信号（多个周期|TS| > 4）再重训练。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 如果你的95%预测区间只有75%的覆盖率，这意味着什么？</summary>

<div class="answer">
<strong>答案：</strong> 模型低估了不确定性——区间太窄。

**可能的原因：**
1. **模型误设：** 真实残差比估计的大
2. **重尾：** 数据有离群值，正态假设没有捕获
3. **异方差：** 未建模的非恒定方差
4. **遗漏模式：** 未建模的季节性或趋势增加方差

**解决方案：**
1. 使用自助法预测区间（更稳健）
2. 应用方差调整：将宽度乘以覆盖率校正因子
3. 建模异方差（GARCH）或使用分位数回归
4. 改进基础模型以捕获更多模式

**调整公式：**
如果覆盖率 = 75%但目标 = 95%，比例因子 ≈ $z_{0.975}/z_{0.875}$ = 1.96/1.15 ≈ 1.7

<div class="pitfall">
<strong>常见陷阱：</strong> 报告窄区间以显得准确。利益相关者需要真实的不确定性；太窄的区间导致糟糕的决策。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你的需求预测模型在测试中表现良好，但生产准确度差很多。可能是什么原因？</summary>

<div class="answer">
<strong>答案：</strong> 训练-测试与生产差距的常见原因：

1. **测试中的数据泄漏：**
   - 特征使用了预测时不可用的信息
   - 训练-测试分割不是严格的时间分割

2. **特征可用性：**
   - 特征在历史上可用但在生产中延迟
   - 外部数据源没有实时更新

3. **分布偏移：**
   - 测试期是异常的/幸运的
   - 生产面临不同条件（季节性、促销）

4. **数据质量：**
   - 生产数据有历史中没有的错误/延迟
   - 缺失值处理方式不同

5. **目标泄漏：**
   - 测试在简单的时间范围评估；生产需要更长

**诊断：**
1. 验证特征工程中无泄漏
2. 比较特征分布：测试 vs 生产
3. 在多个时期回测（不只是一个）
4. 监控生产中的输入数据质量
5. 检查生产时间范围是否与测试匹配

<div class="pitfall">
<strong>常见陷阱：</strong> 在方便的时期测试并假设它能推广。始终在跨越不同条件的多个训练-测试分割上回测。
</div>
</div>
</details>

## 参考文献

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapters 5, 12.
2. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *IJF*, 36(1), 54-74.
3. Gama, J., Zliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1-37.
4. Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. *IJF*, 32(3), 788-803.
