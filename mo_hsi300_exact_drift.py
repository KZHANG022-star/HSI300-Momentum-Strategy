import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# --- 1. 配置与加载数据 ---
DATA_DIR = './market_data'


def load_data():
    print("正在加载本地数据...")

    # 1. 读取全集股价数据
    price_path = os.path.join(DATA_DIR, 'all_stocks_close.csv')
    if not os.path.exists(price_path):
        print("⚠️ 警告：未找到全集数据 all_stocks_close.csv，尝试加载旧数据...")
        price_path = os.path.join(DATA_DIR, 'csi300_close.csv')

    if not os.path.exists(price_path):
        raise FileNotFoundError(f"找不到数据文件: {price_path}，请先运行 getdata.py")

    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)

    # 2. 读取基准指数
    bench_path = os.path.join(DATA_DIR, 'benchmark.csv')
    benchmark = pd.read_csv(bench_path, index_col=0, parse_dates=True)
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark['close']

    # 3. 读取行业分类数据
    ind_path = os.path.join(DATA_DIR, 'stock_industry.csv')
    if not os.path.exists(ind_path):
        raise FileNotFoundError(f"找不到行业数据: {ind_path}")

    ind_df = pd.read_csv(ind_path, dtype={'code': str})
    ind_df['code'] = ind_df['code'].astype(str).str.strip()
    industry_map = dict(zip(ind_df['code'], ind_df['industry']))

    # 4. 读取历史成分股名单
    hist_path = os.path.join(DATA_DIR, 'history_cons.csv')
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"找不到历史成分股名单: {hist_path}，请运行新版 getdata.py")

    cons_df = pd.read_csv(hist_path, index_col=0, parse_dates=True)
    universe_dict = {}
    for date, row in cons_df.iterrows():
        codes_str = str(row['codes'])
        if not codes_str or codes_str == 'nan':
            universe_dict[date] = set()
            continue
        raw_codes = codes_str.split(',')
        clean_codes = {c.split('.')[1] if '.' in c else c for c in raw_codes}
        universe_dict[date] = clean_codes

    # 对齐数据
    common_dates = prices.index.intersection(benchmark.index)
    prices = prices.loc[common_dates]
    benchmark = benchmark.loc[common_dates]

    return prices, benchmark, industry_map, universe_dict, cons_df.index


# --- 2. 策略引擎 ---
class MomentumStrategy:
    def __init__(self, prices, benchmark, industry_map, universe_dict, universe_dates, top_n=30, max_per_ind=5):
        self.prices = prices
        self.benchmark = benchmark
        self.industry_map = industry_map
        self.universe_dict = universe_dict
        self.universe_dates = universe_dates

        self.top_n = top_n
        self.max_per_ind = max_per_ind
        self.cost_rate = 0.0015
        self.trade_logs = []

    def get_valid_universe(self, current_date):
        valid_dates = self.universe_dates[self.universe_dates <= current_date]
        if valid_dates.empty:
            return set()
        latest_report_date = valid_dates[-1]
        return self.universe_dict[latest_report_date]

    def run(self, start_date=None):
        print("正在计算 SPMO 动量因子...")
        mom_ret = self.prices.shift(21) / self.prices.shift(252) - 1
        daily_ret = self.prices.pct_change()
        volatility = daily_ret.rolling(252).std()
        raw_score = mom_ret / volatility

        # 2. 避免未来函数：因子向后移动一天 (T-1 用于 T)
        factor_score = raw_score.shift(1)
        # factor_score = mom_ret / volatility

        # --- 确定回测时间段 ---
        min_start_idx = 253
        full_valid_dates = self.prices.index[min_start_idx:]

        if start_date:
            target_date = pd.to_datetime(start_date)
            valid_dates = full_valid_dates[full_valid_dates >= target_date]
            if valid_dates.empty:
                print(f"错误：指定起点 {start_date} 太晚或太早，无可回测数据！")
                return pd.DataFrame()
            print(f"回测起点已调整为: {valid_dates[0].date()}")
        else:
            valid_dates = full_valid_dates
            print(f"默认回测起点: {valid_dates[0].date()}")

        # 调仓日：三月调仓 ('3ME')
        rb_dates_series = self.prices.index.to_series().resample('3ME').last()
        rebalance_dates = set(rb_dates_series.values)

        capital = 100000.0
        strategy_curve = [capital]

        # [修改点] 改用“当前持仓市值”来记录，而非“当前权重”
        # 初始化持仓市值向量 (0.0)
        current_val = pd.Series(0.0, index=self.prices.columns)

        print("开始回测 (动态成分股 + 行业中性 + 持仓漂移)...")

        for t, date in enumerate(valid_dates):
            # 1. 模拟持仓自然漂移 (Drift)
            # 每天收盘，持有的股票市值会随涨跌幅变化
            if t > 0:
                # 获取当日所有股票涨跌幅
                today_ret = self.prices.loc[date] / self.prices.shift(1).loc[date] - 1
                today_ret = today_ret.fillna(0)
                # 更新持仓市值：Value_t = Value_{t-1} * (1 + Ret_t)
                current_val = current_val * (1 + today_ret)

            # 计算当前总资产 (盘后，调仓前)
            if t == 0:
                total_asset = capital
            else:
                total_asset = current_val.sum()

            # 2. 调仓逻辑
            if date in rebalance_dates or t == 0:
                # A. 获取动态成分股
                valid_universe = self.get_valid_universe(date)

                # B. 获取因子
                current_scores = factor_score.loc[date]
                available_stocks = current_scores.index.intersection(list(valid_universe))
                score_series = current_scores.loc[available_stocks].dropna()

                if not score_series.empty:
                    # C. 行业中性选股
                    sorted_stocks = score_series.sort_values(ascending=False).index
                    selected = []
                    ind_counter = defaultdict(int)

                    for stock in sorted_stocks:
                        ind = self.industry_map.get(stock, 'Unknown')
                        if ind_counter[ind] < self.max_per_ind:
                            selected.append(stock)
                            ind_counter[ind] += 1
                        if len(selected) >= self.top_n:
                            break

                    targets = selected

                    # [修改点] D. 生成目标持仓市值 (而非权重)
                    # 目标是：总资产扣除成本后，均匀分配给 Targets
                    # 但成本取决于调仓量，这里做简化：先按总资产分配，算出调仓量，再从持仓中扣除成本

                    if len(targets) > 0:
                        target_val_per_stock = total_asset / len(targets)
                        target_val = pd.Series(0.0, index=self.prices.columns)
                        target_val[targets] = target_val_per_stock
                    else:
                        target_val = pd.Series(0.0, index=self.prices.columns)

                    # E. 计算买卖差额
                    val_diff = target_val - current_val

                    # F. 计算交易成本
                    turnover_val = val_diff.abs().sum()
                    transaction_cost = turnover_val * self.cost_rate

                    # G. 记录日志 (现在可以记录真实的权重变动了)
                    # 将市值变动转换为权重变动，以便 logs 格式兼容
                    if total_asset > 0:
                        weight_diff = val_diff / total_asset
                        # 只有当变动足够大时才记录 (避免极小的浮点数漂移日志)
                        if t > 0:
                            self.log_trade(date, weight_diff)

                    # H. 执行调仓并扣费
                    # 净资产 = 总资产 - 交易成本
                    net_asset = total_asset - transaction_cost

                    # 根据净资产重新分配持仓
                    if len(targets) > 0:
                        real_target_val = net_asset / len(targets)
                        current_val[:] = 0.0
                        current_val[targets] = real_target_val
                    else:
                        current_val[:] = 0.0

            # 3. 记录当日最终净值
            # 注意：这里的 current_val 已经是包含今日涨跌并完成调仓后的市值
            strategy_curve.append(current_val.sum())

        # 整理结果
        res_df = pd.DataFrame({'Strategy': strategy_curve[1:]}, index=valid_dates)
        bench_start_val = self.benchmark.loc[valid_dates[0]]
        res_df['Benchmark'] = self.benchmark.loc[valid_dates] / bench_start_val * capital

        return res_df

    def log_trade(self, date, pos_diff):
        """记录买卖操作"""
        for code, change in pos_diff.items():
            # 忽略过于微小的变动 (例如小于 0.1% 的再平衡)
            if abs(change) < 0.001: continue

            action = "买入" if change > 0 else "卖出"
            price = self.prices.at[date, code] if code in self.prices.columns else 0
            ind = self.industry_map.get(str(code), 'Unknown')

            self.trade_logs.append({
                '日期': date.strftime('%Y-%m-%d'),
                '代码': code,
                '行业': ind,
                '操作': action,
                '权重变动': f"{change:.2%}",  # 现在这里会显示真实的变动，如 0.5%, -3.33% 等
                '价格': f"{price:.2f}"
            })

    def save_logs(self):
        if not self.trade_logs:
            return None
        log_df = pd.DataFrame(self.trade_logs)
        save_path = os.path.join(DATA_DIR, 'trade_logs_dynamic.csv')
        log_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 交易记录已保存至: {save_path}")
        return log_df


# --- 3. 绩效计算工具 (保持不变) ---
def calculate_annual_returns(equity_curve):
    years = equity_curve.index.year.unique()
    annual_results = {}
    initial_capital = 100000.0
    previous_end_value = initial_capital

    for year in years:
        subset = equity_curve[str(year)]
        if subset.empty: continue
        current_end_value = subset.iloc[-1]
        ret = current_end_value / previous_end_value - 1
        annual_results[year] = ret
        previous_end_value = current_end_value

    return annual_results


def analyze_performance(equity_curve, risk_free_rate=0.02):
    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    trading_days = len(equity_curve)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / trading_days) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility
    roll_max = equity_curve.cummax()
    max_drawdown = ((equity_curve - roll_max) / roll_max).min()

    return {
        "累计收益": f"{total_return:.2%}",
        "年化收益": f"{cagr:.2%}",
        "年化波动": f"{volatility:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_drawdown:.2%}"
    }


# --- 4. 主程序 ---
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 加载数据
        df_prices, df_bench, ind_map, uni_dict, uni_dates = load_data()

        # 初始化策略 (保持参数不变)
        strategy = MomentumStrategy(df_prices, df_bench, ind_map, uni_dict, uni_dates, top_n=20, max_per_ind=5)

        # 运行 (这里可以传入 start_date='2019-01-01' 等)
        result = strategy.run('2019-01-01')

        # 保存日志
        logs_df = strategy.save_logs()

        # 打印分年度收益
        strat_annual = calculate_annual_returns(result['Strategy'])
        bench_annual = calculate_annual_returns(result['Benchmark'])

        print("\n" + "=" * 50)
        print(f"{'年份':<10} | {'策略收益':<15} | {'基准收益':<15}")
        print("-" * 50)
        for year in strat_annual.keys():
            strat_val = strat_annual[year]
            bench_val = bench_annual.get(year, 0.0)
            print(f"{year:<12} | {strat_val:<15.2%} | {bench_val:<15.2%}")
        print("=" * 50)

        # 打印总体指标
        print("\n" + "=" * 50)
        print(f"{'绩效指标':<12} | {'动态成分股策略':<12} | {'沪深300':<10}")
        print("-" * 50)

        metrics_strat = analyze_performance(result['Strategy'])
        metrics_bench = analyze_performance(result['Benchmark'])

        for key in metrics_strat.keys():
            print(f"{key:<15} | {metrics_strat[key]:<14} | {metrics_bench[key]:<12}")
        print("=" * 50 + "\n")

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(result['Strategy'], label='动态成分股HSMO', color='#d62728', linewidth=1.5)
        ax1.plot(result['Benchmark'], label='沪深300指数', color='#1f77b4', alpha=0.6)
        ax1.set_title('策略净值走势 (无幸存者偏差 + 真实持仓漂移)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        strat_dd = (result['Strategy'] - result['Strategy'].cummax()) / result['Strategy'].cummax()
        ax2.fill_between(strat_dd.index, strat_dd, 0, color='#d62728', alpha=0.3)
        ax2.set_title('策略回撤幅度')
        ax2.set_ylabel('回撤 %')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"出错啦: {e}")
        import traceback

        traceback.print_exc()