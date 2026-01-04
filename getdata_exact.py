import baostock as bs
import pandas as pd
import os
import time
from tqdm import tqdm

# --- 配置 ---
DATA_DIR = './market_data'
START_DATE = '2013-01-01'
END_DATE = '2025-12-20'  # 可以设为当天

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def fetch_historical_universe():
    """
    遍历每个月，获取当时的历史成分股名单
    """
    print("正在构建历史成分股名单 (这可能需要几分钟)...")

    # 生成从开始到现在的月末日期序列
    # 沪深300通常半年调整一次，但为了保险，我们按月查询
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')

    all_history = []
    unique_codes = set()

    # 进度条
    pbar = tqdm(date_range)
    for date in pbar:
        date_str = date.strftime('%Y-%m-%d')
        pbar.set_description(f"查询 {date_str}")

        # 查询该日期的成分股
        rs = bs.query_hs300_stocks(date=date_str)

        monthly_codes = []
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            # row[1] 是 code (e.g., 'sh.600519')
            monthly_codes.append(row[1])

        # 如果查不到（比如节假日），就尝试前推几天
        if not monthly_codes:
            # 简单处理：如果月末查不到，说明是非交易日，通常沿用上个月的名单即可
            # 这里为了严谨，我们暂且跳过，后续用 fillna 处理
            pass
        else:
            # 记录数据: 日期, 代码列表字符串
            # 为了方便存储，我们将当月所有代码拼成一个字符串 "sh.000001,sz.000002..."
            code_str = ",".join(monthly_codes)
            all_history.append({'date': date, 'codes': code_str})

            # 收集所有出现过的股票，用于稍后下载数据
            unique_codes.update(monthly_codes)

    # 保存历史名单表
    history_df = pd.DataFrame(all_history)
    history_df.set_index('date', inplace=True)

    # 填充：如果某个月没查到（因为假期），沿用上个月的名单
    history_df = history_df.asfreq('ME', method='ffill')

    save_path = os.path.join(DATA_DIR, 'history_cons.csv')
    history_df.to_csv(save_path)
    print(f"\n✅ 历史成分股名单已保存: {save_path}")
    print(f"🕵️‍♀️ 共有 {len(unique_codes)} 只股票曾入选沪深300")

    return list(unique_codes)


def download_prices(stock_list):
    """下载所有出现过的股票数据"""
    print(f"\n准备下载 {len(stock_list)} 只股票的历史行情...")

    all_data = []
    pbar = tqdm(stock_list)

    for bs_code in pbar:
        pure_code = bs_code.split('.')[1]
        pbar.set_description(f"下载 {pure_code}")

        # 下载日线，后复权
        rs = bs.query_history_k_data_plus(bs_code,
                                          "date,close",
                                          start_date=START_DATE, end_date=END_DATE,
                                          frequency="d", adjustflag="1")

        if rs.error_code == '0':
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['code'] = pure_code
                df['date'] = pd.to_datetime(df['date'])
                df['close'] = df['close'].astype(float)
                all_data.append(df)

    if not all_data:
        print("❌ 下载失败")
        return

    print("正在合并数据...")
    full_df = pd.concat(all_data)
    pivot_df = full_df.pivot(index='date', columns='code', values='close')
    pivot_df = pivot_df.ffill()  # 填充停牌

    save_path = os.path.join(DATA_DIR, 'all_stocks_close.csv')
    pivot_df.to_csv(save_path)
    print(f"✅ 全集股价数据已保存: {save_path}")


def download_industries(stock_list):
    """下载所有股票的行业信息"""
    print("\n正在更新行业分类信息...")
    # 注意：Baostock 的 query_stock_industry 只能查当前的
    # 对于历史回测，使用当前行业分类通常是可以接受的近似（行业属性极少变动）

    # 我们可以一次性拉取全市场的行业，然后跟我们的 list 做匹配
    rs = bs.query_stock_industry()

    ind_list = []
    while (rs.error_code == '0') & rs.next():
        ind_list.append(rs.get_row_data())

    ind_df = pd.DataFrame(ind_list, columns=rs.fields)
    # 处理代码格式 sh.600519 -> 600519
    ind_df['code'] = ind_df['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)

    # 筛选出我们要的那些股票
    # 注意：我们这里只要 pure_code (数字)
    pure_stock_list = [x.split('.')[1] for x in stock_list]
    filtered_ind = ind_df[ind_df['code'].isin(pure_stock_list)]

    final_df = filtered_ind[['code', 'industry']]
    save_path = os.path.join(DATA_DIR, 'stock_industry.csv')
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 行业数据已更新: {save_path}")


def download_benchmark():
    print("\n更新基准指数...")
    rs = bs.query_history_k_data_plus("sh.000300",
                                      "date,close", start_date=START_DATE, end_date=END_DATE,
                                      frequency="d", adjustflag="3")

    data = []
    while rs.next():
        data.append(rs.get_row_data())

    if data:
        df = pd.DataFrame(data, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.to_csv(os.path.join(DATA_DIR, 'benchmark.csv'))
        print("✅ 基准指数已保存")


def main():
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    try:
        # 1. 获取历史全集名单
        stock_list = fetch_historical_universe()

        # 2. 下载全集价格
        download_prices(stock_list)

        # 3. 下载行业信息
        download_industries(stock_list)

        # 4. 下载基准
        download_benchmark()

        print("\n🎉 所有数据准备完毕！现在可以运行 momentum_hsi300.py 了")

    finally:
        bs.logout()


if __name__ == "__main__":
    main()