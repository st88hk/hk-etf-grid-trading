# app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right

# ---------------------------
# Helpers / Indicators / Cost
# ---------------------------
def parse_volume(volume_input):
    """解析成交量（支持 k, m, w, 万, 亿 等）。
    注意：中文“兆”语义不统一，避免使用；若使用请明确单位含义。
    """
    if volume_input is None:
        return 0
    s = str(volume_input).strip().lower()
    if s == "":
        return 0
    multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'w': 10_000,
        '万': 10_000,
        '亿': 100_000_000
        # 不再默认映射 '兆'，以避免混淆
    }
    # 支持带逗号或空格
    s = s.replace(",", "").replace(" ", "")
    unit = None
    number_part = s
    for u in multipliers:
        if s.endswith(u):
            unit = u
            number_part = s[:-len(u)]
            break
    try:
        num = float(number_part)
    except Exception:
        return 0
    return int(round(num * multipliers.get(unit, 1)))


def calculate_trade_cost_simple(amount, cfg, is_single_side=True):
    """计算成本，cfg 是字典包含费率与固定费"""
    # 滑点：按百分比（如 0.15 表示 0.15%）
    slippage_cost = amount * (cfg["slippage_pct"] / 100)
    trade_fee = amount * (cfg["trade_fee_pct"] / 100)
    settlement_fee = amount * (cfg["settlement_fee_pct"] / 100)
    sfc_fee = amount * (cfg["sfc_fee_pct"] / 100)
    frc_fee = amount * (cfg["frc_fee_pct"] / 100)
    platform_fee = cfg["platform_fee"]  # 每笔固定
    single_total = platform_fee + trade_fee + settlement_fee + sfc_fee + frc_fee + slippage_cost
    if not is_single_side:
        # 买+卖 两边费用（平台费两次）
        return round(single_total * 2, 2)
    return round(single_total, 2)


def calculate_atr(highs, lows, closes, period=14):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.concatenate(([closes[0]], closes[:-1]))),
                                            np.abs(lows - np.concatenate(([closes[0]], closes[:-1])))))
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().round(6).tolist()
    # pad with None for first (period-1) if desired: but returning full list is fine
    return atr


def calculate_narrow_bollinger(prices, period=10, num_std=1.5):
    s = pd.Series(prices)
    ma = s.rolling(period, min_periods=1).mean()
    std = s.rolling(period, min_periods=1).std().fillna(0)
    upper = (ma + num_std * std).round(6).tolist()
    mid = ma.round(6).tolist()
    lower = (ma - num_std * std).round(6).tolist()
    return upper, mid, lower


# ---------------------------
# Grid generation (等距)
# ---------------------------
def generate_intraday_grid_arithmetic(current_price, spacing_pct, grid_count, grid_upper, grid_lower):
    """等距网格：每格 = current_price * spacing_pct%
    grid_count 为总档数（买+卖）
    返回：buy_grids(升序低->高), sell_grids(升序低->高)
    """
    spacing = spacing_pct / 100.0
    half = grid_count // 2
    # 买入：current_price - spacing, current_price - 2*spacing ...
    buy = [round(current_price * (1 - spacing * (i + 1)), 4) for i in range(half)]
    sell = [round(current_price * (1 + spacing * (i + 1)), 4) for i in range(half)]
    # 过滤区间
    buy = [p for p in buy if p >= grid_lower * 0.99]
    sell = [p for p in sell if p <= grid_upper * 1.01]
    buy.sort()   # 升序（低->高）
    sell.sort()  # 升序（低->高）
    return buy, sell


# ---------------------------
# 回测：更真实的触发与净值曲线
# ---------------------------
def backtest_intraday_strategy_improved(principal, current_price, buy_grids, sell_grids, minute_data, cfg):
    """
    更真实的回测：
    - 同一分钟内可以多次触发多个档位（price range within minute)
    - 每分钟以当分钟 close 做净值估值并记录
    - 使用 shares per lot (100) 做整数手约束
    """
    trade_records = []
    cash = principal * cfg.get("initial_cash_pct", 0.5)
    shares = 0
    shares_per_lot = cfg.get("shares_per_lot", 100)
    single_trade_amount = cfg.get("single_trade_amount", principal * 0.05)

    # Make copies and keep them sorted
    buy_list = sorted(buy_grids)   # ascending (low->high)
    sell_list = sorted(sell_grids) # ascending (low->high)

    net_values = []
    timestamps = []
    holdings_history = []

    for row in minute_data:
        t = row["time"]
        high = row["high"]
        low = row["low"]
        close = row["close"]

        # BUY: while there exists buy_price >= low and buy_price <= high? For buy we trigger when price <= buy_price
        # Since buy_list sorted low->high, we should process from lowest (most attractive)
        triggered = True
        while triggered:
            triggered = False
            # find all buy prices where low <= price <= high (price reached in this minute)
            # for intraday granularity, if low <= buy_price (we treat it as triggered)
            if buy_list:
                # find leftmost index where buy_price >= low
                # iterate from smallest to largest and pick first that is >= low and <= close? we allow low trigger
                for bp in buy_list:
                    if low <= bp:
                        # Can buy if have enough cash
                        max_lots_by_amount = int((single_trade_amount / bp) // shares_per_lot)
                        max_lots_by_cash = int(cash // (bp * shares_per_lot))
                        lots = min(max_lots_by_amount, max_lots_by_cash)
                        if lots <= 0:
                            # can't afford this buy
                            continue
                        buy_shares = lots * shares_per_lot
                        buy_amount = buy_shares * bp
                        cost = calculate_trade_cost_simple(buy_amount, cfg, is_single_side=True)
                        # execute
                        shares += buy_shares
                        cash -= (buy_amount + cost)
                        trade_records.append({
                            "时间": t,
                            "类型": "买入",
                            "价格(港元)": bp,
                            "股数": buy_shares,
                            "金额(港元)": round(buy_amount, 2),
                            "成本(港元)": round(cost, 2),
                            "剩余现金(港元)": round(cash, 2),
                            "持仓股数": shares
                        })
                        buy_list.remove(bp)
                        triggered = True
                        break  # 重新检查（因为列表改变）
            # SELL: while there exists sell_price <= high
            if sell_list:
                # since sell_list ascending, find any sell price <= high; better to sell highest first to realize max profit
                for sp in reversed(sell_list):
                    if high >= sp and shares >= shares_per_lot:
                        # determine how many lots to sell (bounded by single_trade_amount and current shares)
                        max_lots_by_amount = int((single_trade_amount / sp) // shares_per_lot)
                        max_lots_by_shares = int(shares // shares_per_lot)
                        lots = min(max_lots_by_amount, max_lots_by_shares)
                        if lots <= 0:
                            continue
                        sell_shares = lots * shares_per_lot
                        sell_amount = sell_shares * sp
                        cost = calculate_trade_cost_simple(sell_amount, cfg, is_single_side=True)
                        # execute
                        shares -= sell_shares
                        cash += (sell_amount - cost)
                        trade_records.append({
                            "时间": t,
                            "类型": "卖出",
                            "价格(港元)": sp,
                            "股数": sell_shares,
                            "金额(港元)": round(sell_amount, 2),
                            "成本(港元)": round(cost, 2),
                            "剩余现金(港元)": round(cash, 2),
                            "持仓股数": shares
                        })
                        sell_list.remove(sp)
                        triggered = True
                        break  # 重新检查

        # 每分钟记录净值（用当分钟 close 来估值）
        holdings_value = shares * close
        net_value = cash + holdings_value
        timestamps.append(t)
        net_values.append(net_value)
        holdings_history.append(shares)

    final_total = net_values[-1] if net_values else (cash + shares * current_price)
    total_profit = final_total - principal
    profit_rate = (total_profit / principal) * 100 if principal != 0 else 0

    # 统计
    buys = [r for r in trade_records if r["类型"] == "买入"]
    sells = [r for r in trade_records if r["类型"] == "卖出"]
    total_buy_count = len(buys)
    total_sell_count = len(sells)
    avg_trade_profit = (total_profit / (total_buy_count + total_sell_count)) if (total_buy_count + total_sell_count) > 0 else 0

    # 最大回撤按净值序列计算
    max_drawdown = calculate_max_drawdown_from_series(net_values)

    return {
        "trade_records": trade_records,
        "final_total_value": round(final_total, 2),
        "total_profit": round(total_profit, 2),
        "profit_rate": round(profit_rate, 4),
        "total_buy_count": total_buy_count,
        "total_sell_count": total_sell_count,
        "avg_trade_profit": round(avg_trade_profit, 2),
        "max_drawdown": max_drawdown,
        "net_values": net_values,
        "timestamps": timestamps,
        "holdings_history": holdings_history
    }


def calculate_max_drawdown_from_series(net_values):
    if not net_values:
        return 0.0
    series = pd.Series(net_values)
    running_max = series.cummax()
    drawdown = (running_max - series) / running_max
    max_dd = drawdown.max() * 100  # in %
    return round(float(max_dd), 4)


# ---------------------------
# Default minute data generator
# ---------------------------
def generate_default_minute_data(current_price=27.5, interval=5):
    """生成按 interval 分钟的分钟数据，跳过午休"""
    minute_data = []
    # Helper to step from start to end inclusive in interval minutes
    def create_range(start_str, end_str):
        start = datetime.strptime(start_str, "%H:%M")
        end = datetime.strptime(end_str, "%H:%M")
        t = start
        while t <= end:
            yield t
            t += timedelta(minutes=interval)

    for t in create_range("09:30", "12:00"):
        price_offset = np.random.uniform(-0.003, 0.003)
        close_price = current_price * (1 + price_offset)
        high = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low = close_price * (1 - np.random.uniform(0.0005, 0.001))
        volume = int(np.random.uniform(8000, 25000))
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high, 4), "low": round(low, 4), "close": round(close_price, 4), "volume": volume})

    for t in create_range("13:00", "16:00"):
        price_offset = np.random.uniform(-0.003, 0.003)
        trend_bias = 0.001 if np.random.random() > 0.5 else -0.001
        close_price = current_price * (1 + price_offset + trend_bias)
        high = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low = close_price * (1 - np.random.uniform(0.0005, 0.001))
        volume = int(np.random.uniform(6000, 20000))
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high, 4), "low": round(low, 4), "close": round(close_price, 4), "volume": volume})

    return minute_data


# ---------------------------
# Streamlit UI parts
# ---------------------------
def render_sidebar():
    st.sidebar.header("1. 基础参数")
    principal = st.sidebar.number_input("交易本金（港元）", min_value=1000.0, max_value=1_000_000.0, value=30_000.0, step=1_000.0)
    etf_code = st.sidebar.text_input("ETF 代码", value="02800.HK")
    current_price = st.sidebar.number_input(f"{etf_code} 当前价格（港元）", min_value=0.0001, value=27.5, format="%.4f")

    st.sidebar.header("2. 手续费 & 滑点（可配置）")
    cfg = {
        "platform_fee": st.sidebar.number_input("平台费（每笔，港元）", min_value=0.0, value=15.0, step=1.0),
        "trade_fee_pct": st.sidebar.number_input("交易佣金（%）", min_value=0.0, value=0.00565, step=0.00001),
        "settlement_fee_pct": st.sidebar.number_input("交收费（%）", min_value=0.0, value=0.0042, step=0.00001),
        "sfc_fee_pct": st.sidebar.number_input("证监会费（%）", min_value=0.0, value=0.0027, step=0.00001),
        "frc_fee_pct": st.sidebar.number_input("FRC费（%）", min_value=0.0, value=0.00015, step=0.00001),
        "slippage_pct": st.sidebar.number_input("滑点（%，每笔估算）", min_value=0.0, value=0.15, step=0.01),
    }
    # Convert percent inputs stored as % (e.g. 0.00565) already reflect percent value in your original code — keep user-facing clarity:
    # The UI expects percent value (like 0.00565) to mean 0.00565% (same as original). We will treat it as percent directly in calculate_trade_cost_simple.
    st.sidebar.markdown("说明：上面输入的百分比都按“%”填写，例如输入 `0.15` 表示 `0.15%`。")

    st.sidebar.header("3. 网格配置")
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 10, 15], index=1)
    grid_type = st.sidebar.radio("网格类型", ["动态间距（基于ATR）", "固定间距（手动）"])
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 10, 30, 16, 1)
    fixed_spacing_pct = st.sidebar.slider("固定间距（%）", 0.1, 2.0, 0.3, 0.05) if grid_type != "动态间距（基于ATR）" else None

    st.sidebar.header("4. 回测与仓位")
    initial_cash_pct = st.sidebar.slider("初始现金占本金比（用于日内可用现金）", 0.1, 1.0, 0.5, 0.05)
    single_trade_pct = st.sidebar.slider("单次交易金额占本金（%）", 0.5, 20.0, 5.0, 0.5)
    shares_per_lot = st.sidebar.number_input("每手股数", min_value=1, value=100, step=1)

    cfg.update({
        "initial_cash_pct": initial_cash_pct,
        "single_trade_amount": principal * (single_trade_pct / 100.0),
        "shares_per_lot": int(shares_per_lot)
    })

    return principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct


def render_tab_data():
    st.subheader("分钟级数据（支持编辑）")
    if "minute_data" not in st.session_state:
        st.session_state.minute_data = generate_default_minute_data()

    data_interval = st.session_state.get("data_interval", 5)
    st.write(f"当前数据周期：{data_interval} 分钟（生成数据用于回测）")

    table = []
    for d in st.session_state.minute_data:
        vol_str = f"{d['volume']}" if d['volume'] < 1000 else (f"{d['volume']/1000:.1f}k" if d['volume'] < 10000 else f"{d['volume']/10000:.2f}万")
        table.append({"时间": d["time"], "最高价(港元)": d["high"], "最低价(港元)": d["low"], "收盘价(港元)": d["close"], "成交量": vol_str})

    edited = st.data_editor(table, use_container_width=True, hide_index=True, key="minute_editor")
    if st.button("保存数据"):
        updated = []
        for idx, row in enumerate(edited):
            time_str = row["时间"].strip()
            try:
                hour, minute = map(int, time_str.split(":"))
                # 简单时段验证
                if not ((hour == 9 and minute >= 30) or (10 <= hour < 12) or (hour == 12 and minute == 0) or (13 <= hour < 16) or (hour == 16 and minute == 0)):
                    st.warning(f"第{idx+1}行时间 {time_str} 可能不在交易时段，已跳过")
                    continue
            except:
                st.warning(f"第{idx+1}行时间格式错误，已跳过")
                continue
            try:
                high = float(row["最高价(港元)"])
                low = float(row["最低价(港元)"])
                close = float(row["收盘价(港元)"])
                if high < low or close < low or close > high:
                    # 修正
                    hi = max(high, low, close)
                    lo = min(high, low, close)
                    cl = max(min(close, hi), lo)
                    high, low, close = hi, lo, cl
                vol = parse_volume(row["成交量"])
                updated.append({"time": time_str, "high": round(high, 4), "low": round(low, 4), "close": round(close, 4), "volume": vol})
            except Exception as e:
                st.warning(f"第{idx+1}行数据解析失败：{e}")
        # sort by time
        updated.sort(key=lambda x: datetime.strptime(x["time"], "%H:%M"))
        st.session_state.minute_data = updated
        st.success(f"保存 {len(updated)} 条分钟数据")

    if st.button("生成默认数据"):
        st.session_state.minute_data = generate_default_minute_data()
        st.success("已生成默认分钟数据")


def render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct):
    st.subheader("网格参数与生成")
    minute_data = st.session_state.get("minute_data", generate_default_minute_data())
    highs = [d["high"] for d in minute_data]
    lows = [d["low"] for d in minute_data]
    closes = [d["close"] for d in minute_data]

    atr = calculate_atr(highs, lows, closes, period=14)
    latest_atr = atr[-1] if atr else (max(highs[-5:]) - min(lows[-5:]))/2
    b_up, b_mid, b_low = calculate_narrow_bollinger(closes, period=10, num_std=1.5)
    latest_upper = b_up[-1] if b_up else current_price * 1.01
    latest_lower = b_low[-1] if b_low else current_price * 0.99

    # Dynamic spacing based on ATR (percent)
    base_spacing_pct = (latest_atr * 0.6 / current_price) * 100
    # ensure min spacing covers costs
    single_trade_amount = cfg["single_trade_amount"]
    round_trip_cost = calculate_trade_cost_simple(single_trade_amount, cfg, is_single_side=False)
    min_safe_spacing_pct = (round_trip_cost / single_trade_amount) * 100 * 1.2
    final_spacing_pct = max(base_spacing_pct, min_safe_spacing_pct, 0.2)

    if grid_type != "动态间距（基于ATR）":
        final_spacing_pct = fixed_spacing_pct

    # bounds
    grid_upper = min(latest_upper * 1.005, current_price * 1.02)
    grid_lower = max(latest_lower * 0.995, current_price * 0.98)

    # ensure even grid_count
    if grid_count % 2 != 0:
        grid_count += 1

    buy_grids, sell_grids = generate_intraday_grid_arithmetic(current_price, final_spacing_pct, grid_count, grid_upper, grid_lower)
    st.session_state.update({
        "grid_params": {
            "grid_upper": round(grid_upper, 4), "grid_lower": round(grid_lower, 4),
            "spacing_pct": round(final_spacing_pct, 4), "grid_count": grid_count,
            "atr": round(latest_atr, 6), "round_trip_cost": round(round_trip_cost, 2),
            "single_trade_amount": round(single_trade_amount, 2)
        },
        "buy_grids": buy_grids,
        "sell_grids": sell_grids,
        "data_interval": data_interval
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 基础信息")
        st.write(f"交易标的：{etf_code}")
        st.write(f"本金：{principal:,.0f} 港元")
        st.write(f"当前价：{current_price:.4f} 港元")
    with col2:
        st.markdown("#### 网格参数")
        p = st.session_state["grid_params"]
        st.write(f"区间：{p['grid_lower']} ~ {p['grid_upper']} 港元")
        st.write(f"间距：{p['spacing_pct']} %")
        st.write(f"档数：{p['grid_count']} (买{len(buy_grids)} / 卖{len(sell_grids)})")
        st.write(f"单次交易额：{p['single_trade_amount']} 港元")
        st.write(f"双边成本：{p['round_trip_cost']} 港元")

    st.divider()
    col_b, col_s = st.columns(2)
    with col_b:
        st.markdown("##### 买入网格")
        if buy_grids:
            st.dataframe(pd.DataFrame({"买入档位": [f"买{i+1}" for i in range(len(buy_grids))], "价格(港元)": buy_grids}), use_container_width=True)
        else:
            st.warning("未生成买入网格")

    with col_s:
        st.markdown("##### 卖出网格")
        if sell_grids:
            st.dataframe(pd.DataFrame({"卖出档位": [f"卖{i+1}" for i in range(len(sell_grids))], "价格(港元)": sell_grids}), use_container_width=True)
        else:
            st.warning("未生成卖出网格")

    if st.button("开始回测"):
        with st.spinner("回测进行中..."):
            result = backtest_intraday_strategy_improved(
                principal=principal,
                current_price=current_price,
                buy_grids=buy_grids.copy(),
                sell_grids=sell_grids.copy(),
                minute_data=minute_data,
                cfg=cfg
            )
            st.session_state.backtest_result = result
            st.success("回测完成，切换到回测页查看详情")


def render_tab_backtest(principal, etf_code):
    st.subheader("回测结果")
    result = st.session_state.get("backtest_result")
    if not result:
        st.info("请先生成网格并运行回测")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初始本金(港元)", f"{principal:,.0f}")
    col2.metric("最终市值(港元)", f"{result['final_total_value']:,.2f}")
    col3.metric("总收益(港元)", f"{result['total_profit']:,.2f}", delta=f"{result['profit_rate']:.4f}%")
    col4.metric("最大回撤(%)", f"{result['max_drawdown']:.4f}%")

    st.divider()
    st.markdown("### 净值曲线（按分钟）")
    df_nv = pd.DataFrame({"时间": result["timestamps"], "净值": result["net_values"], "持仓": result["holdings_history"]})
    df_nv.index = pd.to_datetime(df_nv["时间"], format="%H:%M")
    st.line_chart(df_nv[["净值"]])

    st.divider()
    st.markdown("### 交易明细")
    trades = result["trade_records"]
    if trades:
        df_tr = pd.DataFrame(trades)
        st.dataframe(df_tr, use_container_width=True)
        csv = df_tr.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载交易记录 CSV", data=csv, file_name=f"trade_records_{etf_code}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    else:
        st.info("本次回测未触发任何交易")

    st.divider()
    # 简单策略建议
    if result["profit_rate"] > 0.1:
        st.success("回测结果盈利，建议在模拟盘进一步验证（小仓位实盘）")
    elif result["profit_rate"] >= 0:
        st.info("回测持平，考虑优化网格间距或增加档数")
    else:
        st.error("回测亏损，建议暂停实盘并回看参数/数据")


def main():
    st.set_page_config(page_title="香港日内网格 T+0（改进版）", layout="wide")
    st.title("香港股市日内 T+0 网格交易工具 — 改进版")
    principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct = render_sidebar()

    tabs = st.tabs(["分钟数据", "网格策略", "回测结果"])
    with tabs[0]:
        render_tab_data()
    with tabs[1]:
        render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct)
    with tabs[2]:
        render_tab_backtest(principal, etf_code)

    st.caption("说明：本工具用于策略研究与回测，不构成投资建议。实盘需对接券商 API 并做好风控。")


if __name__ == "__main__":
    main()
