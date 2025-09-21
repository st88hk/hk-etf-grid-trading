# app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bisect import bisect_left

# ---------------------------
# Utility / Indicator / Cost
# ---------------------------

def parse_volume(volume_input):
    """解析成交量（支持 k, m, w, 万, 亿 等）。"""
    if volume_input is None:
        return 0
    s = str(volume_input).strip().lower().replace(",", "").replace(" ", "")
    if s == "":
        return 0
    multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'w': 10_000,
        '万': 10_000,
        '亿': 100_000_000
    }
    unit = None
    number_part = s
    for u in multipliers:
        if s.endswith(u):
            unit = u
            number_part = s[:-len(u)]
            break
    try:
        num = float(number_part)
    except:
        return 0
    return int(round(num * multipliers.get(unit, 1)))


def recommend_slippage_by_turnover(avg_daily_turnover):
    """
    基于日均成交额给出滑点推荐（粗略经验值）
    输入：avg_daily_turnover 单位：港元
    返回： (low_pct, mid_pct, high_pct) 单位为百分比（例如 0.15 表示 0.15%）
    """
    # 如果没填或 0，给通用推荐
    if not avg_daily_turnover or avg_daily_turnover <= 0:
        return (0.05, 0.15, 0.3)

    # 经验规则（可根据实盘校准）
    if avg_daily_turnover >= 1_000_000_000:  # >= 10亿
        return (0.03, 0.06, 0.12)
    if avg_daily_turnover >= 500_000_000:  # >=5亿
        return (0.05, 0.12, 0.2)
    if avg_daily_turnover >= 50_000_000:  # >=5000万
        return (0.1, 0.25, 0.5)
    # 小盘
    return (0.3, 0.7, 1.5)


def calculate_trade_cost_simple(amount, cfg, is_single_side=True):
    """计算交易成本：平台费（固定）+ 各项百分比费用 + 滑点（百分比）"""
    slippage_cost = amount * (cfg["slippage_pct"] / 100.0)
    trade_fee = amount * (cfg["trade_fee_pct"] / 100.0)
    settlement_fee = amount * (cfg["settlement_fee_pct"] / 100.0)
    sfc_fee = amount * (cfg["sfc_fee_pct"] / 100.0)
    frc_fee = amount * (cfg["frc_fee_pct"] / 100.0)
    platform_fee = cfg["platform_fee"]
    single_total = platform_fee + trade_fee + settlement_fee + sfc_fee + frc_fee + slippage_cost
    if not is_single_side:
        return round(single_total * 2, 2)
    return round(single_total, 2)


def calculate_atr(highs, lows, closes, period=14):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().round(6).tolist()
    return atr


def calculate_vwap(minute_data):
    """
    计算当日 VWAP（以 minute_data 为序）
    minute_data: list of dict 有 keys 'high','low','close','volume'
    VWAP = sum(price*volume) / sum(volume)，通常用当分钟的 close 作为 price 近似
    """
    prices = np.array([d["close"] for d in minute_data], dtype=float)
    volumes = np.array([d["volume"] for d in minute_data], dtype=float)
    if volumes.sum() == 0:
        return None
    vwap = (prices * volumes).sum() / volumes.sum()
    return round(float(vwap), 6)


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
    """等距网格（算术）：每格按百分比间距递增/递减"""
    spacing = spacing_pct / 100.0
    half = grid_count // 2
    buy = [round(current_price * (1 - spacing * (i + 1)), 4) for i in range(half)]
    sell = [round(current_price * (1 + spacing * (i + 1)), 4) for i in range(half)]
    buy = [p for p in buy if p >= grid_lower * 0.99]
    sell = [p for p in sell if p <= grid_upper * 1.01]
    buy.sort()
    sell.sort()
    return buy, sell


# ---------------------------
# Backtest: improved + stoploss/takeprofit/positionlimit + VWAP usage
# ---------------------------
def backtest_intraday_strategy_improved(principal, current_price, buy_grids, sell_grids, minute_data, cfg):
    """
    回测说明（改进点）：
    - 同一分钟可以连续触发多档买卖（使用当分钟 high/low 判定触发）
    - 每分钟以当分钟 close 估值并记录净值，计算 MDD
    - 支持全局止损/止盈阈值（按净值）
    - 支持最大持仓比例（max_position_pct）
    - 若止损/止盈触发，会尽量在触发的分钟以 close 价格全部清仓（近似）
    """
    trade_records = []
    cash = principal * cfg.get("initial_cash_pct", 0.5)
    shares = 0
    shares_per_lot = cfg.get("shares_per_lot", 100)
    single_trade_amount = cfg.get("single_trade_amount", principal * 0.05)

    buy_list = sorted(buy_grids)   # 升序（低->高）
    sell_list = sorted(sell_grids) # 升序（低->高）

    net_values = []
    timestamps = []
    holdings_history = []

    stop_loss_pct = cfg.get("stop_loss_pct", None)  # e.g., 3 means 3%
    take_profit_pct = cfg.get("take_profit_pct", None)
    max_position_pct = cfg.get("max_position_pct", 0.5)  # e.g., 0.5 表示 50%本金可建仓

    initial_net = principal

    for row in minute_data:
        t = row["time"]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        volume = row["volume"]

        # Volume spike detection: if minute volume > avg*multiplier, can be used as提示（不影响执行）
        # 持续触发买/卖直到该分钟不再有可触发的档位或被资金/仓位限制阻止
        triggered = True
        while triggered:
            triggered = False
            # BUY: 查找第一个被触发的买价（从低到高）
            for bp in buy_list:
                if low <= bp:
                    # 检查最大仓位限制（当前仓位价值/本金）
                    current_position_value = shares * bp
                    allowed_position_value = principal * max_position_pct
                    # 可买金额受单笔限额和仓位限额限制
                    max_allowed_by_amount = int((single_trade_amount / bp) // shares_per_lot) * shares_per_lot
                    # 还需考虑到未超过仓位限额
                    remaining_allowed_value = max(0, allowed_position_value - current_position_value)
                    max_allowed_by_position = int((remaining_allowed_value / bp) // shares_per_lot) * shares_per_lot
                    lots = min(max_allowed_by_amount, max_allowed_by_position)
                    if lots <= 0:
                        # 无法买入（仓位或现金限制）
                        continue
                    buy_shares = lots
                    buy_amount = buy_shares * bp
                    cost = calculate_trade_cost_simple(buy_amount, cfg, is_single_side=True)
                    # 执行
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
                    break  # 变动后重新检测

            # SELL: 查找可触发的卖价（从高到低优先）
            for sp in reversed(sell_list):
                if high >= sp and shares >= shares_per_lot:
                    max_lots_by_amount = int((single_trade_amount / sp) // shares_per_lot) * shares_per_lot
                    max_lots_by_shares = int(shares // shares_per_lot) * shares_per_lot
                    lots = min(max_lots_by_amount, max_lots_by_shares)
                    if lots <= 0:
                        continue
                    sell_shares = lots
                    sell_amount = sell_shares * sp
                    cost = calculate_trade_cost_simple(sell_amount, cfg, is_single_side=True)
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
                    break

        # 每分钟估值（用 close）
        holdings_value = shares * close
        net_value = cash + holdings_value
        timestamps.append(t)
        net_values.append(net_value)
        holdings_history.append(shares)

        # 止损/止盈检查（按净值相对初始本金）
        if stop_loss_pct is not None:
            if net_value <= initial_net * (1 - stop_loss_pct / 100.0):
                # 尽量在该分钟 close 价格全部清仓
                if shares >= shares_per_lot:
                    sell_shares = int(shares // shares_per_lot) * shares_per_lot
                    sell_amount = sell_shares * close
                    cost = calculate_trade_cost_simple(sell_amount, cfg, is_single_side=True)
                    shares -= sell_shares
                    cash += (sell_amount - cost)
                    trade_records.append({
                        "时间": t,
                        "类型": "止损卖出(清仓)",
                        "价格(港元)": close,
                        "股数": sell_shares,
                        "金额(港元)": round(sell_amount, 2),
                        "成本(港元)": round(cost, 2),
                        "剩余现金(港元)": round(cash, 2),
                        "持仓股数": shares
                    })
                # 停止进一步交易（保守处理）
                # 这里我们选择停止整个回测（不再挂网格交易），因为发生了止损
                break

        if take_profit_pct is not None:
            if net_value >= initial_net * (1 + take_profit_pct / 100.0):
                # 全部清仓锁定收益
                if shares >= shares_per_lot:
                    sell_shares = int(shares // shares_per_lot) * shares_per_lot
                    sell_amount = sell_shares * close
                    cost = calculate_trade_cost_simple(sell_amount, cfg, is_single_side=True)
                    shares -= sell_shares
                    cash += (sell_amount - cost)
                    trade_records.append({
                        "时间": t,
                        "类型": "止盈卖出(清仓)",
                        "价格(港元)": close,
                        "股数": sell_shares,
                        "金额(港元)": round(sell_amount, 2),
                        "成本(港元)": round(cost, 2),
                        "剩余现金(港元)": round(cash, 2),
                        "持仓股数": shares
                    })
                break

    final_total = net_values[-1] if net_values else (cash + shares * current_price)
    total_profit = final_total - principal
    profit_rate = (total_profit / principal) * 100 if principal != 0 else 0
    buys = [r for r in trade_records if r["类型"].startswith("买入")]
    sells = [r for r in trade_records if r["类型"].startswith("卖出") or r["类型"].startswith("止")]
    total_buy_count = len(buys)
    total_sell_count = len(sells)
    avg_trade_profit = (total_profit / (total_buy_count + total_sell_count)) if (total_buy_count + total_sell_count) > 0 else 0
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
    max_dd = drawdown.max() * 100
    return round(float(max_dd), 4)


# ---------------------------
# Default minute data generator
# ---------------------------
def generate_default_minute_data(current_price=27.5, interval=5):
    minute_data = []
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
# Streamlit UI
# ---------------------------

def render_sidebar():
    st.sidebar.header("1）资金与标的（新手说明见下方）")
    principal = st.sidebar.number_input("交易本金（港元）",
                                        min_value=1000.0, max_value=2_000_000.0,
                                        value=30_000.0, step=1000.0,
                                        help="说明：这是你用于本次策略研究/回测的总资金（只用于策略模拟，不影响其他钱）。新手建议少量开始，例如 10,000 - 50,000 港元。")
    etf_code = st.sidebar.text_input("ETF 代码（港股）", value="02800.HK",
                                     help="示例：02800.HK（恒生ETF）。请输入完整代码以便记录。")
    current_price = st.sidebar.number_input("当前价格（港元）", min_value=0.0001, value=27.5, format="%.4f",
                                            help="输入最新成交价（精确到小数）。若不清楚可先用最近市价作为近似。")

    st.sidebar.markdown("---")
    st.sidebar.header("2）手续费 / 滑点（新手说明）")
    st.sidebar.caption("说明：百分比输入均按“%”填写，例如输入 0.15 表示 0.15%。平台费是按每笔固定港元计。")
    platform_fee = st.sidebar.number_input("平台费（每笔，港元）", min_value=0.0, value=15.0, step=1.0,
                                          help="券商/平台固定每笔手续费（部分券商有），若不清楚保持默认 15.")
    trade_fee_pct = st.sidebar.number_input("交易佣金（%）", min_value=0.0, value=0.00565, step=0.00001,
                                            help="示例 0.00565 表示 0.00565%。许多券商会有最低收费，请在实盘时确认。")
    settlement_fee_pct = st.sidebar.number_input("交收费（%）", min_value=0.0, value=0.0042, step=0.00001)
    sfc_fee_pct = st.sidebar.number_input("证监会费（%）", min_value=0.0, value=0.0027, step=0.00001)
    frc_fee_pct = st.sidebar.number_input("FRC费（%）", min_value=0.0, value=0.00015, step=0.00001)

    st.sidebar.markdown("----")
    st.sidebar.subheader("滑点智能推荐（基于日均成交额）")
    avg_daily_turnover = st.sidebar.number_input("输入该ETF日均成交额（港元）",
                                                 min_value=0.0, value=500_000_000.0, step=1_000_000.0,
                                                 help="例如 500000000 表示 5 亿港元。若不清楚可先留默认值。")
    rec_low, rec_mid, rec_high = recommend_slippage_by_turnover(avg_daily_turnover)
    st.sidebar.caption(f"推荐滑点范围（经验值）：{rec_low:.3f}% 〜 {rec_high:.3f}%，建议值：{rec_mid:.3f}%")
    # 滑点输入框（支持一键应用推荐）
    slippage_pct = st.sidebar.number_input("滑点（%）", min_value=0.0, value=rec_mid, step=0.01,
                                          help="交易中隐藏的成本。若你做挂单可设小一些（例如0.05%），若扫单或流动性差设大一些。")
    if st.sidebar.button("应用推荐滑点"):
        slippage_pct = rec_mid
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("3）网格 & 回测配置（新手说明）")
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 10, 15], index=1,
                                         help="1分钟 = 高频模拟，5分钟 = 平衡；新手可选 5 分钟")
    grid_type = st.sidebar.radio("网格类型", ["动态间距（基于ATR）", "固定间距（手动）"])
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 10, 30, 16, 1,
                                   help="档数越多触发越频繁，但手续费/滑点更高。新手 12-18 左右比较稳。")
    fixed_spacing_pct = None
    if grid_type != "动态间距（基于ATR）":
        fixed_spacing_pct = st.sidebar.slider("固定间距（%）", 0.1, 2.0, 0.3, 0.05,
                                              help="每格间距（百分比），新手 0.2-0.5% 之间尝试。")

    st.sidebar.markdown("---")
    st.sidebar.header("4）风控（非常重要）")
    initial_cash_pct = st.sidebar.slider("初始可用现金占本金比（%）", 10, 100, 50, 5,
                                         help="日内不建议全部拿出来做，通常设置 30%-70% 作为可用现金。")
    single_trade_pct = st.sidebar.slider("单次交易金额占本金（%）", 0.5, 20.0, 5.0, 0.5,
                                         help="每次下单金额占本金比例，新手建议 1%-5% 之间。")
    shares_per_lot = st.sidebar.number_input("每手股数（港股通常 100）", min_value=1, value=100, step=1)
    max_position_pct = st.sidebar.slider("最大持仓占本金（%）", 10, 100, 50, 5,
                                         help="总仓位上限，避免被套满仓。")
    stop_loss_pct = st.sidebar.number_input("全局止损阈值（%），0 为不启用", min_value=0.0, value=0.0, step=0.1,
                                            help="例如输入 3 表示净值跌破本金 -3% 时全部清仓止损。")
    take_profit_pct = st.sidebar.number_input("全局止盈阈值（%），0 为不启用", min_value=0.0, value=0.0, step=0.1,
                                              help="例如输入 2 表示净值超过本金 +2% 时全部清仓获利了结。")

    cfg = {
        "platform_fee": float(platform_fee),
        "trade_fee_pct": float(trade_fee_pct),
        "settlement_fee_pct": float(settlement_fee_pct),
        "sfc_fee_pct": float(sfc_fee_pct),
        "frc_fee_pct": float(frc_fee_pct),
        "slippage_pct": float(slippage_pct),
        "initial_cash_pct": float(initial_cash_pct / 100.0),
        "single_trade_amount": float(principal * (single_trade_pct / 100.0)),
        "shares_per_lot": int(shares_per_lot),
        "max_position_pct": float(max_position_pct / 100.0),
        "stop_loss_pct": float(stop_loss_pct) if stop_loss_pct > 0 else None,
        "take_profit_pct": float(take_profit_pct) if take_profit_pct > 0 else None
    }

    st.sidebar.markdown("---")
    st.sidebar.caption("注：以上默认与建议均为经验值，实盘前务必用小仓位或模拟账户验证。")

    return principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover


def render_tab_data():
    st.subheader("分钟级数据（可编辑）")
    if "minute_data" not in st.session_state:
        st.session_state.minute_data = generate_default_minute_data()
    data_interval = st.session_state.get("data_interval", 5)
    st.write(f"当前数据周期：{data_interval} 分钟（默认生成，支持手动编辑）")
    table = []
    for d in st.session_state.minute_data:
        vol_str = f"{d['volume']}" if d['volume'] < 1000 else (f"{d['volume']/1000:.1f}k" if d['volume'] < 10000 else f"{d['volume']/10000:.2f}万")
        table.append({"时间": d["time"], "最高价(港元)": d["high"], "最低价(港元)": d["low"], "收盘价(港元)": d["close"], "成交量": vol_str})

    edited = st.data_editor(table, use_container_width=True, hide_index=True, key="minute_editor")
    if st.button("保存数据"):
        updated = []
        for idx, row in enumerate(edited):
            time_str = str(row["时间"]).strip()
            try:
                hour, minute = map(int, time_str.split(":"))
                valid = ((hour == 9 and minute >= 30) or (10 <= hour < 12) or (hour == 12 and minute == 0) or (13 <= hour < 16) or (hour == 16 and minute == 0))
                if not valid:
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
                    hi = max(high, low, close)
                    lo = min(high, low, close)
                    cl = max(min(close, hi), lo)
                    high, low, close = hi, lo, cl
                vol = parse_volume(row["成交量"])
                updated.append({"time": time_str, "high": round(high, 4), "low": round(low, 4), "close": round(close, 4), "volume": vol})
            except Exception as e:
                st.warning(f"第{idx+1}行数据解析失败：{e}")
        updated.sort(key=lambda x: datetime.strptime(x["time"], "%H:%M"))
        st.session_state.minute_data = updated
        st.success(f"保存 {len(updated)} 条分钟数据")

    if st.button("生成默认数据"):
        # use current price and data_interval if available
        current_price = st.session_state.get("current_price", 27.5)
        st.session_state.minute_data = generate_default_minute_data(current_price=current_price, interval=st.session_state.get("data_interval", 5))
        st.success("已生成默认分钟数据")


def render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover):
    st.subheader("网格参数与生成（含推荐）")
    minute_data = st.session_state.get("minute_data", generate_default_minute_data())
    st.session_state["data_interval"] = data_interval
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
    single_trade_amount = cfg["single_trade_amount"]
    round_trip_cost = calculate_trade_cost_simple(single_trade_amount, cfg, is_single_side=False)
    min_safe_spacing_pct = (round_trip_cost / single_trade_amount) * 100 * 1.2
    final_spacing_pct = max(base_spacing_pct, min_safe_spacing_pct, 0.2)

    if grid_type != "动态间距（基于ATR）":
        final_spacing_pct = fixed_spacing_pct if fixed_spacing_pct is not None else final_spacing_pct

    grid_upper = min(latest_upper * 1.005, current_price * 1.02)
    grid_lower = max(latest_lower * 0.995, current_price * 0.98)

    if grid_count % 2 != 0:
        grid_count += 1

    buy_grids, sell_grids = generate_intraday_grid_arithmetic(current_price, final_spacing_pct, grid_count, grid_upper, grid_lower)

    # VWAP & volume stats
    vwap = calculate_vwap(minute_data)
    avg_vol = np.mean([d["volume"] for d in minute_data]) if minute_data else 0
    recent_vol = minute_data[-1]["volume"] if minute_data else 0
    vol_spike = recent_vol > avg_vol * 2 if avg_vol > 0 else False

    st.session_state.update({
        "grid_params": {
            "grid_upper": round(grid_upper, 4), "grid_lower": round(grid_lower, 4),
            "spacing_pct": round(final_spacing_pct, 4), "grid_count": grid_count,
            "atr": round(latest_atr, 6), "round_trip_cost": round(round_trip_cost, 2),
            "single_trade_amount": round(single_trade_amount, 2)
        },
        "buy_grids": buy_grids,
        "sell_grids": sell_grids,
        "current_price": current_price
    })

    # Display summary
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 基本信息")
        st.write(f"交易标的：{etf_code}")
        st.write(f"本金：{principal:,.0f} 港元")
        st.write(f"当前价：{current_price:.4f} 港元")
        st.write(f"VWAP（样本）: {vwap if vwap else 'N/A'} 港元")
        st.write(f"平均分钟成交量: {avg_vol:.0f}，最近分钟成交量: {recent_vol} {'（放量）' if vol_spike else ''}")
    with col2:
        st.markdown("#### 网格参数")
        p = st.session_state["grid_params"]
        st.write(f"区间：{p['grid_lower']} ~ {p['grid_upper']} 港元")
        st.w
