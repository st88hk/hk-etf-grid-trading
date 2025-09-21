# app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from bisect import bisect_left

# ---------------------------
# 工具函数 / 成本计算
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
    number_part = s
    for u, mul in multipliers.items():
        if s.endswith(u):
            number_part = s[:-len(u)]
            try:
                num = float(number_part)
                return int(round(num * mul))
            except:
                return 0
    try:
        return int(round(float(number_part)))
    except:
        return 0


def recommend_slippage_by_turnover(avg_daily_turnover):
    """基于日均成交额给出滑点推荐（百分比，单位 %）"""
    if not avg_daily_turnover or avg_daily_turnover <= 0:
        return (0.05, 0.15, 0.3)
    if avg_daily_turnover >= 1_000_000_000:
        return (0.03, 0.06, 0.12)
    if avg_daily_turnover >= 500_000_000:
        return (0.05, 0.12, 0.2)
    if avg_daily_turnover >= 50_000_000:
        return (0.1, 0.25, 0.5)
    return (0.3, 0.7, 1.5)


def calculate_trade_cost_simple(amount, cfg, is_single_side=True):
    """计算交易成本：平台费（固定）+ 各项百分比费用 + 滑点（滑点以百分比形式存在 cfg['slippage_pct']）"""
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


# ---------------------------
# 数据获取：雅虎财经
# ---------------------------
def fetch_minute_data_yahoo(etf_code, interval="5m", period="1d"):
    """
    从雅虎财经获取分钟数据，并把时间转换为香港时区，且过滤为香港交易时段：
      上午 09:30 - 12:00，下午 13:00 - 16:00
    interval: '1m', '5m', '15m'
    period: '1d', '5d'
    返回：list of dict 每项包含 time, high, low, close, volume （time 为 "HH:MM" 香港时间）
    """
    try:
        data = yf.download(etf_code, interval=interval, period=period, progress=False)

        # 处理 multiindex（例如 yfinance 返回多列时）
        if isinstance(data.columns, pd.MultiIndex):
            # 尝试按第二层的 ticker 名称取列
            try:
                data = data.xs(etf_code, axis=1, level=1)
            except Exception:
                # 若失败，取第一个子列（回退）
                try:
                    data = data.xs(data.columns.levels[1][0], axis=1, level=1)
                except Exception:
                    pass

        if data is None or data.empty:
            return []

        # 把索引转换到香港时区（若无时区则先认为是 UTC）
        try:
            # 如果 tz 为空（tz_naive），先本地化为 UTC，再转换；否则直接转换
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert('Asia/Hong_Kong')
            else:
                data.index = data.index.tz_convert('Asia/Hong_Kong')
        except Exception:
            # 如果上面失败（极少数情况），试着用 pandas to_datetime 再转换
            try:
                data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert('Asia/Hong_Kong')
            except Exception:
                # 最后退回不变
                pass

        # 过滤出香港交易的两段时间：09:30-12:00 与 13:00-16:00
        try:
            idx_times = data.index.time  # ndarray of datetime.time
            mask_morning = (idx_times >= dtime(9, 30)) & (idx_times <= dtime(12, 0))
            mask_afternoon = (idx_times >= dtime(13, 0)) & (idx_times <= dtime(16, 0))
            mask = mask_morning | mask_afternoon
            data = data[mask]
        except Exception:
            # 若过滤失败则不做过滤（保证健壮性）
            pass

        minute_data = []
        for idx, row in data.iterrows():
            try:
                t = pd.to_datetime(idx).strftime("%H:%M")
            except:
                t = str(idx)
            # 有时候行可能包含 numpy 类型，强制转 float 再读
            try:
                high_v = float(row["High"])
                low_v = float(row["Low"])
                close_v = float(row["Close"])
                vol_v = int(row["Volume"]) if not np.isnan(row["Volume"]) else 0
            except Exception:
                # 某些情况下列名可能是小写，尝试小写列名
                r = {k.lower(): v for k, v in dict(row).items()}
                high_v = float(r.get("high", np.nan))
                low_v = float(r.get("low", np.nan))
                close_v = float(r.get("close", np.nan))
                vol_v = int(r.get("volume", 0) if not np.isnan(r.get("volume", 0)) else 0)
            minute_data.append({
                "time": t,
                "high": round(high_v, 4),
                "low": round(low_v, 4),
                "close": round(close_v, 4),
                "volume": int(vol_v)
            })
        return minute_data
    except Exception as e:
        st.error(f"从雅虎财经获取数据失败: {e}")
        return []



# ---------------------------
# 指标函数
# ---------------------------

def calculate_atr(highs, lows, closes, period=14):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    if len(closes) == 0:
        return []
    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().round(6).tolist()
    return atr


def calculate_vwap(minute_data):
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
# 网格生成（等距算术）
# ---------------------------

def generate_intraday_grid_arithmetic(current_price, spacing_pct, grid_count, grid_upper, grid_lower):
    """等距网格（算术）：返回 (buy_list_asc, sell_list_asc)"""
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
# 回测：改进版
# ---------------------------

def calculate_max_drawdown_from_series(net_values):
    if not net_values:
        return 0.0
    series = pd.Series(net_values)
    running_max = series.cummax()
    drawdown = (running_max - series) / running_max
    max_dd = drawdown.max() * 100
    return round(float(max_dd), 4)


def backtest_intraday_strategy_improved(principal, current_price, buy_grids, sell_grids, minute_data, cfg):
    """
    回测说明（要点）：
    - 使用当分钟的 high/low 判定触发，close 用于估值
    - 支持单笔金额限制、每手股数、最大持仓比例、全局止损/止盈
    - 每次触发按整手交易
    """
    trade_records = []
    cash = principal * cfg.get("initial_cash_pct", 0.5)
    shares = 0
    shares_per_lot = cfg.get("shares_per_lot", 100)
    single_trade_amount = cfg.get("single_trade_amount", principal * 0.05)

    buy_list = sorted(buy_grids)   # 升序
    sell_list = sorted(sell_grids) # 升序

    net_values = []
    timestamps = []
    holdings_history = []

    stop_loss_pct = cfg.get("stop_loss_pct", None)
    take_profit_pct = cfg.get("take_profit_pct", None)
    max_position_pct = cfg.get("max_position_pct", 0.5)

    initial_net = principal

    for row in minute_data:
        t = row["time"]
        high = row["high"]
        low = row["low"]
        close = row["close"]

        # 可以在同一分钟多次触发不同档位（只要满足条件且资金/仓位允许）
        triggered = True
        while triggered:
            triggered = False
            # BUY: 从低向高遍历买档（优先低档）
            for bp in buy_list:
                if low <= bp:
                    current_position_value = shares * bp
                    allowed_position_value = principal * max_position_pct
                    remaining_allowed_value = max(0, allowed_position_value - current_position_value)
                    max_allowed_by_position = int((remaining_allowed_value / bp) // shares_per_lot) * shares_per_lot
                    max_allowed_by_amount = int((single_trade_amount / bp) // shares_per_lot) * shares_per_lot
                    lots = min(max_allowed_by_amount, max_allowed_by_position)
                    if lots <= 0:
                        continue
                    buy_shares = lots
                    buy_amount = buy_shares * bp
                    cost = calculate_trade_cost_simple(buy_amount, cfg, is_single_side=True)
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
                    break

            # SELL: 从高向低遍历卖档（优先高档）
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

        # 每分钟估值
        holdings_value = shares * close
        net_value = cash + holdings_value
        timestamps.append(t)
        net_values.append(net_value)
        holdings_history.append(shares)

        # 全局止损/止盈检查（按净值）
        if stop_loss_pct is not None:
            if net_value <= initial_net * (1 - stop_loss_pct / 100.0):
                # 尽量以该分钟 close 清仓
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
                break  # 停止回测

        if take_profit_pct is not None:
            if net_value >= initial_net * (1 + take_profit_pct / 100.0):
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


# ---------------------------
# 默认分钟数据生成（用于演示）
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
    # 上午盘
    for t in create_range("09:30", "12:00"):
        price_offset = np.random.uniform(-0.003, 0.003)
        close_price = current_price * (1 + price_offset)
        high = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low = close_price * (1 - np.random.uniform(0.0005, 0.001))
        volume = int(np.random.uniform(8000, 25000))
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high, 4), "low": round(low, 4), "close": round(close_price, 4), "volume": volume})
    # 午后盘
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
    st.sidebar.header("1）资金与标的（新手说明在主页面）")
    principal = st.sidebar.number_input("交易本金（港元）",
                                        min_value=1000.0, max_value=5_000_000.0,
                                        value=100000.0, step=1000.0,
                                        help="这是本次用于回测/策略验证的资金规模（不影响你真实账户）。")
    etf_code = st.sidebar.text_input("ETF 代码（雅虎代码）", value="02800.HK",
                                     help="示例：02800.HK。如果要拉美股/美股，格式通常是 'SPY' 或 'AAPL'。")
    current_price = st.sidebar.number_input("当前价格（港元）", min_value=0.0001, value=27.5, format="%.4f",
                                            help="若从雅虎财经拉取数据，此项会自动更新为当日最后价；也可手动输入。")

    st.sidebar.markdown("---")
    st.sidebar.header("2）手续费 / 滑点（新手说明）")
    st.sidebar.caption("百分比以 % 计，例如输入 0.005 表示 0.005%。平台费按每笔固定港元计（若券商无此项填写 0）。")
    platform_fee = st.sidebar.number_input("平台费（每笔，港元）", min_value=0.0, value=15.0, step=1.0)
    trade_fee_pct = st.sidebar.number_input("交易佣金（%）", min_value=0.0, value=0.00565, step=0.00001)
    settlement_fee_pct = st.sidebar.number_input("交收费（%）", min_value=0.0, value=0.0042, step=0.00001)
    sfc_fee_pct = st.sidebar.number_input("证监会费（%）", min_value=0.0, value=0.0027, step=0.00001)
    frc_fee_pct = st.sidebar.number_input("FRC费（%）", min_value=0.0, value=0.00015, step=0.00001)

    st.sidebar.markdown("----")
    st.sidebar.subheader("滑点智能推荐（基于日均成交额）")
    avg_daily_turnover = st.sidebar.number_input("该ETF日均成交额（港元）",
                                                 min_value=0.0, value=500_000_000.0, step=1_000_000.0,
                                                 help="例如 500000000 表示 5 亿港元。若不清楚留默认值。")
    rec_low, rec_mid, rec_high = recommend_slippage_by_turnover(avg_daily_turnover)
    st.sidebar.caption(f"经验推荐滑点范围：{rec_low:.3f}% 〜 {rec_high:.3f}%，建议值：{rec_mid:.3f}%")
    slippage_pct = st.sidebar.number_input("滑点（%）", min_value=0.0, value=rec_mid, step=0.01,
                                          help="交易中隐藏的成本，市值、流动性差的品种要设大一些。")

    if st.sidebar.button("应用推荐滑点"):
        slippage_pct = rec_mid
        # We cannot directly mutate outer scope variable; we'll rely on returning cfg

    st.sidebar.markdown("---")
    st.sidebar.header("3）网格 & 回测配置（新手说明）")
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 15], index=1,
                                         help="1 分钟模拟更精细但数据量大；5 分钟为折中选择；15 分钟更平滑。")
    grid_type = st.sidebar.radio("网格间距方式", ["动态间距（基于ATR）", "固定间距（手动）"])
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 10, 30, 16, 1,
                                   help="档数越多触发越频繁，但手续费/滑点更高。")
    fixed_spacing_pct = None
    if grid_type != "动态间距（基于ATR）":
        fixed_spacing_pct = st.sidebar.slider("固定间距（%）", 0.1, 2.0, 0.3, 0.05,
                                              help="每格间距（百分比），新手 0.2-0.5% 之间尝试。")

    st.sidebar.markdown("---")
    st.sidebar.header("4）风控（强烈建议）")
    initial_cash_pct = st.sidebar.slider("初始可用现金占本金比（%）", 10, 100, 50, 5,
                                         help="日内不建议全部拿出来做，通常设置 30%-70% 作为可用现金。")
    single_trade_pct = st.sidebar.slider("单次交易金额占本金（%）", 0.5, 20.0, 5.0, 0.5,
                                         help="每次下单金额占本金比例，新手建议 1%-5%。")
    shares_per_lot = st.sidebar.number_input("每手股数（港股通常 100）", min_value=1, value=100, step=1)
    max_position_pct = st.sidebar.slider("最大持仓占本金（%）", 10, 100, 50, 5,
                                         help="总仓位上限，避免被套满仓。")
    stop_loss_pct = st.sidebar.number_input("全局止损阈值（%），0 为不启用", min_value=0.0, value=0.0, step=0.1)
    take_profit_pct = st.sidebar.number_input("全局止盈阈值（%），0 为不启用", min_value=0.0, value=0.0, step=0.1)

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
    st.subheader("分钟级数据（可自动拉取 / 编辑 / 生成模拟）")

    if "minute_data" not in st.session_state:
        st.session_state.minute_data = []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("数据来源：雅虎财经（yfinance）")
    with col2:
        if st.button("从雅虎财经获取当天分钟数据"):
            etf_code = st.session_state.get("etf_code", "02800.HK")
            data_interval = st.session_state.get("data_interval", 5)
            interval_map = {1: "1m", 5: "5m", 15: "15m"}
            with st.spinner(f"从雅虎财经下载 {etf_code} {interval_map.get(data_interval, '5m')} 数据..."):
                md = fetch_minute_data_yahoo(etf_code, interval=interval_map.get(data_interval, "5m"), period="1d")
            if md:
                st.session_state.minute_data = md
                # 更新 current_price 为最后一条的 close 以便网格计算
                st.session_state.current_price = md[-1]["close"]
                st.success(f"已获取 {len(md)} 条分钟数据，当前价已更新为 {md[-1]['close']}")
            else:
                st.warning("未获取到数据，请确认代码或稍后重试。")
    with col3:
        if st.button("生成示例/模拟数据"):
            cp = st.session_state.get("current_price", 27.5)
            di = st.session_state.get("data_interval", 5)
            st.session_state.minute_data = generate_default_minute_data(current_price=cp, interval=di)
            st.success("已生成模拟分钟数据")

    if not st.session_state.minute_data:
        st.info("当前无分钟数据，建议使用上方按钮从雅虎财经获取或生成模拟数据。")
        st.session_state.minute_data = generate_default_minute_data()

    # 展示并可编辑（使用 data_editor）
    table = []
    for d in st.session_state.minute_data:
        vol_str = f"{d['volume']}" if d['volume'] < 1000 else (f"{d['volume']/1000:.1f}k" if d['volume'] < 10000 else f"{d['volume']/10000:.2f}万")
        table.append({"时间": d["time"], "最高价(港元)": d["high"], "最低价(港元)": d["low"], "收盘价(港元)": d["close"], "成交量": vol_str})

    edited = st.data_editor(pd.DataFrame(table), use_container_width=True, num_rows="dynamic", key="minute_editor")

    if st.button("保存编辑"):
        updated = []
        for idx, row in edited.iterrows():
            time_str = str(row["时间"]).strip()
            try:
                high = float(row["最高价(港元)"])
                low = float(row["最低价(港元)"])
                close = float(row["收盘价(港元)"])
                vol = parse_volume(row["成交量"])
                updated.append({"time": time_str, "high": round(high,4), "low": round(low,4), "close": round(close,4), "volume": int(vol)})
            except Exception as e:
                st.warning(f"第 {idx+1} 行解析失败，已跳过：{e}")
        if updated:
            # sort by time
            try:
                updated.sort(key=lambda x: datetime.strptime(x["time"], "%H:%M"))
            except:
                pass
            st.session_state.minute_data = updated
            st.success(f"已保存 {len(updated)} 条分钟数据")
        else:
            st.warning("无有效数据保存。")

    st.caption("提示：时间格式需为 HH:MM，例如 09:30。成交量支持 'k','万' 等单位输入。")


def render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover):
    st.subheader("网格参数与生成（含推荐）")
    minute_data = st.session_state.get("minute_data", generate_default_minute_data())

    # 保存上下文
    st.session_state["cfg"] = cfg
    st.session_state["etf_code"] = etf_code
    st.session_state["data_interval"] = data_interval
    st.session_state["current_price"] = st.session_state.get("current_price", current_price)

    highs = [d["high"] for d in minute_data] if minute_data else []
    lows = [d["low"] for d in minute_data] if minute_data else []
    closes = [d["close"] for d in minute_data] if minute_data else []

    atr = calculate_atr(highs, lows, closes, period=14) if closes else [0.0]
    latest_atr = atr[-1] if atr else 0.01

    b_up, b_mid, b_low = calculate_narrow_bollinger(closes, period=10, num_std=1.5) if closes else ([], [], [])
    latest_upper = b_up[-1] if b_up else st.session_state.get("current_price", current_price) * 1.01
    latest_lower = b_low[-1] if b_low else st.session_state.get("current_price", current_price) * 0.99

    # 动态 spacing 计算
    cp = st.session_state.get("current_price", current_price) or current_price
    base_spacing_pct = (latest_atr * 0.6 / cp) * 100 if cp > 0 else 0.3
    single_trade_amount = cfg["single_trade_amount"]
    round_trip_cost = calculate_trade_cost_simple(single_trade_amount, cfg, is_single_side=False)
    min_safe_spacing_pct = (round_trip_cost / single_trade_amount) * 100 * 1.2 if single_trade_amount > 0 else 0.2
    final_spacing_pct = max(base_spacing_pct, min_safe_spacing_pct, 0.2)

    if grid_type != "动态间距（基于ATR）" and fixed_spacing_pct is not None:
        final_spacing_pct = fixed_spacing_pct

    grid_upper = min(latest_upper * 1.005, cp * 1.05)
    grid_lower = max(latest_lower * 0.995, cp * 0.95)

    if grid_count % 2 != 0:
        grid_count += 1

    buy_grids, sell_grids = generate_intraday_grid_arithmetic(cp, final_spacing_pct, grid_count, grid_upper, grid_lower)

    st.session_state.update({
        "grid_params": {
            "grid_upper": round(grid_upper, 4),
            "grid_lower": round(grid_lower, 4),
            "spacing_pct": round(final_spacing_pct, 4),
            "grid_count": grid_count,
            "atr": round(latest_atr, 6),
            "round_trip_cost": round(round_trip_cost, 2),
            "single_trade_amount": round(single_trade_amount, 2)
        },
        "buy_grids": buy_grids,
        "sell_grids": sell_grids
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 基本信息")
        st.write(f"交易标的：{etf_code}")
        st.write(f"本金：{principal:,.0f} 港元")
        st.write(f"当前价（用于生成网格）：{cp:.4f} 港元")
        vwap = calculate_vwap(minute_data)
        st.write(f"VWAP（样本）：{vwap if vwap else 'N/A'}")
    with col2:
        st.markdown("#### 网格参数")
        p = st.session_state["grid_params"]
        st.write(f"区间：{p['grid_lower']} ~ {p['grid_upper']} 港元")
        st.write(f"间距：{p['spacing_pct']} %")
        st.write(f"档数：{p['grid_count']} (买{len(buy_grids)} / 卖{len(sell_grids)})")
        st.write(f"单次交易额：{p['single_trade_amount']:.2f} 港元")
        st.write(f"估算双边成本（round-trip）：{p['round_trip_cost']:.2f} 港元")

    st.markdown("##### 买入网格（低->高）")
    if buy_grids:
        st.dataframe(pd.DataFrame({"买入档位": [f"买{i+1}" for i in range(len(buy_grids))], "价格(港元)": buy_grids}), use_container_width=True)
    else:
        st.warning("未生成买入网格")

    st.markdown("##### 卖出网格（低->高）")
    if sell_grids:
        st.dataframe(pd.DataFrame({"卖出档位": [f"卖{i+1}" for i in range(len(sell_grids))], "价格(港元)": sell_grids}), use_container_width=True)
    else:
        st.warning("未生成卖出网格")

    if st.button("开始回测"):
        with st.spinner("回测进行中..."):
            result = backtest_intraday_strategy_improved(
                principal=principal,
                current_price=cp,
                buy_grids=buy_grids.copy(),
                sell_grids=sell_grids.copy(),
                minute_data=minute_data,
                cfg=cfg
            )
            st.session_state.backtest_result = result
            st.success("回测完成，请切换到 回测结果 页查看")


def render_tab_backtest(principal, etf_code):
    st.subheader("回测结果与建议")
    result = st.session_state.get("backtest_result")
    if not result:
        st.info("请先在 网格策略 页生成网格并运行回测")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初始本金(港元)", f"{principal:,.0f}")
    col2.metric("最终市值(港元)", f"{result['final_total_value']:,.2f}")
    col3.metric("总收益(港元)", f"{result['total_profit']:,.2f}", delta=f"{result['profit_rate']:.4f}%")
    col4.metric("最大回撤(%)", f"{result['max_drawdown']:.4f}%")

    st.divider()
    st.markdown("### 净值曲线（按分钟）")
    df_nv = pd.DataFrame({"时间": result["timestamps"], "净值": result["net_values"], "持仓": result["holdings_history"]})
    try:
        df_nv.index = pd.to_datetime(df_nv["时间"], format="%H:%M")
    except:
        pass
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
    # 轻量间距扫描建议
    if "grid_params" in st.session_state:
        cur_spacing = st.session_state["grid_params"]["spacing_pct"]
        st.markdown("### 网格间距小范围优化建议（轻量扫描）")
        scan_values = [round(cur_spacing * (1 + delta), 4) for delta in [-0.3, -0.15, 0, 0.15, 0.3]]
        quick_results = []
        minute_data = st.session_state.get("minute_data", generate_default_minute_data())
        grid_count = st.session_state["grid_params"]["grid_count"]
        grid_upper = st.session_state["grid_params"]["grid_upper"]
        grid_lower = st.session_state["grid_params"]["grid_lower"]
        cfg = st.session_state.get("cfg", {})
        cp = st.session_state.get("current_price", 27.5)
        for sp in scan_values:
            buy_g, sell_g = generate_intraday_grid_arithmetic(cp, sp, grid_count, grid_upper, grid_lower)
            res = backtest_intraday_strategy_improved(principal, cp, buy_g.copy(), sell_g.copy(), minute_data, cfg=cfg)
            quick_results.append((sp, res["total_profit"], res["profit_rate"], res["max_drawdown"]))
        qr_df = pd.DataFrame(quick_results, columns=["spacing_pct(%)", "total_profit", "profit_rate(%)", "max_drawdown(%)"])
        st.dataframe(qr_df, use_container_width=True)
        best = qr_df.sort_values(by="total_profit", ascending=False).iloc[0]
        st.markdown(f"建议：在扫描范围内，**最佳间距 = {best['spacing_pct(%)']:.4f}%**，对应总收益 {best['total_profit']:.2f} 港元（仅供参考）")

    st.divider()
    st.markdown("### 实盘建议（新手必读）")
    st.write("""
    1. 从模拟盘或小仓位开始验证回测结论；  
    2. 控制单次风险（建议 1%〜5% 本金）；  
    3. 注意成交量和 VWAP，放量时策略表现会变化；  
    4. 回测对手续费/滑点敏感，实盘中先设保守滑点；  
    5. 保存每次回测参数并复盘。  
    """)
    st.caption("免责声明：本工具仅用于策略研究与回测，不构成投资建议。实盘需做好风控并对接正规券商 API。")


def render_tab_help():
    st.subheader("新手指南与参数说明")
    st.markdown("""
    **快捷操作**
    - 在“分钟数据”页，点击“从雅虎财经获取当天分钟数据”可自动拉取（需网络）。
    - 若不确定参数，可先使用右侧侧边栏的默认值并生成模拟数据体验流程。

    **关键参数说明**
    - 交易本金：用于回测的资金总额（模拟）
    - ETF 代码：雅虎财经代码，例如 `02800.HK`
    - 平台费：每笔固定费用（港元）
    - 交易佣金 / 交收费 / 证监会费 / FRC费：按成交额百分比计
    - 滑点：以百分比表示（推荐依据日均成交额给出）
    - 网格间距：每格价格间隔（%），可为动态（ATR）或固定
    - 单次交易金额：每次下单的目标金额（侧边栏设定为 %）
    - 每手股数：港股通常为 100 股整手交易
    - 最大持仓占比：整个策略中允许的最大仓位（%）
    - 止损 / 止盈：按净值触发全部清仓（%）
    """)


def main():
    st.set_page_config(page_title="香港日内网格 T+0（新版：雅虎财经支持）", layout="wide")
    st.title("香港日内 T+0 网格交易工具（含雅虎财经分钟数据拉取 & 新手说明）")

    # Sidebar inputs
    principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover = render_sidebar()
    # 保存关键上下文
    st.session_state["cfg"] = cfg
    st.session_state["etf_code"] = etf_code
    st.session_state["data_interval"] = data_interval
    st.session_state["current_price"] = st.session_state.get("current_price", current_price)

    tabs = st.tabs(["分钟数据", "网格策略", "回测结果", "新手说明"])
    with tabs[0]:
        render_tab_data()
    with tabs[1]:
        render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover)
    with tabs[2]:
        render_tab_backtest(principal, etf_code)
    with tabs[3]:
        render_tab_help()

    st.caption("提示：把鼠标移到输入框的 '?' 上可以看到简短说明；若遇到报错或不明白的参数，可以把报错贴到聊天里，我会帮你查。")

if __name__ == "__main__":
    main()
