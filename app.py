# app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytz

# ---------------------------
# Utilities
# ---------------------------

def parse_volume(volume_input):
    """Parse volume like '12k' or '3.5万' to integer."""
    if volume_input is None:
        return 0
    s = str(volume_input).strip().lower().replace(",", "").replace(" ", "")
    if s == "":
        return 0
    multipliers = {'k': 1_000, 'm': 1_000_000, 'w': 10_000, '万': 10_000, '亿': 100_000_000}
    for suffix, mul in multipliers.items():
        if s.endswith(suffix):
            try:
                return int(round(float(s[:-len(suffix)]) * mul))
            except:
                return 0
    try:
        return int(round(float(s)))
    except:
        return 0

def recommend_slippage_by_turnover(avg_daily_turnover):
    """Return (low, mid, high) sliding % recommendations."""
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
    """Estimate cost: fixed platform fee + percentage fees + slippage (percentage)."""
    slippage_cost = amount * (cfg["slippage_pct"] / 100.0)
    trade_fee = amount * (cfg["trade_fee_pct"] / 100.0)
    settlement_fee = amount * (cfg["settlement_fee_pct"] / 100.0)
    sfc_fee = amount * (cfg["sfc_fee_pct"] / 100.0)
    frc_fee = amount * (cfg["frc_fee_pct"] / 100.0)
    platform_fee = cfg.get("platform_fee", 0.0)
    single_total = platform_fee + trade_fee + settlement_fee + sfc_fee + frc_fee + slippage_cost
    return round(single_total if is_single_side else single_total * 2, 2)

def get_avg_turnover(ticker, days=20):
    """Get average daily turnover (Close * Volume) for past `days` trading days."""
    try:
        data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if data is None or data.empty:
            return None
        avg_turnover = (data["Close"] * data["Volume"]).mean()
        return float(avg_turnover)
    except Exception as e:
        st.warning(f"获取日均成交额失败：{e}")
        return None

# 交易时间提醒相关
def get_hk_trading_status():
    """返回香港市场当前交易状态"""
    now = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    today = now.date()
    is_weekday = now.weekday() < 5  # 0-4为周一至周五
    
    morning_start = datetime.combine(today, dtime(9, 30))
    morning_end = datetime.combine(today, dtime(12, 0))
    afternoon_start = datetime.combine(today, dtime(13, 0))
    afternoon_end = datetime.combine(today, dtime(16, 0))
    
    now_dt = datetime.combine(today, now.time())
    
    if not is_weekday:
        return {"status": "休市", "message": "今日非交易日", "next_open": "下周一 09:30"}
    
    if morning_start <= now_dt < morning_end:
        remaining = (morning_end - now_dt).total_seconds() // 60
        return {"status": "交易中", "message": f"上午场剩余 {remaining} 分钟", "next_open": None}
    elif afternoon_start <= now_dt < afternoon_end:
        remaining = (afternoon_end - now_dt).total_seconds() // 60
        return {"status": "交易中", "message": f"下午场剩余 {remaining} 分钟", "next_open": None}
    elif now_dt < morning_start:
        wait = (morning_start - now_dt).total_seconds() // 60
        return {"status": "未开盘", "message": f"距离上午开盘还有 {wait} 分钟", "next_open": "09:30"}
    elif now_dt < afternoon_start:
        wait = (afternoon_start - now_dt).total_seconds() // 60
        return {"status": "午间休市", "message": f"距离下午开盘还有 {wait} 分钟", "next_open": "13:00"}
    else:
        return {"status": "已收盘", "message": "今日交易已结束", "next_open": "次日 09:30"}

# 网格参数敏感性分析
def analyze_grid_sensitivity(principal, current_price, minute_data, cfg, base_params):
    """分析网格参数变化对结果的影响"""
    results = []
    # 测试不同网格数量
    for grid_count in [10, 16, 22, 28]:
        # 测试不同间距（固定模式下）
        for spacing in [base_params['spacing'] * 0.7, base_params['spacing'], base_params['spacing'] * 1.3]:
            buy_grids, sell_grids = generate_intraday_grid_arithmetic(
                current_price, spacing, grid_count, 
                base_params['upper'], base_params['lower']
            )
            backtest_res = backtest_intraday_strategy_improved(
                principal, current_price, buy_grids, sell_grids, minute_data, cfg
            )
            results.append({
                "网格数量": grid_count,
                "间距(%)": round(spacing, 3),
                "收益(%)": backtest_res['profit_rate'],
                "交易次数": backtest_res['total_buy_count'] + backtest_res['total_sell_count'],
                "最大回撤(%)": backtest_res['max_drawdown']
            })
    return pd.DataFrame(results)

# 多ETF对比
def compare_etfs(etf_codes, principal, data_interval, cfg):
    """对比多个ETF的日内T+0效果"""
    comparison = []
    imap = {1:"1m",5:"5m",15:"15m"}
    interval = imap.get(data_interval, "5m")
    
    for code in etf_codes:
        with st.spinner(f"正在分析 {code}..."):
            minute_data = fetch_minute_data_yahoo(code, interval=interval, period="1d")
            if not minute_data:
                st.warning(f"{code} 获取数据失败，跳过")
                continue
            
            current_price = minute_data[-1]['close']
            # 使用默认网格参数
            buy_grids, sell_grids = generate_intraday_grid_arithmetic(
                current_price, 0.3, 16, current_price*1.05, current_price*0.95
            )
            res = backtest_intraday_strategy_improved(
                principal, current_price, buy_grids, sell_grids, minute_data, cfg
            )
            
            comparison.append({
                "ETF代码": code,
                "当前价格": current_price,
                "收益(%)": res['profit_rate'],
                "交易次数": res['total_buy_count'] + res['total_sell_count'],
                "最大回撤(%)": res['max_drawdown'],
                "最终净值": res['final_total_value']
            })
    return pd.DataFrame(comparison)

# ---------------------------
# Yahoo Finance minute fetch (HK timezone + filter trading hours)
# ---------------------------

def fetch_minute_data_yahoo(etf_code, interval="5m", period="1d"):
    """
    Return list of dicts: [{'time':'HH:MM','high':..., 'low':..., 'close':..., 'volume':...}, ...]
    Time converted to Asia/Hong_Kong and filtered to trading hours 09:30-12:00 and 13:00-16:00.
    """
    try:
        data = yf.download(etf_code, interval=interval, period=period, progress=False)
        if data is None or data.empty:
            return []
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(etf_code, axis=1, level=1)
            except Exception:
                try:
                    data = data.xs(data.columns.levels[1][0], axis=1, level=1)
                except Exception:
                    pass
        # tz handling
        try:
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert('Asia/Hong_Kong')
            else:
                data.index = data.index.tz_convert('Asia/Hong_Kong')
        except Exception:
            try:
                data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert('Asia/Hong_Kong')
            except Exception:
                pass
        # filter HK trading hours
        try:
            idx_times = data.index.time
            mask_morning = (idx_times >= dtime(9, 30)) & (idx_times <= dtime(12, 0))
            mask_afternoon = (idx_times >= dtime(13, 0)) & (idx_times <= dtime(16, 0))
            mask = mask_morning | mask_afternoon
            data = data[mask]
        except Exception:
            pass
        minute_data = []
        for idx, row in data.iterrows():
            try:
                t = pd.to_datetime(idx).strftime("%H:%M")
            except:
                t = str(idx)
            try:
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])
                v = int(row["Volume"]) if not np.isnan(row["Volume"]) else 0
            except Exception:
                r = {k.lower(): v for k, v in dict(row).items()}
                h = float(r.get("high", np.nan))
                l = float(r.get("low", np.nan))
                c = float(r.get("close", np.nan))
                v = int(r.get("volume", 0) if not np.isnan(r.get("volume", 0)) else 0)
            minute_data.append({"time": t, "high": round(h, 6), "low": round(l, 6), "close": round(c, 6), "volume": int(v)})
        return minute_data
    except Exception as e:
        st.error(f"从雅虎财经获取数据失败: {e}")
        return []

# ---------------------------
# Indicators and grid/backtest logic
# ---------------------------

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) == 0:
        return []
    highs = np.array(highs); lows = np.array(lows); closes = np.array(closes)
    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().round(6).tolist()
    return atr

def calculate_vwap(minute_data):
    prices = np.array([d["close"] for d in minute_data], dtype=float)
    volumes = np.array([d["volume"] for d in minute_data], dtype=float)
    if volumes.sum() == 0:
        return None
    return round(float((prices * volumes).sum() / volumes.sum()), 6)

def generate_intraday_grid_arithmetic(current_price, spacing_pct, grid_count, grid_upper, grid_lower):
    spacing = spacing_pct / 100.0
    half = grid_count // 2
    buy = [round(current_price * (1 - spacing * (i + 1)), 6) for i in range(half)]
    sell = [round(current_price * (1 + spacing * (i + 1)), 6) for i in range(half)]
    buy = [p for p in buy if p >= grid_lower]
    sell = [p for p in sell if p <= grid_upper]
    buy.sort(); sell.sort()
    return buy, sell

def calculate_max_drawdown_from_series(net_values):
    if not net_values:
        return 0.0
    s = pd.Series(net_values)
    rm = s.cummax()
    dd = (rm - s) / rm
    return round(float(dd.max() * 100), 4)

def backtest_intraday_strategy_improved(principal, current_price, buy_grids, sell_grids, minute_data, cfg):
    # improved backtest logic
    trade_records = []
    cash = principal * cfg.get("initial_cash_pct", 0.5)
    shares = 0
    shares_per_lot = cfg.get("shares_per_lot", 100)
    single_trade_amount = cfg.get("single_trade_amount", principal * 0.05)

    buy_list = sorted(buy_grids)
    sell_list = sorted(sell_grids)

    net_values = []
    timestamps = []
    holdings_history = []

    stop_loss_pct = cfg.get("stop_loss_pct", None)
    take_profit_pct = cfg.get("take_profit_pct", None)
    max_position_pct = cfg.get("max_position_pct", 0.5)

    initial_net = principal

    for row in minute_data:
        t = row["time"]
        high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
        triggered = True
        while triggered:
            triggered = False
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
                        "time": t, "type": "buy", "price": bp, "shares": buy_shares,
                        "amount": round(buy_amount, 2), "cost": round(cost, 2),
                        "cash_after": round(cash, 2), "holding_after": shares
                    })
                    buy_list.remove(bp)
                    triggered = True
                    break
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
                        "time": t, "type": "sell", "price": sp, "shares": sell_shares,
                        "amount": round(sell_amount, 2), "cost": round(cost, 2),
                        "cash_after": round(cash, 2), "holding_after": shares
                    })
                    sell_list.remove(sp)
                    triggered = True
                    break
        holdings_value = shares * close
        net_value = cash + holdings_value
        timestamps.append(t)
        net_values.append(round(net_value, 4))
        holdings_history.append(shares)
        if stop_loss_pct is not None and net_value <= initial_net * (1 - stop_loss_pct / 100.0):
            if shares >= shares_per_lot:
                sell_shares = int(shares // shares_per_lot) * shares_per_lot
                sell_amount = sell_shares * close
                cost = calculate_trade_cost_simple(sell_amount, cfg, is_single_side=True)
                shares -= sell_shares
                cash += (sell_amount - cost)
                trade_records.append({
                    "time": t, "type": "stoploss_sell", "price": close, "shares": sell_shares,
                    "amount": round(sell_amount, 2), "cost": round(cost, 2),
                    "cash_after": round(cash, 2), "holding_after": shares
                })
            break
        if take_profit_pct is not None and net_value >= initial_net * (1 + take_profit_pct / 100.0):
            if shares >= shares_per_lot:
                sell_shares = int(shares // shares_per_lot) * shares_per_lot
                sell_amount = sell_shares * close
                cost = calculate_trade_cost_simple(sell_amount, cfg, is_single_side=True)
                shares -= sell_shares
                cash += (sell_amount - cost)
                trade_records.append({
                    "time": t, "type": "takeprofit_sell", "price": close, "shares": sell_shares,
                    "amount": round(sell_amount, 2), "cost": round(cost, 2),
                    "cash_after": round(cash, 2), "holding_after": shares
                })
            break

    final_total = net_values[-1] if net_values else (cash + shares * current_price)
    total_profit = final_total - principal
    profit_rate = (total_profit / principal) * 100 if principal != 0 else 0
    buys = [r for r in trade_records if r["type"] == "buy"]
    sells = [r for r in trade_records if r["type"].startswith("sell")]
    total_buy_count = len(buys); total_sell_count = len(sells)
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
# Default data generator
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
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high,6), "low": round(low,6), "close": round(close_price,6), "volume": volume})
    for t in create_range("13:00", "16:00"):
        price_offset = np.random.uniform(-0.003, 0.003)
        trend_bias = 0.001 if np.random.random() > 0.5 else -0.001
        close_price = current_price * (1 + price_offset + trend_bias)
        high = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low = close_price * (1 - np.random.uniform(0.0005, 0.001))
        volume = int(np.random.uniform(6000, 20000))
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high,6), "low": round(low,6), "close": round(close_price,6), "volume": volume})
    return minute_data

# ---------------------------
# Streamlit UI pieces
# ---------------------------

def render_sidebar():
    st.sidebar.header("参数与风控")
    principal = st.sidebar.number_input("交易本金（港元）", min_value=1000.0, max_value=5_000_000.0, value=100000.0, step=1000.0)
    etf_code = st.sidebar.text_input("ETF 代码（雅虎财经）", value="2800.HK")
    current_price = st.sidebar.number_input("当前价格（港元）", min_value=0.0001, value=27.5, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.subheader("ETF 日均成交额")
    mode = st.sidebar.radio("日均成交额来源", ["自动获取", "手动输入"], horizontal=True)
    if mode == "自动获取":
        turnover_days = st.sidebar.selectbox("取多少日均成交额", [5, 10, 20, 60], index=2)
        avg_daily_turnover = get_avg_turnover(etf_code, days=turnover_days)
        if avg_daily_turnover:
            st.sidebar.success(f"过去 {turnover_days} 日均成交额：{avg_daily_turnover:,.0f} 港元")
        else:
            avg_daily_turnover = st.sidebar.number_input("ETF 日均成交额（港元）", min_value=0.0, value=500_000_000.0, step=1_000_000.0)
    else:
        avg_daily_turnover = st.sidebar.number_input("ETF 日均成交额（港元）", min_value=0.0, value=500_000_000.0, step=1_000_000.0)

    rec_low, rec_mid, rec_high = recommend_slippage_by_turnover(avg_daily_turnover)
    slippage_pct = st.sidebar.number_input("滑点（%）", min_value=0.0, value=rec_mid, step=0.01)
    if st.sidebar.button("应用建议滑点"):
        slippage_pct = rec_mid
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("网格与风控")
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 15], index=1)
    grid_type = st.sidebar.radio("网格间距方式", ["动态间距（基于ATR）", "固定间距（手动）"])
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 10, 30, 16, 2)
    fixed_spacing_pct = None
    if grid_type != "动态间距（基于ATR）":
        fixed_spacing_pct = st.sidebar.slider("固定间距（%）", 0.1, 2.0, 0.3, 0.05)
    initial_cash_pct = st.sidebar.slider("初始可用现金占本金（%）", 10, 100, 50, 5)
    single_trade_pct = st.sidebar.slider("单次交易金额占本金（%）", 0.5, 20.0, 5.0, 0.5)
    shares_per_lot = st.sidebar.number_input("每手股数", min_value=1, value=100, step=1)
    max_position_pct = st.sidebar.slider("最大持仓占本金（%）", 10, 100, 50, 5)
    stop_loss_pct = st.sidebar.number_input("全局止损阈值（%），0为不启用", min_value=0.0, value=0.0, step=0.1)
    take_profit_pct = st.sidebar.number_input("全局止盈阈值（%），0为不启用", min_value=0.0, value=0.0, step=0.1)

    cfg = {
        "platform_fee": 15.0,
        "trade_fee_pct": 0.00565,
        "settlement_fee_pct": 0.0042,
        "sfc_fee_pct": 0.0027,
        "frc_fee_pct": 0.00015,
        "slippage_pct": float(slippage_pct),
        "initial_cash_pct": float(initial_cash_pct / 100.0),
        "single_trade_amount": float(principal * (single_trade_pct / 100.0)),
        "shares_per_lot": int(shares_per_lot),
        "max_position_pct": float(max_position_pct / 100.0),
        "stop_loss_pct": float(stop_loss_pct) if stop_loss_pct > 0 else None,
        "take_profit_pct": float(take_profit_pct) if take_profit_pct > 0 else None
    }
    return principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover

# ---------------------------
# Tabs: data / strategy / backtest / 参数分析 / ETF对比 / help
# ---------------------------

def render_tab_data():
    st.subheader("分钟数据（获取 / 编辑 / 生成）")
    if "minute_data" not in st.session_state:
        st.session_state.minute_data = []
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.write("数据来源：雅虎财经（yfinance）")
    with c2:
        if st.button("从雅虎财经获取当天分钟数据"):
            etf_code = st.session_state.get("etf_code", "02800.HK")
            interval = st.session_state.get("data_interval", 5)
            imap = {1:"1m",5:"5m",15:"15m"}
            with st.spinner(f"下载 {etf_code} {imap.get(interval,'5m')} 数据..."):
                md = fetch_minute_data_yahoo(etf_code, interval=imap.get(interval,"5m"), period="1d")
            if md:
                st.session_state.minute_data = md
                st.session_state.current_price = md[-1]["close"]
                st.success(f"已获取 {len(md)} 条分钟数据，当前价 {md[-1]['close']:.4f}")
            else:
                st.warning("未获取到有效数据，可能为休市或代码错误。")
    with c3:
        if st.button("生成模拟数据"):
            cp = st.session_state.get("current_price", 27.5)
            di = st.session_state.get("data_interval", 5)
            st.session_state.minute_data = generate_default_minute_data(current_price=cp, interval=di)
            st.success("已生成模拟分钟数据")

    if not st.session_state.minute_data:
        st.session_state.minute_data = generate_default_minute_data()
    
    # 显示数据表格
    if st.session_state.minute_data:
        df = pd.DataFrame(st.session_state.minute_data)
        st.dataframe(df, height=300)
        
        # 价格图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name='收盘价', line=dict(color='blue')))
        fig.update_layout(title='价格走势', xaxis_title='时间', yaxis_title='价格（港元）', height=400)
        st.plotly_chart(fig)

def render_tab_strategy():
    st.subheader("网格策略设置")
    if not st.session_state.get("minute_data"):
        st.warning("请先在【数据】标签页获取或生成分钟数据")
        return
    
    current_price = st.session_state.current_price
    grid_count = st.session_state.grid_count
    grid_type = st.session_state.grid_type
    fixed_spacing_pct = st.session_state.fixed_spacing_pct
    
    # 计算网格参数
    if grid_type == "动态间距（基于ATR）":
        highs = [d['high'] for d in st.session_state.minute_data]
        lows = [d['low'] for d in st.session_state.minute_data]
        closes = [d['close'] for d in st.session_state.minute_data]
        atr_values = calculate_atr(highs, lows, closes)
        atr = atr_values[-1] if atr_values else 0
        atr_pct = (atr / current_price) * 100 if current_price != 0 else 0.3
        spacing_pct = max(0.1, round(atr_pct / 2, 2))  # ATR的一半作为间距
        st.info(f"基于ATR的动态间距：{spacing_pct}%（最新ATR：{atr:.4f}）")
    else:
        spacing_pct = fixed_spacing_pct
    
    # 网格上下限
    upper_limit = current_price * 1.05
    lower_limit = current_price * 0.95
    st.write(f"网格范围：{lower_limit:.4f} - {upper_limit:.4f}（当前价±5%）")
    
    # 生成网格
    buy_grids, sell_grids = generate_intraday_grid_arithmetic(
        current_price, spacing_pct, grid_count, upper_limit, lower_limit
    )
    
    # 显示网格
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("买入网格")
        buy_df = pd.DataFrame({"价格（港元）": buy_grids})
        st.dataframe(buy_df)
    with col2:
        st.subheader("卖出网格")
        sell_df = pd.DataFrame({"价格（港元）": sell_grids})
        st.dataframe(sell_df)
    
    # 网格可视化
    fig = go.Figure()
    fig.add_hline(y=current_price, line_dash="dash", line_color="black", name="当前价")
    for price in buy_grids:
        fig.add_hline(y=price, line_color="green", line_width=1, name="买入价" if price == buy_grids[0] else None)
    for price in sell_grids:
        fig.add_hline(y=price, line_color="red", line_width=1, name="卖出价" if price == sell_grids[0] else None)
    fig.update_yaxes(range=[min(buy_grids[0] if buy_grids else lower_limit, lower_limit*0.99), 
                           max(sell_grids[-1] if sell_grids else upper_limit, upper_limit*1.01)])
    fig.update_layout(title="网格分布", yaxis_title="价格（港元）", height=300)
    st.plotly_chart(fig)
    
    # 保存网格到会话状态
    st.session_state.buy_grids = buy_grids
    st.session_state.sell_grids = sell_grids

def render_tab_backtest():
    st.subheader("策略回测结果")
    if not all(key in st.session_state for key in ["minute_data", "buy_grids", "sell_grids"]):
        st.warning("请先在【策略】标签页生成网格")
        return
    
    # 执行回测
    if st.button("开始回测"):
        with st.spinner("正在执行回测..."):
            result = backtest_intraday_strategy_improved(
                principal=st.session_state.principal,
                current_price=st.session_state.current_price,
                buy_grids=st.session_state.buy_grids,
                sell_grids=st.session_state.sell_grids,
                minute_data=st.session_state.minute_data,
                cfg=st.session_state.cfg
            )
            st.session_state.backtest_result = result
        
        # 显示关键指标
        if "backtest_result" in st.session_state:
            res = st.session_state.backtest_result
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最终净值", f"HK${res['final_total_value']:,}")
            with col2:
                profit_color = "normal" if res['total_profit'] >= 0 else "normal"
                st.metric("总收益", f"HK${res['total_profit']:,}", 
                          f"{res['profit_rate']:.2f}%", delta_color=profit_color)
            with col3:
                st.metric("总交易次数", res['total_buy_count'] + res['total_sell_count'])
            with col4:
                st.metric("最大回撤", f"{res['max_drawdown']:.2f}%")
            
            # 净值曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['timestamps'], y=res['net_values'], name='净值'))
            fig.update_layout(title='日内净值变化', xaxis_title='时间', yaxis_title='净值（港元）', height=300)
            st.plotly_chart(fig)
            
            # 交易记录
            st.subheader("交易记录")
            if res['trade_records']:
                trade_df = pd.DataFrame(res['trade_records'])
                st.dataframe(trade_df, height=300)
            else:
                st.info("未产生交易记录")

def render_tab_sensitivity():
    st.subheader("网格参数敏感性分析")
    st.write("通过调整网格数量和间距，查看对回测结果的影响")
    
    if not st.session_state.get("minute_data"):
        st.warning("请先获取或生成分钟数据")
        return
    
    # 基础参数设置
    base_spacing = st.number_input("基准间距(%)", 0.1, 2.0, 0.3, 0.05)
    upper_limit = st.number_input("网格上限(当前价倍数)", 1.01, 1.1, 1.05, 0.01)
    lower_limit = st.number_input("网格下限(当前价倍数)", 0.9, 0.99, 0.95, 0.01)
    
    base_params = {
        "spacing": base_spacing,
        "upper": st.session_state.current_price * upper_limit,
        "lower": st.session_state.current_price * lower_limit
    }
    
    if st.button("开始分析"):
        results = analyze_grid_sensitivity(
            st.session_state.principal,
            st.session_state.current_price,
            st.session_state.minute_data,
            st.session_state.cfg,
            base_params
        )
        st.dataframe(results)
        
        # 可视化关键指标
        fig = make_subplots(rows=1, cols=2, subplot_titles=("收益 vs 网格数量", "最大回撤 vs 网格数量"))
        fig.add_trace(
            go.Scatter(x=results["网格数量"], y=results["收益(%)"], mode="markers", name="收益"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=results["网格数量"], y=results["最大回撤(%)"], mode="markers", name="最大回撤"),
            row=1, col=2
        )
        st.plotly_chart(fig)

def render_tab_etf_compare():
    st.subheader("多ETF日内T+0效果对比")
    etf_input = st.text_area("输入ETF代码（每行一个，如2800.HK）", "2800.HK\n3033.HK\n2828.HK")
    etf_codes = [code.strip() for code in etf_input.split("\n") if code.strip()]
    
    if st.button("开始对比") and etf_codes:
        comparison_df = compare_etfs(
            etf_codes,
            st.session_state.principal,
            st.session_state.data_interval,
            st.session_state.cfg
        )
        if not comparison_df.empty:
            st.dataframe(comparison_df.sort_values("收益(%)", ascending=False))
            
            # 可视化对比
            fig = go.Figure(data=[
                go.Bar(x=comparison_df["ETF代码"], y=comparison_df["收益(%)"], name="收益(%)"),
                go.Bar(x=comparison_df["ETF代码"], y=comparison_df["最大回撤(%)"], name="最大回撤(%)")
            ])
            fig.update_layout(barmode='group', title='ETF性能对比', height=400)
            st.plotly_chart(fig)

def render_tab_help():
    st.subheader("交易时间提醒")
    status = get_hk_trading_status()
    status_color = "green" if status["status"] == "交易中" else "orange" if status["status"] in ["未开盘", "午间休市"] else "red"
    st.markdown(f"**当前状态**: <span style='color:{status_color}'>{status['status']}</span>", unsafe_allow_html=True)
    st.info(status["message"])
    
    st.subheader("网格参数设置指南")
    st.markdown("""
    1. **网格数量**: 
       - 波动大的ETF建议10-16档（减少频繁交易）
       - 波动小的ETF建议22-28档（增加交易机会）
    
    2. **网格间距**:
       - 参考ATR指标（动态间距更优）
       - 流动性高的ETF（日均成交额>1亿）建议0.1-0.3%
       - 流动性低的ETF（日均成交额<1千万）建议0.5-1%
    
    3. **风险控制**:
       - 最大持仓不超过本金的50%（单边市风险）
       - 滑点设置参考日均成交额（系统会自动推荐）
    """)
    
   

def render_tabs():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "数据", "策略", "回测", "参数分析", "ETF对比", "帮助"
    ])
    
    with tab1:
        render_tab_data()
    with tab2:
        render_tab_strategy()
    with tab3:
        render_tab_backtest()
    with tab4:
        render_tab_sensitivity()
    with tab5:
        render_tab_etf_compare()
    with tab6:
        render_tab_help()

# ---------------------------
# Main app
# ---------------------------

def main():
    st.set_page_config(page_title="ETF日内网格策略", layout="wide")
    st.title("ETF日内T+0网格交易策略")
    
    # 初始化会话状态
    if "principal" not in st.session_state:
        principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover = render_sidebar()
        st.session_state.principal = principal
        st.session_state.etf_code = etf_code
        st.session_state.current_price = current_price
        st.session_state.cfg = cfg
        st.session_state.data_interval = data_interval
        st.session_state.grid_type = grid_type
        st.session_state.grid_count = grid_count
        st.session_state.fixed_spacing_pct = fixed_spacing_pct
        st.session_state.avg_daily_turnover = avg_daily_turnover
        st.session_state.minute_data = []
        st.session_state.buy_grids = []
        st.session_state.sell_grids = []
        st.session_state.backtest_result = None
    else:
        # 更新侧边栏参数
        principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover = render_sidebar()
        st.session_state.update({
            "principal": principal,
            "etf_code": etf_code,
            "current_price": current_price,
            "cfg": cfg,
            "data_interval": data_interval,
            "grid_type": grid_type,
            "grid_count": grid_count,
            "fixed_spacing_pct": fixed_spacing_pct,
            "avg_daily_turnover": avg_daily_turnover
        })
    
    # 渲染标签页
    render_tabs()

if __name__ == "__main__":
    main()

