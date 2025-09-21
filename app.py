# app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------------------
# Utilities / Cost / Parsing
# ---------------------------

def parse_volume(volume_input):
    """Parse volume text like '12k','3.5万' to integer."""
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
    """Return (low, mid, high) percentages (unit: %)"""
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
    """Estimate cost for one-side or round-trip"""
    slippage_cost = amount * (cfg["slippage_pct"] / 100.0)
    trade_fee = amount * (cfg["trade_fee_pct"] / 100.0)
    settlement_fee = amount * (cfg["settlement_fee_pct"] / 100.0)
    sfc_fee = amount * (cfg["sfc_fee_pct"] / 100.0)
    frc_fee = amount * (cfg["frc_fee_pct"] / 100.0)
    platform_fee = cfg.get("platform_fee", 0.0)
    single_total = platform_fee + trade_fee + settlement_fee + sfc_fee + frc_fee + slippage_cost
    return round(single_total if is_single_side else single_total * 2, 2)


# ---------------------------
# Yahoo Finance minute fetch (HK timezone + filter trading hours)
# ---------------------------

def fetch_minute_data_yahoo(etf_code, interval="5m", period="1d"):
    """
    Return list of dicts: [{'time':'HH:MM','high':..., 'low':..., 'close':..., 'volume':...}, ...]
    Time is converted to Asia/Hong_Kong and filtered to exchange trading hours:
      09:30-12:00 and 13:00-16:00 (inclusive)
    """
    try:
        data = yf.download(etf_code, interval=interval, period=period, progress=False)
        if data is None or data.empty:
            return []
        # If MultiIndex columns (sometimes for tickers), extract the right subframe
        if isinstance(data.columns, pd.MultiIndex):
            # Try to pick the ticker column if present
            try:
                data = data.xs(etf_code, axis=1, level=1)
            except Exception:
                # fallback: pick first ticker slice
                try:
                    data = data.xs(data.columns.levels[1][0], axis=1, level=1)
                except Exception:
                    # If still fails, leave as-is and hope column names match
                    pass
        # Ensure timezone is Hong Kong
        try:
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert('Asia/Hong_Kong')
            else:
                data.index = data.index.tz_convert('Asia/Hong_Kong')
        except Exception:
            # best-effort conversion
            try:
                data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert('Asia/Hong_Kong')
            except Exception:
                pass

        # filter by HK trading hours
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
            # handle different column name cases
            try:
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])
                v = int(row["Volume"]) if not np.isnan(row["Volume"]) else 0
            except Exception:
                # try lowercase keys
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
# Indicators / Grid / Backtest (improved)
# ---------------------------

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) == 0:
        return []
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
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
    buy.sort()
    sell.sort()
    return buy, sell


def calculate_max_drawdown_from_series(net_values):
    if not net_values:
        return 0.0
    s = pd.Series(net_values)
    rm = s.cummax()
    dd = (rm - s) / rm
    return round(float(dd.max() * 100), 4)


def backtest_intraday_strategy_improved(principal, current_price, buy_grids, sell_grids, minute_data, cfg):
    """
    Improved backtest:
      - use minute high/low to trigger
      - use close to value portfolio each minute
      - support stop loss / take profit (on net value)
      - support single trade amount (rounded to lots)
    Returns dictionary with records, metrics, net_values, timestamps.
    """
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
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        volume = int(row["volume"])

        triggered = True
        # allow multiple triggers within same minute as long as there are grid levels left
        while triggered:
            triggered = False
            # BUY: from low to high
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

            # SELL: from high to low
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

        # stoploss / takeprofit check (net value vs initial)
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
# Default minute data generator (morning + afternoon)
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
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high, 6), "low": round(low, 6), "close": round(close_price, 6), "volume": volume})
    for t in create_range("13:00", "16:00"):
        price_offset = np.random.uniform(-0.003, 0.003)
        trend_bias = 0.001 if np.random.random() > 0.5 else -0.001
        close_price = current_price * (1 + price_offset + trend_bias)
        high = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low = close_price * (1 - np.random.uniform(0.0005, 0.001))
        volume = int(np.random.uniform(6000, 20000))
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high, 6), "low": round(low, 6), "close": round(close_price, 6), "volume": volume})
    return minute_data


# ---------------------------
# Streamlit UI components
# ---------------------------

def render_sidebar():
    st.sidebar.header("参数与风控（鼠标悬停查看提示）")
    principal = st.sidebar.number_input("交易本金（港元）", min_value=1000.0, max_value=5_000_000.0, value=100000.0, step=1000.0, help="用于本次回测/策略验证的模拟资金。")
    etf_code = st.sidebar.text_input("ETF 代码（雅虎财经）", value="02800.HK", help="示例：02800.HK（盈富基金）。")
    current_price = st.sidebar.number_input("当前价格（港元）", min_value=0.0001, value=27.5, format="%.4f", help="用于生成网格的参考价；从雅虎财经拉取数据时会自动更新。")

    st.sidebar.markdown("---")
    st.sidebar.subheader("手续费 / 滑点")
    platform_fee = st.sidebar.number_input("平台费（每笔，港元）", min_value=0.0, value=15.0, step=1.0, help="部分券商每笔固定费用。")
    trade_fee_pct = st.sidebar.number_input("交易佣金（%）", min_value=0.0, value=0.00565, step=0.00001, help="示例 0.00565 表示 0.00565%。")
    settlement_fee_pct = st.sidebar.number_input("交收费（%）", min_value=0.0, value=0.0042, step=0.00001)
    sfc_fee_pct = st.sidebar.number_input("证监会费（%）", min_value=0.0, value=0.0027, step=0.00001)
    frc_fee_pct = st.sidebar.number_input("FRC费（%）", min_value=0.0, value=0.00015, step=0.00001)

    avg_daily_turnover = st.sidebar.number_input("ETF 日均成交额（港元）", min_value=0.0, value=500_000_000.0, step=1_000_000.0, help="用于滑点推荐，可在港交所/券商软件查看。")
    rec_low, rec_mid, rec_high = recommend_slippage_by_turnover(avg_daily_turnover)
    st.sidebar.caption(f"经验推荐滑点范围：{rec_low:.3f}% ~ {rec_high:.3f}%，建议：{rec_mid:.3f}%")
    slippage_pct = st.sidebar.number_input("滑点（%）", min_value=0.0, value=rec_mid, step=0.01)

    if st.sidebar.button("应用建议滑点"):
        slippage_pct = rec_mid
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("网格 & 回测参数")
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 15], index=1, help="1m 精细；5m 折中；15m 平滑。")
    grid_type = st.sidebar.radio("网格间距方式", ["动态间距（基于ATR）", "固定间距（手动）"])
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 10, 30, 16, 2, help="建议 12-20 之间。")
    fixed_spacing_pct = None
    if grid_type != "动态间距（基于ATR）":
        fixed_spacing_pct = st.sidebar.slider("固定间距（%）", 0.1, 2.0, 0.3, 0.05, help="每格间距百分比。")

    st.sidebar.markdown("---")
    st.sidebar.subheader("风控")
    initial_cash_pct = st.sidebar.slider("初始可用现金占本金（%）", 10, 100, 50, 5, help="用于挂单/扫单的初始可用资金比例。")
    single_trade_pct = st.sidebar.slider("单次交易金额占本金（%）", 0.5, 20.0, 5.0, 0.5, help="每次下单目标资金比例。")
    shares_per_lot = st.sidebar.number_input("每手股数", min_value=1, value=100, step=1, help="港股通常为 100 股整手。")
    max_position_pct = st.sidebar.slider("最大持仓占本金（%）", 10, 100, 50, 5, help="策略允许的最大仓位，避免满仓风险。")
    stop_loss_pct = st.sidebar.number_input("全局止损阈值（%），0为不启用", min_value=0.0, value=0.0, step=0.1)
    take_profit_pct = st.sidebar.number_input("全局止盈阈值（%），0为不启用", min_value=0.0, value=0.0, step=0.1)

    cfg = {
        "platform_fee": float(platform_fee),
        "trade_fee_pct": float(trade_fee_pct),
        "settlement_fee_pct": float(settlement_fee_pct),
        "sfc_fee_pct": float(sfc_fee_pct),
        "frc_fee_pct": float(frc_fee_pct),
        "slippage_pct": float(slippage_pct),
        "initial_cash_pct": float(initial_cash_pct / 100.0),
        "single_trade_amount": float(principal := principal if (principal := locals().get('principal', None)) else None)  # placeholder - overridden below
    }

    # Override single_trade_amount properly using principal and selected percentage
    cfg["single_trade_amount"] = float(principal) * (single_trade_pct / 100.0) if principal else 0.0
    cfg["shares_per_lot"] = int(shares_per_lot)
    cfg["max_position_pct"] = float(max_position_pct / 100.0)
    cfg["stop_loss_pct"] = float(stop_loss_pct) if stop_loss_pct > 0 else None
    cfg["take_profit_pct"] = float(take_profit_pct) if take_profit_pct > 0 else None
    cfg["initial_cash_pct"] = float(initial_cash_pct / 100.0)

    return principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover


# ---------------------------
# Tabs rendering
# ---------------------------

def render_tab_data():
    st.subheader("分钟数据（获取 / 编辑 / 生成）")
    if "minute_data" not in st.session_state:
        st.session_state.minute_data = []

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write("来源：雅虎财经（yfinance）")
    with col2:
        if st.button("从雅虎财经获取当天分钟数据"):
            etf_code = st.session_state.get("etf_code", "02800.HK")
            interval = st.session_state.get("data_interval", 5)
            imap = {1:"1m",5:"5m",15:"15m"}
            with st.spinner(f"下载 {etf_code} {imap.get(interval,'5m')} 数据..."):
                md = fetch_minute_data_yahoo(etf_code, interval=imap.get(interval,"5m"), period="1d")
            if md:
                st.session_state.minute_data = md
                st.session_state.current_price = md[-1]["close"]
                st.success(f"已获取 {len(md)} 条分钟数据（时间为香港时区），当前价更新为 {md[-1]['close']:.4f}")
            else:
                st.warning("未获取到有效数据：可能为休市或代码错误。")
    with col3:
        if st.button("生成模拟/示例数据"):
            cp = st.session_state.get("current_price", 27.5)
            di = st.session_state.get("data_interval", 5)
            st.session_state.minute_data = generate_default_minute_data(current_price=cp, interval=di)
            st.success("已生成模拟分钟数据（仅用于测试）")

    if not st.session_state.minute_data:
        st.info("当前无分钟数据，建议点击“从雅虎财经获取当天分钟数据”或生成模拟数据。")
        st.session_state.minute_data = generate_default_minute_data()

    # show editable table
    df_display = pd.DataFrame(st.session_state.minute_data)
    # show nicely formatted volume
    df_display["成交量(示例)"] = df_display["volume"].apply(lambda v: f"{v}" if v < 1000 else (f"{v/1000:.1f}k" if v<10000 else f"{v/10000:.2f}万"))
    st.write("（提示：可在表格编辑并点击“保存编辑”）")
    try:
        edited = st.data_editor(df_display[["time","high","low","close","成交量(示例)"]], use_container_width=True, num_rows="dynamic", key="minute_editor")
    except Exception:
        # fallback if data_editor not available
        edited = st.experimental_data_editor(df_display[["time","high","low","close","成交量(示例)"]], use_container_width=True, num_rows="dynamic", key="minute_editor")

    if st.button("保存编辑"):
        updated = []
        for idx, row in edited.iterrows():
            t = str(row["time"]).strip()
            try:
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = parse_volume(row["成交量(示例)"])
                updated.append({"time": t, "high": round(h,6), "low": round(l,6), "close": round(c,6), "volume": int(v)})
            except Exception as e:
                st.warning(f"第{idx+1}行解析失败，已跳过：{e}")
        if updated:
            try:
                updated.sort(key=lambda x: datetime.strptime(x["time"], "%H:%M"))
            except:
                pass
            st.session_state.minute_data = updated
            st.success(f"已保存 {len(updated)} 条分钟数据")
        else:
            st.warning("未保存任何有效数据")

    st.caption("时间格式 HH:MM，例如 09:30。成交量支持 'k','万' 等单位输入。")


def render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover):
    st.subheader("网格生成与参数（含智能推荐）")
    minute_data = st.session_state.get("minute_data", generate_default_minute_data(current_price))
    st.session_state["cfg"] = cfg
    st.session_state["etf_code"] = etf_code
    st.session_state["data_interval"] = data_interval
    st.session_state["current_price"] = st.session_state.get("current_price", current_price)

    closes = [d["close"] for d in minute_data] if minute_data else []
    highs = [d["high"] for d in minute_data] if minute_data else []
    lows = [d["low"] for d in minute_data] if minute_data else []

    atr = calculate_atr(highs, lows, closes, period=14) if closes else [0.0]
    latest_atr = atr[-1] if atr else 0.01

    # base spacing
    cp = st.session_state.get("current_price", current_price) or current_price
    base_spacing_pct = (latest_atr * 0.6 / cp) * 100 if cp > 0 else 0.3
    single_trade_amount = cfg.get("single_trade_amount", principal * 0.05)
    round_trip_cost = calculate_trade_cost_simple(single_trade_amount, cfg, is_single_side=False)
    min_safe_spacing_pct = (round_trip_cost / single_trade_amount) * 100 * 1.2 if single_trade_amount > 0 else 0.2
    final_spacing_pct = max(base_spacing_pct, min_safe_spacing_pct, 0.2)

    if grid_type != "动态间距（基于ATR）" and fixed_spacing_pct is not None:
        final_spacing_pct = fixed_spacing_pct

    latest_upper = (max(closes[-10:]) if closes else cp * 1.01) * 1.005
    latest_lower = (min(closes[-10:]) if closes else cp * 0.99) * 0.995
    grid_upper = min(latest_upper, cp * 1.05)
    grid_lower = max(latest_lower, cp * 0.95)

    if grid_count % 2 != 0:
        grid_count += 1

    buy_grids, sell_grids = generate_intraday_grid_arithmetic(cp, final_spacing_pct, grid_count, grid_upper, grid_lower)

    st.markdown("### 基本信息与参数")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"交易标的：**{etf_code}**")
        st.write(f"本金：{principal:,.0f} 港元")
        st.write(f"当前价（用于生成网格）：{cp:.4f}")
        vwap = calculate_vwap(minute_data)
        st.write(f"样本 VWAP：{vwap if vwap else 'N/A'}")
    with col2:
        st.write(f"网格区间：{round(grid_lower,6)} ~ {round(grid_upper,6)} 港元")
        st.write(f"间距（%）：{round(final_spacing_pct,4)}%")
        st.write(f"档数：{grid_count} (买{len(buy_grids)} / 卖{len(sell_grids)})")
        st.write(f"单次交易额：{round(single_trade_amount,2)} 港元")
        st.write(f"估算 round-trip 成本：{round(round_trip_cost,2)} 港元")

    st.markdown("##### 买入档位（低->高）")
    if buy_grids:
        st.dataframe(pd.DataFrame({"买入档位": [f"买{i+1}" for i in range(len(buy_grids))], "价格(港元)": buy_grids}), use_container_width=True)
    else:
        st.warning("未生成买入网格")

    st.markdown("##### 卖出档位（低->高）")
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
    st.subheader("回测结果与可视化")
    result = st.session_state.get("backtest_result")
    minute_data = st.session_state.get("minute_data", None)
    if not result or not minute_data:
        st.info("请先生成网格并运行回测（或先获取分钟数据）")
        return

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初始本金(港元)", f"{principal:,.0f}")
    col2.metric("最终市值(港元)", f"{result['final_total_value']:,.2f}", delta=f"{result['total_profit']:.2f}")
    col3.metric("收益率(%)", f"{result['profit_rate']:.4f}%")
    col4.metric("最大回撤(%)", f"{result['max_drawdown']:.4f}%")

    # Prepare dataframes
    df_trades = pd.DataFrame(result["trade_records"])
    df_nv = pd.DataFrame({"time": result["timestamps"], "net_value": result["net_values"], "holding": result["holdings_history"]})

    # Baseline (buy-and-hold) — theoretical (no fees)
    first_price = minute_data[0]["close"] if minute_data else 1.0
    baseline = [(principal / first_price) * row["close"] for row in minute_data]
    times = [row["time"] for row in minute_data]
    volumes = [row["volume"] for row in minute_data]

    # Plotly subplot: net value (top) and volume (bottom)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=times, y=result["net_values"], mode="lines", name="网格净值"), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=baseline, mode="lines", name="买入持有(理论)", line=dict(dash="dot")), row=1, col=1)

    # trade markers
    if not df_trades.empty:
        buys = df_trades[df_trades["type"] == "buy"]
        sells = df_trades[df_trades["type"].isin(["sell", "stoploss_sell", "takeprofit_sell"])]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["time"], y=buys["price"], mode="markers", marker=dict(color="green", size=9), name="买入点"), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["time"], y=sells["price"], mode="markers", marker=dict(color="red", size=9), name="卖出点"), row=1, col=1)

    # volume bars on second row
    fig.add_trace(go.Bar(x=times, y=volumes, name="成交量", marker=dict(opacity=0.6)), row=2, col=1)

    fig.update_layout(height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="时间 (HH:MM)")
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 交易明细")
    if df_trades.empty:
        st.info("本次回测未产生交易")
    else:
        st.dataframe(df_trades)
        csv = df_trades.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载交易明细 CSV", data=csv, file_name=f"trade_records_{etf_code}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    st.markdown("### 净值表（可下载）")
    df_nv_download = pd.DataFrame({"time": times, "close": [r["close"] for r in minute_data], "net_value": result["net_values"]})
    csv_nv = df_nv_download.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下载净值 CSV", data=csv_nv, file_name=f"net_values_{etf_code}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")


def render_tab_help():
    st.subheader("新手指南 & 常见问题")
    st.markdown("""
    **使用步骤**
    1. 在侧边栏填写本金、ETF 代码等参数（鼠标悬停可看说明）。  
    2. 到“分钟数据”页，点击“从雅虎财经获取当天分钟数据”或生成模拟数据。  
    3. 到“网格策略”页查看推荐间距并点击“开始回测”。  
    4. 在“回测结果”页查看净值曲线、买卖点与交易明细，点击下载保存 CSV。

    **ETF 日均成交额从哪查？**
    - 雅虎财经（Yahoo Finance）页面的 Statistics 或 Summary（Average Volume）；  
    - 港交所（HKEX）官网行情页面；  
    - 券商行情软件（例如富途、华盛或同花顺）通常在个股详情里显示成交额。

    **注意**
    - yfinance 拉取分钟数据有时会受限（非交易日或调整），若获取失败请稍后或用模拟数据测试。  
    - 回测为近似模拟，滑点/手续费/最小收费/整手限制会影响实盘表现，务必小仓位验证。  
    """)


# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(page_title="香港日内网格 T+0（增强版）", layout="wide")
    st.title("🇭🇰 香港日内 网格 T+0 策略工具（增强版）")

    principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover = render_sidebar()

    # Keep context in session
    st.session_state["cfg"] = cfg
    st.session_state["etf_code"] = etf_code
    st.session_state["data_interval"] = data_interval
    if "current_price" not in st.session_state:
        st.session_state["current_price"] = current_price

    tabs = st.tabs(["分钟数据", "网格策略", "回测结果", "新手说明"])
    with tabs[0]:
        render_tab_data()
    with tabs[1]:
        render_tab_strategy(principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover)
    with tabs[2]:
        render_tab_backtest(principal, etf_code)
    with tabs[3]:
        render_tab_help()

    st.caption("提示：若你不清楚某个参数的意义，把鼠标移到该输入框上查看帮助，或在聊天里把参数名发给我，我会具体解释该如何取值。")

if __name__ == "__main__":
    main()
