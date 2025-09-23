# ETF日内网格策略 - 完整增强版
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytz
import math
import io

# ---------------------------
# 工具函数
# ---------------------------

def parse_volume(volume_input):
    """解析成交量字符串（如12k、3.5万）为整数"""
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
    """根据日均成交额推荐滑点"""
    if not avg_daily_turnover or avg_daily_turnover <= 0:
        return (0.05, 0.15, 0.3)
    if avg_daily_turnover >= 1_000_000_000:
        return (0.03, 0.06, 0.12)
    if avg_daily_turnover >= 500_000_000:
        return (0.05, 0.12, 0.2)
    if avg_daily_turnover >= 50_000_000:
        return (0.1, 0.25, 0.5)
    return (0.3, 0.7, 1.5)

def calculate_trade_cost_simple(amount, cfg, side='buy'):
    """计算交易成本"""
    if amount <= 0:
        return 0.0
    slippage_cost = amount * (cfg.get("slippage_pct", 0.0) / 100.0)
    trade_fee = amount * (cfg.get("trade_fee_pct", 0.0) / 100.0)
    settlement_fee = amount * (cfg.get("settlement_fee_pct", 0.0) / 100.0)
    sfc_fee = amount * (cfg.get("sfc_fee_pct", 0.0) / 100.0)
    frc_fee = amount * (cfg.get("frc_fee_pct", 0.0) / 100.0)
    platform_fee = cfg.get("platform_fee", 0.0)
    stamp = 0.0
    if side == 'sell' and cfg.get("stamp_duty_pct", 0.0) > 0:
        stamp = amount * (cfg.get("stamp_duty_pct", 0.0) / 100.0)
    total = platform_fee + trade_fee + settlement_fee + sfc_fee + frc_fee + slippage_cost + stamp
    return round(total, 2)

def get_avg_turnover(ticker, days=20):
    """获取日均成交额"""
    try:
        data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if data is None or data.empty:
            return None
        avg_turnover = (data["Close"] * data["Volume"]).mean()
        return float(avg_turnover)
    except Exception:
        return None

def get_hk_trading_status():
    """获取港股交易状态"""
    now = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    today = now.date()
    is_weekday = now.weekday() < 5
    
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

# ---------------------------
# 数据获取函数
# ---------------------------

def fetch_minute_data_yahoo(etf_code, interval="5m", period="1d"):
    """从雅虎财经获取分钟数据"""
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
        # 时区处理
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
        # 过滤交易时间
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

def generate_default_minute_data(current_price=27.5, interval=5):
    """生成模拟分钟数据"""
    minute_data = []
    def create_range(start_str, end_str):
        start = datetime.strptime(start_str, "%H:%M")
        end = datetime.strptime(end_str, "%H:%M")
        t = start
        while t <= end:
            yield t
            t += timedelta(minutes=interval)
    
    # 上午交易时段
    for t in create_range("09:30", "12:00"):
        price_offset = np.random.uniform(-0.003, 0.003)
        close_price = current_price * (1 + price_offset)
        high = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low = close_price * (1 - np.random.uniform(0.0005, 0.001))
        volume = int(np.random.uniform(8000, 25000))
        minute_data.append({"time": t.strftime("%H:%M"), "high": round(high,6), "low": round(low,6), "close": round(close_price,6), "volume": volume})
    
    # 下午交易时段
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
# 技术指标计算
# ---------------------------

def calculate_atr(highs, lows, closes, period=14):
    """计算平均真实波幅(ATR)"""
    if len(closes) == 0:
        return []
    highs = np.array(highs); lows = np.array(lows); closes = np.array(closes)
    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().round(6).tolist()
    return atr

def calculate_vwap(minute_data):
    """计算成交量加权平均价(VWAP)"""
    prices = np.array([d["close"] for d in minute_data], dtype=float)
    volumes = np.array([d["volume"] for d in minute_data], dtype=float)
    if volumes.sum() == 0:
        return None
    return round(float((prices * volumes).sum() / volumes.sum()), 6)

def calculate_rsi(prices, period=14):
    """计算相对强弱指数(RSI)"""
    if len(prices) < period:
        return [50] * len(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = pd.Series(gains).rolling(period).mean()
    avg_losses = pd.Series(losses).rolling(period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return [50] + rsi.fillna(50).tolist()

# ---------------------------
# 网格生成函数
# ---------------------------

def generate_intraday_grid_arithmetic(current_price, spacing_pct, grid_count, grid_upper, grid_lower, 
                                    center_moving=False, center_price=None, volatility_mode=False, minute_data=None):
    """生成日内网格"""
    if center_moving and center_price is not None:
        base = center_price
    else:
        base = current_price
        
    if volatility_mode and minute_data:
        closes = np.array([d['close'] for d in minute_data], dtype=float)
        if len(closes) > 1:
            std = float(np.std(closes))
            std_pct = (std / base) * 100.0
            effective_spacing = (std_pct * spacing_pct) / 100.0
        else:
            effective_spacing = (spacing_pct / 100.0)
    else:
        effective_spacing = spacing_pct / 100.0
        
    half = grid_count // 2
    buy = [round(base * (1 - effective_spacing * (i + 1)), 6) for i in range(half)]
    sell = [round(base * (1 + effective_spacing * (i + 1)), 6) for i in range(half)]
    
    buy = [p for p in buy if p >= grid_lower]
    sell = [p for p in sell if p <= grid_upper]
    
    buy.sort(); sell.sort()
    return buy, sell

# ---------------------------
# 回测引擎
# ---------------------------

def calculate_max_drawdown_from_series(net_values):
    """计算最大回撤"""
    if not net_values:
        return 0.0
    s = pd.Series(net_values)
    rm = s.cummax()
    dd = (rm - s) / rm
    return round(float(dd.max() * 100), 4)

def compute_risk_metrics(net_values, principal, profit_rate, max_drawdown, trade_records, minute_data):
    """计算风险指标"""
    metrics = {}
    
    # 夏普比率
    if len(net_values) >= 2:
        arr = np.array(net_values, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        mean_ret = np.mean(rets)
        std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 0.0
        sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else None
        metrics['sharpe'] = round(float(sharpe), 4) if sharpe is not None else None
    else:
        metrics['sharpe'] = None
    
    # 卡尔玛比率
    annual_return = (profit_rate / 100.0) * 252
    metrics['calmar'] = round(float(annual_return / (max_drawdown/100.0)), 4) if max_drawdown > 0 else None
    
    # 交易统计
    realized_trades = []
    buy_q = []
    
    for r in trade_records:
        if r['type'] == 'buy':
            buy_q.append({'price': r['price'], 'shares': r['shares']})
        elif r['type'].startswith('sell'):
            rem = r['shares']
            while rem > 0 and buy_q:
                lot = buy_q[0]
                if lot['shares'] <= rem:
                    matched = lot['shares']
                    pnl = matched * (r['price'] - lot['price'])
                    realized_trades.append(pnl)
                    rem -= matched
                    buy_q.pop(0)
                else:
                    matched = rem
                    pnl = matched * (r['price'] - lot['price'])
                    realized_trades.append(pnl)
                    lot['shares'] -= matched
                    rem = 0
    
    # 胜率、盈亏比等
    wins = [p for p in realized_trades if p > 0]
    losses = [p for p in realized_trades if p <= 0]
    win_rate = (len(wins) / len(realized_trades) * 100) if realized_trades else None
    avg_win = np.mean(wins) if wins else None
    avg_loss = np.mean(losses) if losses else None
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and abs(sum(losses)) > 0 else None
    
    metrics['win_rate'] = round(float(win_rate), 2) if win_rate is not None else None
    metrics['avg_win'] = round(float(avg_win), 2) if avg_win is not None else None
    metrics['avg_loss'] = round(float(avg_loss), 2) if avg_loss is not None else None
    metrics['profit_factor'] = round(float(profit_factor), 4) if profit_factor is not None else None
    
    return metrics

def backtest_intraday_strategy_improved(principal, current_price, buy_grids, sell_grids, minute_data, cfg):
    """改进的日内策略回测"""
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
                    cost = calculate_trade_cost_simple(buy_amount, cfg, side='buy')
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
                    cost = calculate_trade_cost_simple(sell_amount, cfg, side='sell')
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
        
        # 止损检查
        if stop_loss_pct is not None and net_value <= initial_net * (1 - stop_loss_pct / 100.0):
            if shares >= shares_per_lot:
                sell_shares = int(shares // shares_per_lot) * shares_per_lot
                sell_amount = sell_shares * close
                cost = calculate_trade_cost_simple(sell_amount, cfg, side='sell')
                shares -= sell_shares
                cash += (sell_amount - cost)
                trade_records.append({
                    "time": t, "type": "stoploss_sell", "price": close, "shares": sell_shares,
                    "amount": round(sell_amount, 2), "cost": round(cost, 2),
                    "cash_after": round(cash, 2), "holding_after": shares
                })
            break
            
        # 止盈检查
        if take_profit_pct is not None and net_value >= initial_net * (1 + take_profit_pct / 100.0):
            if shares >= shares_per_lot:
                sell_shares = int(shares // shares_per_lot) * shares_per_lot
                sell_amount = sell_shares * close
                cost = calculate_trade_cost_simple(sell_amount, cfg, side='sell')
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
    
    # 计算风险指标
    metrics = compute_risk_metrics(net_values, principal, profit_rate, max_drawdown, trade_records, minute_data)
    
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
        "holdings_history": holdings_history,
        "metrics": metrics
    }

# ---------------------------
# 敏感性分析和ETF对比
# ---------------------------

def analyze_grid_sensitivity(principal, current_price, minute_data, cfg, base_params):
    """分析网格参数敏感性"""
    results = []
    for grid_count in [10, 16, 22, 28]:
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

def compare_etfs(etf_codes, principal, data_interval, cfg):
    """对比多个ETF的日内T+0效果"""
    comparison = []
    imap = {1:"1m", 5:"5m", 15:"15m"}
    interval = imap.get(data_interval, "5m")
    
    for code in etf_codes:
        with st.spinner(f"正在分析 {code}..."):
            minute_data = fetch_minute_data_yahoo(code, interval=interval, period="1d")
            if not minute_data:
                st.warning(f"{code} 获取数据失败，跳过")
                continue
            
            current_price = minute_data[-1]['close']
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
# 侧边栏参数设置
# ---------------------------

def render_sidebar():
    st.sidebar.header("🎯 参数与风控设置")
    
    # 基本信息
    principal_str = st.sidebar.text_input("交易本金（港元）", value="100000", 
                                        help="总投入本金。例如100000。新手建议 50,000-200,000 HKD")
    try:
        principal = float(principal_str)
    except:
        principal = 100000.0
        
    etf_code = st.sidebar.text_input("ETF 代码（雅虎财经）", value="2800.HK", 
                                   help="雅虎财经的代码，例如 2800.HK、3033.HK")
    
    current_price_str = st.sidebar.text_input("当前价格（港元）", value="27.5", 
                                            help="ETF 当前价格，完整输入小数，例如 6.03")
    try:
        current_price = float(current_price_str)
    except:
        current_price = 27.5

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 成交额 & 滑点")
    
    # 成交额设置
    turnover_mode = st.sidebar.radio("日均成交额来源", ["自动获取", "手动输入"], horizontal=True)
    if turnover_mode == "自动获取":
        turnover_days = st.sidebar.selectbox("取多少日均成交额", [5, 10, 20, 60], index=2)
        avg_daily_turnover = get_avg_turnover(etf_code, days=turnover_days)
        if avg_daily_turnover:
            st.sidebar.success(f"过去 {turnover_days} 日均成交额：{avg_daily_turnover:,.0f} 港元")
        else:
            turnover_str = st.sidebar.text_input("日均成交额（港元）", value="500000000")
            try:
                avg_daily_turnover = float(turnover_str)
            except:
                avg_daily_turnover = 500_000_000.0
    else:
        turnover_str = st.sidebar.text_input("日均成交额（港元）", value="500000000")
        try:
            avg_daily_turnover = float(turnover_str)
        except:
            avg_daily_turnover = 500_000_000.0

    # 滑点设置
    rec_low, rec_mid, rec_high = recommend_slippage_by_turnover(avg_daily_turnover)
    slippage_pct = st.sidebar.number_input("滑点（%）", min_value=0.0, value=rec_mid, step=0.01,
                                         help="成交价格偏离预期估计，高流动性0.03%-0.3%，低流动性更高")
    
    if st.sidebar.button("应用建议滑点"):
        slippage_pct = rec_mid

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 网格与数据周期")
    
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 15], index=1)
    grid_type = st.sidebar.radio("网格间距方式", ["动态间距（基于ATR）", "固定间距（手动）", "基于波动率（Std）"])
    
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 10, 30, 16, 2,
                                 help="网格总档数越高越密集，交易越频繁。新手推荐 12-20")
    
    fixed_spacing_pct = None
    volatility_multiplier = None
    
    if grid_type == "固定间距（手动）":
        fixed_spacing_str = st.sidebar.text_input("固定间距（%）", value="0.3")
        try:
            fixed_spacing_pct = float(fixed_spacing_str)
        except:
            fixed_spacing_pct = 0.3
    elif grid_type == "基于波动率（Std）":
        vol_mult_str = st.sidebar.text_input("波动率间距倍数", value="0.5")
        try:
            volatility_multiplier = float(vol_mult_str)
        except:
            volatility_multiplier = 0.5
    
    dynamic_grid_center = st.sidebar.checkbox("动态网格中心（随VWAP/均线移动）", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ 仓位与风控")
    
    # 仓位参数
    initial_cash_pct_str = st.sidebar.text_input("初始可用现金占本金（%）", value="50")
    try:
        initial_cash_pct = float(initial_cash_pct_str) / 100.0
    except:
        initial_cash_pct = 0.5
        
    single_trade_pct_str = st.sidebar.text_input("单次交易金额占本金（%）", value="5")
    try:
        single_trade_pct = float(single_trade_pct_str) / 100.0
    except:
        single_trade_pct = 0.05
        
    shares_per_lot_str = st.sidebar.text_input("每手股数", value="100")
    try:
        shares_per_lot = int(float(shares_per_lot_str))
    except:
        shares_per_lot = 100
        
    max_position_pct_str = st.sidebar.text_input("最大持仓占本金（%）", value="50")
    try:
        max_position_pct = float(max_position_pct_str) / 100.0
    except:
        max_position_pct = 0.5

    # 风控参数
    stop_loss_pct_str = st.sidebar.text_input("全局止损阈值（%），0为不启用", value="0")
    try:
        stop_loss_pct = float(stop_loss_pct_str) if float(stop_loss_pct_str) > 0 else None
    except:
        stop_loss_pct = None
        
    take_profit_pct_str = st.sidebar.text_input("全局止盈阈值（%），0为不启用", value="0")
    try:
        take_profit_pct = float(take_profit_pct_str) if float(take_profit_pct_str) > 0 else None
    except:
        take_profit_pct = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 扩展费用")
    
    stamp_duty_str = st.sidebar.text_input("印花税（卖出，%）", value="0")
    try:
        stamp_duty_pct = float(stamp_duty_str)
    except:
        stamp_duty_pct = 0.0

    # 构建配置字典
    cfg = {
        "platform_fee": 15.0,
        "trade_fee_pct": 0.00565,
        "settlement_fee_pct": 0.0042,
        "sfc_fee_pct": 0.0027,
        "frc_fee_pct": 0.00015,
        "slippage_pct": float(slippage_pct),
        "initial_cash_pct": initial_cash_pct,
        "single_trade_amount": principal * single_trade_pct,
        "shares_per_lot": shares_per_lot,
        "max_position_pct": max_position_pct,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "stamp_duty_pct": stamp_duty_pct,
        "grid_type": grid_type,
        "volatility_multiplier": volatility_multiplier,
        "dynamic_grid_center": dynamic_grid_center,
        "data_interval": data_interval,
    }
    
    return principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover

# ---------------------------
# 标签页渲染函数
# ---------------------------

def render_tab_data():
    st.subheader("📊 分钟数据管理")
    
    if "minute_data" not in st.session_state:
        st.session_state.minute_data = []
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.write("**数据来源：雅虎财经**")
        
    with col2:
        if st.button("🔄 从雅虎财经获取当天分钟数据", type="primary"):
            etf_code = st.session_state.get("etf_code", "2800.HK")
            interval = st.session_state.get("data_interval", 5)
            imap = {1:"1m", 5:"5m", 15:"15m"}
            
            with st.spinner(f"下载 {etf_code} {imap.get(interval,'5m')} 数据..."):
                md = fetch_minute_data_yahoo(etf_code, interval=imap.get(interval,"5m"), period="1d")
            
            if md:
                st.session_state.minute_data = md
                st.session_state.current_price = md[-1]["close"]
                st.success(f"✅ 已获取 {len(md)} 条分钟数据，当前价 {md[-1]['close']:.4f}")
            else:
                st.warning("❌ 未获取到有效数据，可能为休市或代码错误")
    
    with col3:
        if st.button("🎲 生成模拟数据"):
            cp = st.session_state.get("current_price", 27.5)
            di = st.session_state.get("data_interval", 5)
            st.session_state.minute_data = generate_default_minute_data(current_price=cp, interval=di)
            st.success("✅ 已生成模拟分钟数据")

    if not st.session_state.minute_data:
        st.session_state.minute_data = generate_default_minute_data()
    
    # 显示数据表格和图表
    if st.session_state.minute_data:
        df = pd.DataFrame(st.session_state.minute_data)
        
        st.subheader("数据预览")
        st.dataframe(df, height=300, use_container_width=True)
        
        # 价格图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name='收盘价', 
                               line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['high'], name='最高价', 
                               line=dict(color='green', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['time'], y=df['low'], name='最低价', 
                               line=dict(color='red', width=1, dash='dot')))
        
        fig.update_layout(
            title='价格走势图',
            xaxis_title='时间',
            yaxis_title='价格（港元）',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 成交量图表
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df['time'], y=df['volume'], name='成交量',
                               marker_color='lightblue'))
        fig_vol.update_layout(
            title='成交量',
            xaxis_title='时间',
            yaxis_title='成交量',
            height=300
        )
        st.plotly_chart(fig_vol, use_container_width=True)

def render_tab_strategy():
    st.subheader("🎯 网格策略设置")
    
    if not st.session_state.get("minute_data"):
        st.warning("⚠️ 请先在【数据】标签页获取或生成分钟数据")
        return
    
    current_price = st.session_state.current_price
    cfg = st.session_state.cfg
    grid_count = st.session_state.grid_count
    grid_type = st.session_state.grid_type
    fixed_spacing_pct = st.session_state.fixed_spacing_pct
    
    # 计算网格间距
    if grid_type == "动态间距（基于ATR）":
        highs = [d['high'] for d in st.session_state.minute_data]
        lows = [d['low'] for d in st.session_state.minute_data]
        closes = [d['close'] for d in st.session_state.minute_data]
        atr_values = calculate_atr(highs, lows, closes)
        atr = atr_values[-1] if atr_values else 0
        atr_pct = (atr / current_price) * 100 if current_price != 0 else 0.3
        spacing_pct = max(0.1, round(atr_pct / 2, 2))
        st.info(f"📏 基于ATR的动态间距：{spacing_pct}%（最新ATR：{atr:.4f}）")
    elif grid_type == "基于波动率（Std）":
        spacing_pct = cfg.get("volatility_multiplier", 0.5)
        st.info(f"📊 基于Std的间距倍数：{spacing_pct}")
    else:
        spacing_pct = fixed_spacing_pct if fixed_spacing_pct is not None else 0.3
        st.info(f"📍 固定间距：{spacing_pct}%")
    
    # 动态网格中心
    center_price = None
    if cfg.get("dynamic_grid_center", False):
        vwap = calculate_vwap(st.session_state.minute_data)
        if vwap:
            center_price = vwap
            st.info(f"🎯 动态网格中心使用 VWAP: {vwap:.4f}")
        else:
            df = pd.DataFrame(st.session_state.minute_data)
            if len(df) >= 5:
                center_price = df['close'].rolling(5).mean().iloc[-1]
                st.info(f"🎯 动态中心使用 MA5: {center_price:.4f}")
            else:
                center_price = current_price
    
    # 网格范围
    upper_limit = current_price * 1.05
    lower_limit = current_price * 0.95
    st.write(f"📋 网格范围：{lower_limit:.4f} - {upper_limit:.4f}（当前价±5%）")
    
    # 生成网格
    if grid_type == "基于波动率（Std）":
        buy_grids, sell_grids = generate_intraday_grid_arithmetic(
            current_price, spacing_pct, grid_count, upper_limit, lower_limit,
            center_moving=bool(center_price is not None), center_price=center_price,
            volatility_mode=True, minute_data=st.session_state.minute_data
        )
    else:
        buy_grids, sell_grids = generate_intraday_grid_arithmetic(
            current_price, spacing_pct, grid_count, upper_limit, lower_limit,
            center_moving=bool(center_price is not None), center_price=center_price
        )
    
    # 显示网格
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 买入网格")
        if buy_grids:
            buy_df = pd.DataFrame({"价格（港元）": buy_grids})
            st.dataframe(buy_df, use_container_width=True)
        else:
            st.info("无买入网格")
    
    with col2:
        st.subheader("🔴 卖出网格")
        if sell_grids:
            sell_df = pd.DataFrame({"价格（港元）": sell_grids})
            st.dataframe(sell_df, use_container_width=True)
        else:
            st.info("无卖出网格")
    
    # 网格可视化
    if buy_grids or sell_grids:
        fig = go.Figure()
        fig.add_hline(y=current_price, line_dash="dash", line_color="black", 
                     annotation_text="当前价", annotation_position="bottom right")
        
        for i, price in enumerate(buy_grids):
            fig.add_hline(y=price, line_color="green", line_width=1,
                         annotation_text=f"买{i+1}" if i == 0 else "")
        
        for i, price in enumerate(sell_grids):
            fig.add_hline(y=price, line_color="red", line_width=1,
                         annotation_text=f"卖{i+1}" if i == 0 else "")
        
        y_min = min(buy_grids[0] if buy_grids else lower_limit, lower_limit * 0.99)
        y_max = max(sell_grids[-1] if sell_grids else upper_limit, upper_limit * 1.01)
        
        fig.update_yaxes(range=[y_min, y_max])
        fig.update_layout(
            title="网格分布图",
            yaxis_title="价格（港元）",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 保存网格到会话状态
    st.session_state.buy_grids = buy_grids
    st.session_state.sell_grids = sell_grids

def render_tab_backtest():
    st.subheader("📈 策略回测结果")
    
    if not all(key in st.session_state for key in ["minute_data", "buy_grids", "sell_grids"]):
        st.warning("⚠️ 请先在【策略】标签页生成网格")
        return
    
    if st.button("🚀 开始回测", type="primary"):
        with st.spinner("正在执行回测，请稍候..."):
            result = backtest_intraday_strategy_improved(
                principal=st.session_state.principal,
                current_price=st.session_state.current_price,
                buy_grids=st.session_state.buy_grids,
                sell_grids=st.session_state.sell_grids,
                minute_data=st.session_state.minute_data,
                cfg=st.session_state.cfg
            )
            st.session_state.backtest_result = result
            st.success("✅ 回测完成！")
    
    if "backtest_result" in st.session_state and st.session_state.backtest_result:
        res = st.session_state.backtest_result
        
        # 关键指标展示
        st.subheader("📊 回测概览")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("最终净值", f"HK${res['final_total_value']:,}")
        with col2:
            profit_color = "normal" if res['total_profit'] >= 0 else "inverse"
            st.metric("总收益", f"HK${res['total_profit']:,}", 
                     f"{res['profit_rate']:.2f}%", delta_color=profit_color)
        with col3:
            st.metric("总交易次数", res['total_buy_count'] + res['total_sell_count'])
        with col4:
            st.metric("最大回撤", f"{res['max_drawdown']:.2f}%")
        
        # 风险指标
        st.subheader("🛡️ 风险指标")
        metrics = res.get("metrics", {})
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("夏普比率", f"{metrics.get('sharpe', 'N/A')}")
        mcol2.metric("卡尔玛比率", f"{metrics.get('calmar', 'N/A')}")
        mcol3.metric("胜率", f"{metrics.get('win_rate', 'N/A')}%")
        mcol4.metric("盈亏比", f"{metrics.get('profit_factor', 'N/A')}")
        
        # 净值曲线
        st.subheader("📈 净值曲线")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['timestamps'], y=res['net_values'], 
                               name='净值', line=dict(color='blue', width=2)))
        fig.update_layout(
            title='日内净值变化',
            xaxis_title='时间',
            yaxis_title='净值（港元）',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 持仓变化
        st.subheader("📦 持仓变化")
        fig_holding = go.Figure()
        fig_holding.add_trace(go.Scatter(x=res['timestamps'], y=res['holdings_history'],
                                       name='持仓数量', line=dict(color='orange')))
        fig_holding.update_layout(
            title='持仓数量变化',
            xaxis_title='时间',
            yaxis_title='持仓数量',
            height=300
        )
        st.plotly_chart(fig_holding, use_container_width=True)
        
        # 交易记录
        st.subheader("📋 交易记录")
        if res['trade_records']:
            trade_df = pd.DataFrame(res['trade_records'])
            
            # 添加总价值列
            def compute_total_after(row):
                try:
                    return round(row['cash_after'] + row['holding_after'] * row['price'], 2)
                except:
                    return None
                    
            trade_df['总价值'] = trade_df.apply(compute_total_after, axis=1)
            st.dataframe(trade_df, height=400, use_container_width=True)
            
            # CSV导出
            csv = trade_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载交易记录 CSV",
                data=csv,
                file_name="trade_records.csv",
                mime="text/csv"
            )
        else:
            st.info("📝 未产生交易记录")

def render_tab_sensitivity():
    st.subheader("🔬 网格参数敏感性分析")
    st.write("通过调整网格数量和间距，查看对回测结果的影响")
    
    if not st.session_state.get("minute_data"):
        st.warning("⚠️ 请先获取或生成分钟数据")
        return
    
    # 基础参数设置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_spacing = st.number_input("基准间距(%)", 0.1, 2.0, 0.3, 0.05)
    with col2:
        upper_limit = st.number_input("网格上限(当前价倍数)", 1.01, 1.1, 1.05, 0.01)
    with col3:
        lower_limit = st.number_input("网格下限(当前价倍数)", 0.9, 0.99, 0.95, 0.01)
    
    if st.button("开始敏感性分析", type="primary"):
        base_params = {
            "spacing": base_spacing,
            "upper": st.session_state.current_price * upper_limit,
            "lower": st.session_state.current_price * lower_limit
        }
        
        with st.spinner("正在进行敏感性分析..."):
            results = analyze_grid_sensitivity(
                st.session_state.principal,
                st.session_state.current_price,
                st.session_state.minute_data,
                st.session_state.cfg,
                base_params
            )
        
        st.subheader("分析结果")
        st.dataframe(results, use_container_width=True)
        
        # 可视化
        st.subheader("可视化分析")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("收益 vs 网格数量", "最大回撤 vs 网格数量", 
                          "交易次数 vs 网格数量", "收益 vs 间距")
        )
        
        # 收益 vs 网格数量
        for spacing in results["间距(%)"].unique():
            subset = results[results["间距(%)"] == spacing]
            fig.add_trace(
                go.Scatter(x=subset["网格数量"], y=subset["收益(%)"], 
                          name=f"间距{spacing}%", mode="lines+markers"),
                row=1, col=1
            )
        
        # 最大回撤 vs 网格数量
        for spacing in results["间距(%)"].unique():
            subset = results[results["间距(%)"] == spacing]
            fig.add_trace(
                go.Scatter(x=subset["网格数量"], y=subset["最大回撤(%)"], 
                          name=f"间距{spacing}%", mode="lines+markers", showlegend=False),
                row=1, col=2
            )
        
        # 交易次数 vs 网格数量
        for spacing in results["间距(%)"].unique():
            subset = results[results["间距(%)"] == spacing]
            fig.add_trace(
                go.Scatter(x=subset["网格数量"], y=subset["交易次数"], 
                          name=f"间距{spacing}%", mode="lines+markers", showlegend=False),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def render_tab_etf_compare():
    st.subheader("📊 多ETF日内T+0效果对比")
    
    etf_input = st.text_area(
        "输入ETF代码（每行一个，如2800.HK）", 
        "2800.HK\n3033.HK\n2828.HK",
        help="每行输入一个ETF代码，系统会自动对比它们的日内交易效果"
    )
    
    etf_codes = [code.strip() for code in etf_input.split("\n") if code.strip()]
    
    if st.button("开始ETF对比", type="primary") and etf_codes:
        with st.spinner("正在对比多个ETF..."):
            comparison_df = compare_etfs(
                etf_codes,
                st.session_state.principal,
                st.session_state.data_interval,
                st.session_state.cfg
            )
        
        if not comparison_df.empty:
            st.subheader("对比结果")
            st.dataframe(comparison_df.sort_values("收益(%)", ascending=False), 
                        use_container_width=True)
            
            # 可视化对比
            st.subheader("可视化对比")
            fig = go.Figure(data=[
                go.Bar(name="收益(%)", x=comparison_df["ETF代码"], y=comparison_df["收益(%)"]),
                go.Bar(name="最大回撤(%)", x=comparison_df["ETF代码"], y=comparison_df["最大回撤(%)"])
            ])
            fig.update_layout(
                barmode='group',
                title='ETF性能对比',
                height=400,
                xaxis_title='ETF代码',
                yaxis_title='百分比(%)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 散点图分析
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=comparison_df["收益(%)"], 
                y=comparison_df["最大回撤(%)"],
                mode='markers+text',
                text=comparison_df["ETF代码"],
                textposition="top center",
                marker=dict(size=15)
            ))
            fig_scatter.update_layout(
                title='收益 vs 最大回撤',
                xaxis_title='收益(%)',
                yaxis_title='最大回撤(%)',
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("❌ 未能获取任何ETF的数据，请检查代码是否正确")

def render_tab_indicators():
    st.subheader("📈 趋势与指标分析")
    
    if not st.session_state.get("minute_data"):
        st.warning("⚠️ 请先在【数据】标签页获取或生成分钟数据")
        return
    
    df = pd.DataFrame(st.session_state.minute_data)
    
    # 计算技术指标
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    
    vwap = calculate_vwap(st.session_state.minute_data)
    atr_values = calculate_atr(df['high'].tolist(), df['low'].tolist(), df['close'].tolist())
    atr = atr_values[-1] if atr_values else None
    
    rsi_values = calculate_rsi(df['close'].tolist())
    current_rsi = rsi_values[-1] if rsi_values else 50
    
    # 价格和均线图表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name="收盘价", line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=df['time'], y=df['MA5'], name="MA5", line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df['time'], y=df['MA10'], name="MA10", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], name="MA20", line=dict(color='red', width=1)))
    
    if vwap:
        fig.add_hline(y=vwap, line_dash="dot", line_color="green", 
                     annotation_text=f"VWAP={vwap:.2f}")
    
    fig.update_layout(
        title="价格与移动平均线",
        height=400,
        xaxis_title='时间',
        yaxis_title='价格（港元）'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI图表
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['time'], y=rsi_values[:len(df)], name="RSI", 
                               line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
    fig_rsi.update_layout(title="RSI指标", height=300, yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # 指标提示
    st.subheader("💡 指标提示")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
            st.success("MA5 > MA20\n短期趋势偏多")
        else:
            st.warning("MA5 < MA20\n短期趋势偏空")
    
    with col2:
        if vwap:
            if df['close'].iloc[-1] > vwap:
                st.success("价格 > VWAP\n资金面偏强")
            else:
                st.warning("价格 < VWAP\n资金面偏弱")
        else:
            st.info("VWAP不可用")
    
    with col3:
        if current_rsi > 70:
            st.error(f"RSI: {current_rsi:.1f}\n超买区域")
        elif current_rsi < 30:
            st.success(f"RSI: {current_rsi:.1f}\n超卖区域")
        else:
            st.info(f"RSI: {current_rsi:.1f}\n中性区域")
    
    with col4:
        if atr:
            atr_pct = (atr / df['close'].iloc[-1]) * 100
            st.info(f"ATR: {atr:.4f}\n波动率: {atr_pct:.2f}%")

def render_tab_signals():
    st.subheader("🔔 策略信号提示")
    
    if not st.session_state.get("minute_data"):
        st.warning("⚠️ 请先在【数据】标签页获取或生成分钟数据")
        return

    last_price = st.session_state.minute_data[-1]['close']
    vwap = calculate_vwap(st.session_state.minute_data)
    buy_levels = st.session_state.get("buy_grids", [])
    sell_levels = st.session_state.get("sell_grids", [])
    
    # 计算技术指标用于信号判断
    df = pd.DataFrame(st.session_state.minute_data)
    ma5 = df['close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else last_price
    ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else last_price
    
    # 信号判断
    signals = []
    
    # 网格信号
    if buy_levels and last_price <= buy_levels[0]:
        signals.append(("🟢", f"价格已接近买入网格 {buy_levels[0]:.4f} → 可以考虑小仓位买入"))
    elif sell_levels and last_price >= sell_levels[0]:
        signals.append(("🔴", f"价格已接近卖出网格 {sell_levels[0]:.4f} → 可以考虑部分卖出"))
    else:
        signals.append(("🟡", "价格处于网格中性区间，耐心等待信号"))
    
    # 趋势信号
    if ma5 > ma20:
        signals.append(("🟢", "MA5在MA20上方 → 短期趋势偏多"))
    else:
        signals.append(("🔴", "MA5在MA20下方 → 短期趋势偏空"))
    
    # VWAP信号
    if vwap:
        if last_price > vwap:
            signals.append(("🟢", "当前价高于VWAP → 资金面偏强"))
        else:
            signals.append(("🔴", "当前价低于VWAP → 资金面偏弱"))
    
    # 显示信号
    st.subheader("实时交易信号")
    
    for emoji, signal in signals:
        if "🟢" in emoji:
            st.success(f"{emoji} {signal}")
        elif "🔴" in emoji:
            st.error(f"{emoji} {signal}")
        else:
            st.info(f"{emoji} {signal}")
    
    # 操作建议汇总
    st.subheader("💎 操作建议汇总")
    
    buy_signals = sum(1 for s in signals if "🟢" in s[0])
    sell_signals = sum(1 for s in signals if "🔴" in s[0])
    neutral_signals = sum(1 for s in signals if "🟡" in s[0])
    
    if buy_signals > sell_signals:
        st.success(f"📈 建议：偏多操作（{buy_signals}个买入信号，{sell_signals}个卖出信号）")
    elif sell_signals > buy_signals:
        st.error(f"📉 建议：偏空操作（{buy_signals}个买入信号，{sell_signals}个卖出信号）")
    else:
        st.info(f"⚖️ 建议：观望为主（{buy_signals}个买入信号，{sell_signals}个卖出信号）")

def render_tab_help():
    st.subheader("🕒 交易时间提醒")
    
    status = get_hk_trading_status()
    status_color = {
        "交易中": "green",
        "未开盘": "orange", 
        "午间休市": "orange",
        "已收盘": "red",
        "休市": "red"
    }.get(status["status"], "gray")
    
    st.markdown(f"**当前状态**: <span style='color:{status_color}; font-size: 1.2em; font-weight: bold'>{status['status']}</span>", 
                unsafe_allow_html=True)
    st.info(f"💡 {status['message']}")
    
    if status.get("next_open"):
        st.write(f"⏰ 下次开盘: {status['next_open']}")

def render_tab_guide():
    st.subheader("👨‍🏫 新手指南")
    
    st.markdown("""
    ## 🚀 快速开始指南
    
    ### 第一步：获取数据
    1. 打开【数据】标签页
    2. 点击"从雅虎财经获取当天分钟数据"获取真实数据
    3. 或点击"生成模拟数据"进行练习
    
    ### 第二步：设置网格策略
    1. 打开【策略】标签页
    2. 选择合适的网格间距方式：
       - **动态间距（基于ATR）**：根据波动率自动调整（推荐新手）
       - **固定间距（手动）**：手动设置固定间距
       - **基于波动率（Std）**：根据标准差调整
    3. 调整网格数量（12-20档适合新手）
    
    ### 第三步：执行回测
    1. 打开【回测】标签页
    2. 点击"开始回测"查看策略效果
    3. 分析净值曲线和交易记录
    
    ### 第四步：优化策略
    1. 使用【参数敏感性分析】找到最优参数
    2. 通过【多ETF对比】选择合适的产品
    3. 参考【趋势指标】和【策略信号】辅助决策
    
    ## 📊 关键参数说明
    
    ### 基础参数
    - **交易本金**: 建议50,000-200,000 HKD开始
    - **ETF代码**: 香港市场ETF代码，如2800.HK（盈富基金）
    - **当前价格**: 会自动从数据中更新
    
    ### 网格参数
    - **网格档数**: 总买卖档位数量，影响交易频率
    - **网格间距**: 每档价格间隔，影响触发频率
    - **动态网格中心**: 让网格随价格趋势移动
    
    ### 风控参数
    - **最大持仓**: 建议不超过本金的50%
    - **止损止盈**: 设置全局风险控制
    - **滑点设置**: 根据流动性调整
    
    ## 💡 新手建议
    
    1. **先用模拟数据练习**，熟悉流程后再用真实数据
    2. **从小本金开始**，逐步增加投资金额
    3. **重视风险控制**，设置合理的止损止盈
    4. **多品种对比**，选择流动性好的ETF
    5. **定期回顾**，根据回测结果优化策略
    
    ## 🎯 网格策略原理
    
    网格交易是一种均值回归策略，基本原理：
    - 在价格下跌时分批买入
    - 在价格上涨时分批卖出
    - 通过价差获取收益
    - 适合震荡市行情
    
    **优点**: 机械化操作，避免情绪影响
    **缺点**: 单边市可能亏损，需要严格风控
    """)

# ---------------------------
# 主应用
# ---------------------------

def main():
    st.set_page_config(
        page_title="ETF日内网格策略 - 专业版",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用标题和介绍
    st.title("📈 ETF日内T+0网格交易策略")
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化会话状态
    if "principal" not in st.session_state:
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
            "avg_daily_turnover": avg_daily_turnover,
            "minute_data": [],
            "buy_grids": [],
            "sell_grids": [],
            "backtest_result": None
        })
    else:
        # 更新参数
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
    
    # 标签页配置
    tabs = st.tabs([
        "📊 数据", "🎯 策略", "📈 回测", "🔬 参数分析", 
        "📊 ETF对比", "📈 趋势指标", "🔔 策略信号", 
        "🕒 交易时间", "👨‍🏫 新手指南"
    ])
    
    with tabs[0]:
        render_tab_data()
    with tabs[1]:
        render_tab_strategy()
    with tabs[2]:
        render_tab_backtest()
    with tabs[3]:
        render_tab_sensitivity()
    with tabs[4]:
        render_tab_etf_compare()
    with tabs[5]:
        render_tab_indicators()
    with tabs[6]:
        render_tab_signals()
    with tabs[7]:
        render_tab_help()
    with tabs[8]:
        render_tab_guide()
    
    # 页脚信息
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>ETF日内网格交易策略系统 | 适合港股ETF T+0交易 | 数据来源: Yahoo Finance</p>
    <p>⚠️ 风险提示: 本系统仅供学习参考，实际交易请谨慎决策</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()