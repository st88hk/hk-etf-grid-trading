# ETF日内网格策略 - 增强专业版
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytz
import math
import ta  # 新增技术指标库

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
# 增强技术指标计算
# ---------------------------

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    if len(prices) < slow:
        return None, None, None
    
    df = pd.DataFrame({'close': prices})
    df['EMA_fast'] = df['close'].ewm(span=fast).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    return df['MACD'].iloc[-1], df['MACD_signal'].iloc[-1], df['MACD_histogram'].iloc[-1]

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """计算布林带"""
    if len(prices) < window:
        return None, None, None
    
    df = pd.DataFrame({'close': prices})
    df['MA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['Upper'] = df['MA'] + (df['STD'] * num_std)
    df['Lower'] = df['MA'] - (df['STD'] * num_std)
    
    return df['Upper'].iloc[-1], df['MA'].iloc[-1], df['Lower'].iloc[-1]

def calculate_ichimoku_cloud(highs, lows, closes, tenkan=9, kijun=26, senkou=52):
    """计算一目均衡表（Ichimoku Cloud）"""
    if len(closes) < senkou:
        return None
    
    high = pd.Series(highs)
    low = pd.Series(lows)
    close = pd.Series(closes)
    
    # 转换线
    tenkan_high = high.rolling(tenkan).max()
    tenkan_low = low.rolling(tenkan).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # 基准线
    kijun_high = high.rolling(kijun).max()
    kijun_low = low.rolling(kijun).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # 先行带A
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # 先行带B
    senkou_high = high.rolling(senkou).max()
    senkou_low = low.rolling(senkou).min()
    senkou_b = ((senkou_high + senkou_low) / 2).shift(kijun)
    
    # 延迟线
    chikou_span = close.shift(-kijun)
    
    return {
        'tenkan': tenkan_sen.iloc[-1],
        'kijun': kijun_sen.iloc[-1],
        'senkou_a': senkou_a.iloc[-1],
        'senkou_b': senkou_b.iloc[-1],
        'chikou': chikou_span.iloc[-kijun] if len(closes) >= kijun*2 else None
    }

def calculate_fibonacci_levels(high, low):
    """计算斐波那契回撤水平"""
    diff = high - low
    return {
        '0.0%': high,
        '23.6%': high - diff * 0.236,
        '38.2%': high - diff * 0.382,
        '50.0%': high - diff * 0.5,
        '61.8%': high - diff * 0.618,
        '78.6%': high - diff * 0.786,
        '100.0%': low
    }

def calculate_support_resistance(prices, window=20):
    """自动计算支撑阻力位"""
    if len(prices) < window:
        return [], []
    
    df = pd.DataFrame({'price': prices})
    
    # 使用滚动窗口识别局部极值点
    df['max'] = df['price'].rolling(window, center=True).max()
    df['min'] = df['price'].rolling(window, center=True).min()
    
    resistance = df[df['price'] == df['max']]['price'].unique()
    support = df[df['price'] == df['min']]['price'].unique()
    
    # 取最重要的几个水平
    resistance = sorted(resistance, reverse=True)[:5]
    support = sorted(support)[:5]
    
    return support, resistance

# ---------------------------
# 机器学习辅助预测（简化版）
# ---------------------------

def calculate_price_trend(minute_data, lookback_periods=[5, 10, 20]):
    """计算价格趋势强度"""
    if len(minute_data) < max(lookback_periods):
        return 0
    
    closes = [d['close'] for d in minute_data]
    current_price = closes[-1]
    
    trend_strength = 0
    for period in lookback_periods:
        if len(closes) >= period:
            past_price = closes[-period]
            change_pct = (current_price - past_price) / past_price * 100
            # 权重随周期增加而减小
            weight = 1.0 / period
            trend_strength += change_pct * weight
    
    return trend_strength

def predict_next_movement(minute_data, method='simple'):
    """简单预测下一期价格运动"""
    if len(minute_data) < 10:
        return 0, 0.5
    
    closes = [d['close'] for d in minute_data]
    volumes = [d['volume'] for d in minute_data]
    
    # 简单移动平均趋势
    ma_short = sum(closes[-5:]) / 5
    ma_long = sum(closes[-10:]) / 10
    ma_trend = 1 if ma_short > ma_long else -1
    
    # 价格动量
    momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
    
    # 成交量变化
    volume_trend = (volumes[-1] - sum(volumes[-5:-1])/4) / (sum(volumes[-5:-1])/4) * 100
    
    # 综合预测
    if method == 'simple':
        direction = 1 if momentum > 0 else -1
        confidence = min(abs(momentum) / 2, 0.8)  # 置信度基于动量大小
        
    return direction, confidence

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

def calculate_obv(prices, volumes):
    """计算能量潮(OBV)"""
    if len(prices) < 2:
        return [0]
    
    obv = [0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    return obv

# ---------------------------
# 智能网格生成函数
# ---------------------------

def generate_intraday_grid_arithmetic(current_price, spacing_pct, grid_count, grid_upper, grid_lower, 
                                    center_moving=False, center_price=None, volatility_mode=False, 
                                    minute_data=None, trend_adjustment=False):
    """生成智能日内网格"""
    if center_moving and center_price is not None:
        base = center_price
    else:
        base = current_price
        
    # 趋势调整因子
    trend_factor = 1.0
    if trend_adjustment and minute_data and len(minute_data) > 10:
        closes = [d['close'] for d in minute_data]
        short_ma = sum(closes[-5:]) / 5
        long_ma = sum(closes[-10:]) / 10
        if short_ma > long_ma:  # 上升趋势
            trend_factor = 0.8  # 收紧买入网格
        else:  # 下降趋势
            trend_factor = 1.2  # 放宽买入网格
        
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
    buy = [round(base * (1 - effective_spacing * (i + 1) * trend_factor), 6) for i in range(half)]
    sell = [round(base * (1 + effective_spacing * (i + 1)), 6) for i in range(half)]
    
    buy = [p for p in buy if p >= grid_lower]
    sell = [p for p in sell if p <= grid_upper]
    
    buy.sort(); sell.sort()
    return buy, sell

def generate_adaptive_grid(current_price, minute_data, grid_count=16, method='volatility'):
    """自适应网格生成"""
    if not minute_data or len(minute_data) < 10:
        return generate_intraday_grid_arithmetic(current_price, 0.3, grid_count, 
                                               current_price*1.05, current_price*0.95)
    
    closes = [d['close'] for d in minute_data]
    volumes = [d['volume'] for d in minute_data]
    
    if method == 'volatility':
        # 基于波动率的网格
        volatility = np.std(closes) / current_price * 100
        spacing_pct = max(0.1, min(1.0, volatility * 0.5))
        
    elif method == 'volume_weighted':
        # 基于成交量的网格
        avg_volume = np.mean(volumes)
        recent_volume = volumes[-1] if volumes else avg_volume
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        spacing_pct = 0.3 * min(2.0, max(0.5, volume_ratio))
        
    elif method == 'trend_following':
        # 趋势跟随网格
        trend = calculate_price_trend(minute_data)
        if trend > 0.5:  # 强上升趋势
            spacing_pct = 0.2  # 收紧网格
        elif trend < -0.5:  # 强下降趋势
            spacing_pct = 0.4  # 放宽网格
        else:
            spacing_pct = 0.3  # 中性
            
    else:
        spacing_pct = 0.3
    
    return generate_intraday_grid_arithmetic(current_price, spacing_pct, grid_count,
                                           current_price*1.05, current_price*0.95)

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
    
    # 时间段分析
    am_trades = 0; pm_trades = 0; am_pnl = 0.0; pm_pnl = 0.0
    for r in trade_records:
        t = r.get('time','')
        if ':' in t:
            hh = int(t.split(':')[0])
            if 9 <= hh <= 11:
                if r['type'].startswith('sell'):
                    am_trades += 1
                    am_pnl += (r.get('amount',0) - r.get('cost',0))
            elif 13 <= hh <= 16:
                if r['type'].startswith('sell'):
                    pm_trades += 1
                    pm_pnl += (r.get('amount',0) - r.get('cost',0))
    
    metrics['am_trades'] = am_trades
    metrics['pm_trades'] = pm_trades
    metrics['am_pnl'] = round(am_pnl, 2)
    metrics['pm_pnl'] = round(pm_pnl, 2)
    
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
    
    # 新增风控参数
    trailing_stop_pct = cfg.get("trailing_stop_pct", None)
    time_based_exit = cfg.get("time_based_exit", None)
    volatility_filter = cfg.get("volatility_filter", None)

    initial_net = principal
    peak_net = principal  # 用于跟踪止损
    total_trades_today = 0
    realized_pnl = 0.0
    buy_queue = []
    
    # 波动率过滤
    if volatility_filter and len(minute_data) > 10:
        closes = [d['close'] for d in minute_data[:10]]  # 使用前10个数据点计算初始波动率
        initial_volatility = np.std(closes) / np.mean(closes) if np.mean(closes) > 0 else 0

    for i, row in enumerate(minute_data):
        t = row["time"]
        high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
        
        # 波动率过滤
        if volatility_filter and i >= 10:
            recent_closes = [d['close'] for d in minute_data[max(0, i-10):i]]
            current_volatility = np.std(recent_closes) / np.mean(recent_closes) if np.mean(recent_closes) > 0 else 0
            if current_volatility > initial_volatility * volatility_filter:
                # 波动率过高，跳过交易
                holdings_value = shares * close
                net_value = cash + holdings_value
                timestamps.append(t)
                net_values.append(round(net_value, 4))
                holdings_history.append(shares)
                continue
        
        triggered = True
        while triggered:
            triggered = False
            for bp in list(buy_list):
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
                    if cash < (buy_amount + cost):
                        continue
                    shares += buy_shares
                    cash -= (buy_amount + cost)
                    total_trades_today += 1
                    buy_queue.append({'price': bp, 'shares': buy_shares, 'cost': cost})
                    trade_records.append({
                        "time": t, "type": "buy", "price": bp, "shares": buy_shares,
                        "amount": round(buy_amount, 2), "cost": round(cost, 2),
                        "cash_after": round(cash, 2), "holding_after": shares
                    })
                    buy_list.remove(bp)
                    triggered = True
                    break
            if triggered:
                continue
                
            for sp in list(reversed(sell_list)):
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
                    total_trades_today += 1
                    remaining = sell_shares
                    realized_this = 0.0
                    while remaining > 0 and buy_queue:
                        lot = buy_queue[0]
                        if lot['shares'] <= remaining:
                            matched = lot['shares']
                            realized_this += matched * (sp - lot['price'])
                            remaining -= matched
                            buy_queue.pop(0)
                        else:
                            matched = remaining
                            realized_this += matched * (sp - lot['price'])
                            lot['shares'] -= matched
                            remaining = 0
                    realized_pnl += realized_this
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
        
        # 更新峰值净值（用于跟踪止损）
        if net_value > peak_net:
            peak_net = net_value
            
        # 跟踪止损
        if trailing_stop_pct and net_value < peak_net * (1 - trailing_stop_pct / 100.0):
            if shares >= shares_per_lot:
                sell_shares = int(shares // shares_per_lot) * shares_per_lot
                sell_amount = sell_shares * close
                cost = calculate_trade_cost_simple(sell_amount, cfg, side='sell')
                shares -= sell_shares
                cash += (sell_amount - cost)
                trade_records.append({
                    "time": t, "type": "trailing_stop_sell", "price": close, "shares": sell_shares,
                    "amount": round(sell_amount, 2), "cost": round(cost, 2),
                    "cash_after": round(cash, 2), "holding_after": shares
                })
            break
            
        # 时间止损
        if time_based_exit and ':' in t:
            hour = int(t.split(':')[0])
            if hour >= time_based_exit and shares > 0:
                sell_shares = int(shares // shares_per_lot) * shares_per_lot
                if sell_shares > 0:
                    sell_amount = sell_shares * close
                    cost = calculate_trade_cost_simple(sell_amount, cfg, side='sell')
                    shares -= sell_shares
                    cash += (sell_amount - cost)
                    trade_records.append({
                        "time": t, "type": "time_exit_sell", "price": close, "shares": sell_shares,
                        "amount": round(sell_amount, 2), "cost": round(cost, 2),
                        "cash_after": round(cash, 2), "holding_after": shares
                    })
                break

        timestamps.append(t)
        net_values.append(round(net_value, 4))
        holdings_history.append(shares)
        
        # 固定止损检查
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
            
        # 固定止盈检查
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

    # 收盘强制平仓
    if cfg.get("force_close_end_of_day", False) and shares > 0:
        last_close = minute_data[-1]['close'] if minute_data else current_price
        sell_shares = int(shares // shares_per_lot) * shares_per_lot
        if sell_shares > 0:
            sell_amount = sell_shares * last_close
            cost = calculate_trade_cost_simple(sell_amount, cfg, side='sell')
            shares -= sell_shares
            cash += (sell_amount - cost)
            trade_records.append({
                "time": minute_data[-1]['time'] if minute_data else '', 
                "type": "eod_sell", "price": last_close, "shares": sell_shares,
                "amount": round(sell_amount, 2), "cost": round(cost, 2),
                "cash_after": round(cash, 2), "holding_after": shares
            })

    final_total = cash + shares * (minute_data[-1]['close'] if minute_data else current_price)
    total_profit = final_total - principal
    profit_rate = (total_profit / principal) * 100 if principal != 0 else 0
    
    total_buy_count = sum(1 for r in trade_records if r['type']=='buy')
    total_sell_count = sum(1 for r in trade_records if r['type'].startswith('sell'))
    avg_trade_profit = (total_profit / (total_buy_count + total_sell_count)) if (total_buy_count + total_sell_count) > 0 else 0
    max_drawdown = calculate_max_drawdown_from_series(net_values)
    
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
# 侧边栏参数设置（优化版）
# ---------------------------

def render_sidebar():
    st.sidebar.header("🎯 参数与风控设置")
    
    # 基本信息
    st.sidebar.subheader("📋 基本信息")
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
    slippage_pct = st.sidebar.slider("滑点（%）", min_value=0.01, max_value=2.0, value=float(rec_mid), step=0.01,
                                   help="成交价格偏离预期估计，高流动性0.03%-0.3%，低流动性更高")
    
    if st.sidebar.button("应用建议滑点"):
        slippage_pct = rec_mid

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 网格与数据周期")
    
    data_interval = st.sidebar.selectbox("数据周期（分钟）", [1, 5, 15], index=1)
    
    # 高级网格选项
    grid_type = st.sidebar.selectbox("网格策略类型", 
                                   ["标准网格", "动态间距（基于ATR）", "基于波动率（Std）", "自适应网格", "趋势调整网格"],
                                   help="选择不同的网格生成策略")
    
    grid_count = st.sidebar.slider("网格总档数（买+卖）", 8, 40, 16, 2,
                                 help="网格总档数越高越密集，交易越频繁。新手推荐 12-20")
    
    # 动态参数
    fixed_spacing_pct = None
    volatility_multiplier = None
    adaptive_method = None
    
    if grid_type == "标准网格":
        fixed_spacing_pct = st.sidebar.slider("固定间距（%）", 0.1, 1.0, 0.3, 0.05)
    elif grid_type == "基于波动率（Std）":
        volatility_multiplier = st.sidebar.slider("波动率间距倍数", 0.1, 2.0, 0.5, 0.1)
    elif grid_type == "自适应网格":
        adaptive_method = st.sidebar.selectbox("自适应方法", ["volatility", "volume_weighted", "trend_following"])
    
    dynamic_grid_center = st.sidebar.checkbox("动态网格中心（随VWAP/均线移动）", value=False)
    trend_adjustment = st.sidebar.checkbox("趋势调整网格间距", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ 仓位与风控（重要）")
    
    # 使用滑块优化输入体验
    initial_cash_pct = st.sidebar.slider("初始可用现金占本金（%）", 10, 100, 50, 5,
                                       help="初始用于交易的现金占本金比例。默认50%")
    initial_cash_pct = initial_cash_pct / 100.0
        
    single_trade_pct = st.sidebar.slider("单次交易金额占本金（%）", 1, 20, 5, 1,
                                       help="单笔委托最大占本金比例。推荐 2-10%，新手 5%")
    single_trade_pct = single_trade_pct / 100.0
        
    # 修复每手股数输入 - 使用数字输入框，步长为100
    shares_per_lot = st.sidebar.number_input("每手股数", min_value=1, max_value=10000, value=100, step=100,
                                           help="香港市场通常一手100股（ETF通常100）。请根据具体ETF调整")
        
    max_position_pct = st.sidebar.slider("最大持仓占本金（%）", 10, 100, 50, 5,
                                       help="单日最大可持仓占本金比例，防止单边风险。新手建议 30%-50%")
    max_position_pct = max_position_pct / 100.0

    # 高级风控参数
    st.sidebar.markdown("**🎯 止损止盈设置**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        stop_loss_pct = st.slider("止损（%）", 0.0, 10.0, 0.0, 0.5,
                                help="当回测净值较初始下跌超过该阈值时触发平仓保护")
        stop_loss_pct = stop_loss_pct if stop_loss_pct > 0 else None
        
    with col2:
        take_profit_pct = st.slider("止盈（%）", 0.0, 20.0, 0.0, 0.5,
                                  help="当回测净值较初始上涨超过该阈值时触发止盈平仓")
        take_profit_pct = take_profit_pct if take_profit_pct > 0 else None

    # 新增高级风控
    st.sidebar.markdown("**⚡ 高级风控选项**")
    
    trailing_stop_pct = st.sidebar.slider("跟踪止损（%）", 0.0, 5.0, 0.0, 0.1,
                                        help="从最高点回撤该百分比时触发止损")
    trailing_stop_pct = trailing_stop_pct if trailing_stop_pct > 0 else None
    
    time_based_exit = st.sidebar.slider("时间止损（小时）", 0, 16, 0,
                                      help="在指定时间强制平仓（0为不启用）")
    time_based_exit = time_based_exit if time_based_exit > 0 else None
    
    volatility_filter = st.sidebar.slider("波动率过滤倍数", 1.0, 3.0, 1.0, 0.1,
                                        help="当波动率超过初始值倍数时暂停交易")
    volatility_filter = volatility_filter if volatility_filter > 1.0 else None

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 扩展费用 & 限制")
    
    stamp_duty_pct = st.sidebar.slider("印花税（卖出，%）", 0.0, 1.0, 0.0, 0.01,
                                     help="卖出时适用的印花税百分比，如适用请填写（ETF通常为0）")
    
    financing_interest_pct = st.sidebar.slider("融资利息年化（%）", 0.0, 10.0, 0.0, 0.1,
                                            help="若回测需要考虑融资融券利息，可在此输入年化利率")
    
    max_daily_trades = st.sidebar.slider("单日最大交易次数", 0, 100, 0, 5,
                                       help="限制每天最大交易笔数以防过度交易。0 表示不限制")
    max_daily_trades = max_daily_trades if max_daily_trades > 0 else None
    
    single_max_loss_pct = st.sidebar.slider("单日最大亏损阈值（%）", 0.0, 10.0, 0.0, 0.5,
                                          help="当日已实现亏损超过此阈值则强制清仓")
    single_max_loss_pct = single_max_loss_pct if single_max_loss_pct > 0 else None
    
    force_close_end_of_day = st.sidebar.checkbox("收盘强制清仓（只做日内）", value=False)

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
        "trailing_stop_pct": trailing_stop_pct,
        "time_based_exit": time_based_exit,
        "volatility_filter": volatility_filter,
        "stamp_duty_pct": stamp_duty_pct,
        "financing_interest_pct": financing_interest_pct,
        "max_daily_trades": max_daily_trades,
        "single_max_loss_pct": single_max_loss_pct,
        "force_close_end_of_day": force_close_end_of_day,
        "grid_type": grid_type,
        "volatility_multiplier": volatility_multiplier,
        "adaptive_method": adaptive_method,
        "dynamic_grid_center": dynamic_grid_center,
        "trend_adjustment": trend_adjustment,
        "data_interval": data_interval,
    }
    
    return principal, etf_code, current_price, cfg, data_interval, grid_type, grid_count, fixed_spacing_pct, avg_daily_turnover

# ---------------------------
# 标签页实现
# ---------------------------

def render_tab_data():
    st.subheader("📊 分钟数据管理")
    
    if "minute_data" not in st.session_state:
        st.session_state.minute_data = []
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.write("**数据来源：雅虎财经**")
        
    with col2:
        if st.button("🔄 从雅虎财经获取当天每隔5分钟数据", type="primary"):
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
        st.warning("请先在数据标签页获取数据")
        return
    
    current_price = st.session_state.current_price
    minute_data = st.session_state.minute_data
    
    # 网格生成
    if st.session_state.grid_type == "标准网格":
        buy_grids, sell_grids = generate_intraday_grid_arithmetic(
            current_price, 
            st.session_state.fixed_spacing_pct,
            st.session_state.grid_count,
            current_price * 1.05,
            current_price * 0.95
        )
    elif st.session_state.grid_type == "自适应网格":
        buy_grids, sell_grids = generate_adaptive_grid(
            current_price,
            minute_data,
            st.session_state.grid_count,
            st.session_state.cfg.get("adaptive_method", "volatility")
        )
    else:
        # 默认网格
        buy_grids, sell_grids = generate_intraday_grid_arithmetic(
            current_price, 0.3, st.session_state.grid_count,
            current_price * 1.05, current_price * 0.95
        )
    
    st.session_state.buy_grids = buy_grids
    st.session_state.sell_grids = sell_grids
    
    # 显示网格
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**买入网格**")
        if buy_grids:
            for i, price in enumerate(buy_grids):
                st.write(f"{i+1}. {price:.4f} ({((current_price - price)/current_price*100):.2f}%)")
        else:
            st.write("无买入网格")
    
    with col2:
        st.write("**卖出网格**")
        if sell_grids:
            for i, price in enumerate(sell_grids):
                st.write(f"{i+1}. {price:.4f} ({((price - current_price)/current_price*100):.2f}%)")
        else:
            st.write("无卖出网格")
    
    # 网格统计信息
    if buy_grids and sell_grids:
        st.subheader("📈 网格统计信息")
        
        avg_buy_spacing = np.mean([current_price - p for p in buy_grids]) / current_price * 100 if buy_grids else 0
        avg_sell_spacing = np.mean([p - current_price for p in sell_grids]) / current_price * 100 if sell_grids else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("买入档数", len(buy_grids))
        col2.metric("卖出档数", len(sell_grids))
        col3.metric("平均买入间距", f"{avg_buy_spacing:.2f}%")
        col4.metric("平均卖出间距", f"{avg_sell_spacing:.2f}%")
        
        # 网格覆盖范围
        coverage_low = min(buy_grids) if buy_grids else current_price
        coverage_high = max(sell_grids) if sell_grids else current_price
        coverage_pct = (coverage_high - coverage_low) / current_price * 100
        
        st.info(f"网格覆盖范围: {coverage_low:.4f} - {coverage_high:.4f} (±{coverage_pct/2:.2f}%)")
    
    # 网格可视化 - 修复版本
    st.subheader("📊 网格价格分布图")

    if buy_grids or sell_grids:
        # 创建价格区间
        min_price = min(buy_grids) if buy_grids else current_price * 0.95
        max_price = max(sell_grids) if sell_grids else current_price * 1.05
        price_range = max_price - min_price
        
        # 扩展范围以便更好显示
        y_min = min_price - price_range * 0.1
        y_max = max_price + price_range * 0.1
        
        fig = go.Figure()
        
        # 当前价格线
        fig.add_hline(y=current_price, line_dash="dash", line_color="red", 
                     annotation_text=f"当前价格: {current_price:.4f}", 
                     annotation_position="top left")
        
        # 买入网格线（绿色）
        for i, price in enumerate(buy_grids):
            fig.add_hline(y=price, line_color="green", line_width=2,
                         annotation_text=f"买入{i+1}: {price:.4f}", 
                         annotation_position="bottom left")
        
        # 卖出网格线（蓝色）
        for i, price in enumerate(sell_grids):
            fig.add_hline(y=price, line_color="blue", line_width=2,
                         annotation_text=f"卖出{i+1}: {price:.4f}", 
                         annotation_position="top right")
        
        # 添加一些虚拟数据点以确保图表正确显示
        fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[y_min, y_max], 
            mode='markers',
            marker=dict(size=0.1, opacity=0),  # 不可见点，只是为了设置范围
            showlegend=False
        ))
        
        fig.update_layout(
            title="网格价格分布",
            xaxis=dict(showticklabels=False, title=""),  # 隐藏x轴
            yaxis_title="价格",
            showlegend=False,
            height=500,
            yaxis=dict(range=[y_min, y_max])  # 设置y轴范围
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("无法生成网格价格分布图")

def render_tab_backtest():
    st.subheader("📈 策略回测")
    
    if not st.session_state.get("minute_data") or not st.session_state.get("buy_grids"):
        st.warning("请先获取数据并生成网格")
        return
    
    if st.button("开始回测"):
        with st.spinner("回测中..."):
            result = backtest_intraday_strategy_improved(
                st.session_state.principal,
                st.session_state.current_price,
                st.session_state.buy_grids,
                st.session_state.sell_grids,
                st.session_state.minute_data,
                st.session_state.cfg
            )
            
            st.session_state.backtest_result = result
            
            # 显示回测结果
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("最终净值", f"{result['final_total_value']:,.2f}")
                st.metric("总利润", f"{result['total_profit']:,.2f}")
            
            with col2:
                st.metric("收益率", f"{result['profit_rate']:.2f}%")
                st.metric("最大回撤", f"{result['max_drawdown']:.2f}%")
            
            with col3:
                st.metric("买入次数", result['total_buy_count'])
                st.metric("卖出次数", result['total_sell_count'])
            
            # 风险指标
            if result['metrics']:
                st.subheader("📊 风险指标")
                metrics = result['metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                if metrics.get('sharpe') is not None:
                    col1.metric("夏普比率", f"{metrics['sharpe']:.2f}")
                if metrics.get('calmar') is not None:
                    col2.metric("卡尔玛比率", f"{metrics['calmar']:.2f}")
                if metrics.get('win_rate') is not None:
                    col3.metric("胜率", f"{metrics['win_rate']:.1f}%")
                if metrics.get('profit_factor') is not None:
                    col4.metric("盈亏比", f"{metrics['profit_factor']:.2f}")
            
            # 净值曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result['timestamps'], 
                y=result['net_values'],
                mode='lines',
                name='净值'
            ))
            fig.update_layout(
                title="净值曲线",
                xaxis_title="时间",
                yaxis_title="净值"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易记录
            st.subheader("交易记录")
            if result['trade_records']:
                trades_df = pd.DataFrame(result['trade_records'])
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.write("无交易记录")

def render_tab_advanced_analysis():
    st.subheader("🔬 高级技术分析")
    
    if not st.session_state.get("minute_data"):
        st.warning("⚠️ 请先在【数据】标签页获取或生成分钟数据")
        return
    
    df = pd.DataFrame(st.session_state.minute_data)
    closes = df['close'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    volumes = df['volume'].tolist()
    
    # MACD分析
    st.subheader("📊 MACD指标")
    macd, signal, histogram = calculate_macd(closes)
    
    if macd is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("MACD", f"{macd:.4f}")
        col2.metric("信号线", f"{signal:.4f}")
        col3.metric("柱状图", f"{histogram:.4f}")
        
        # MACD图表
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['time'], y=closes, name="价格", line=dict(color='black')))
        fig_macd.update_layout(height=300, title="价格走势与MACD分析")
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # 布林带分析
    st.subheader("📈 布林带分析")
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(closes)
    
    if upper_bb is not None:
        current_price = closes[-1]
        bb_position = (current_price - lower_bb) / (upper_bb - lower_bb) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("上轨", f"{upper_bb:.4f}")
        col2.metric("中轨", f"{middle_bb:.4f}")
        col3.metric("下轨", f"{lower_bb:.4f}")
        col4.metric("位置%", f"{bb_position:.1f}%")
        
        # 布林带图表
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['time'], y=df['close'], name="收盘价", line=dict(color='black')))
        fig_bb.add_trace(go.Scatter(x=df['time'], y=[upper_bb] * len(df), name="上轨", line=dict(color='red', dash='dash')))
        fig_bb.add_trace(go.Scatter(x=df['time'], y=[middle_bb] * len(df), name="中轨", line=dict(color='blue', dash='dash')))
        fig_bb.add_trace(go.Scatter(x=df['time'], y=[lower_bb] * len(df), name="下轨", line=dict(color='green', dash='dash')))
        fig_bb.update_layout(height=400, title="布林带")
        st.plotly_chart(fig_bb, use_container_width=True)
    
    # 一目均衡表分析
    st.subheader("☁️ 一目均衡表（Ichimoku Cloud）")
    ichimoku = calculate_ichimoku_cloud(highs, lows, closes)

    if ichimoku is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("转换线", f"{ichimoku['tenkan']:.4f}")
        col2.metric("基准线", f"{ichimoku['kijun']:.4f}")
        col3.metric("先行带A", f"{ichimoku['senkou_a']:.4f}")
        col4.metric("先行带B", f"{ichimoku['senkou_b']:.4f}")
        col5.metric("延迟线", f"{ichimoku['chikou']:.4f}" if ichimoku['chikou'] else "N/A")
        
        # 云图分析
        current_price = closes[-1]
        if ichimoku['senkou_a'] and ichimoku['senkou_b']:
            if current_price > max(ichimoku['senkou_a'], ichimoku['senkou_b']):
                st.success("📈 价格在云层之上 - 强势信号")
            elif current_price < min(ichimoku['senkou_a'], ichimoku['senkou_b']):
                st.error("📉 价格在云层之下 - 弱势信号")
            else:
                st.warning("☁️ 价格在云层之中 - 震荡行情")

    # ATR波动率分析
    st.subheader("📊 ATR波动率分析")
    atr_values = calculate_atr(highs, lows, closes)
    if atr_values:
        current_atr = atr_values[-1]
        atr_percent = (current_atr / current_price) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("ATR(14)", f"{current_atr:.4f}")
        col2.metric("ATR%", f"{atr_percent:.2f}%")
        
        # ATR图表
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df['time'], y=atr_values, name="ATR", line=dict(color='purple')))
        fig_atr.update_layout(height=300, title="ATR波动率")
        st.plotly_chart(fig_atr, use_container_width=True)

    # OBV能量潮分析
    st.subheader("🌊 OBV能量潮")
    obv_values = calculate_obv(closes, volumes)
    if obv_values:
        current_obv = obv_values[-1]
    
        # 修复趋势判断逻辑
        if len(obv_values) > 5:
            obv_trend = "上升" if current_obv > obv_values[-5] else "下降"
        else:
            obv_trend = "数据不足"
    
        st.metric("OBV", f"{current_obv:,.0f}", obv_trend)
    
        # OBV图表
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=df['time'], y=obv_values, name="OBV", line=dict(color='orange')))
        fig_obv.update_layout(height=300, title="OBV能量潮")
        st.plotly_chart(fig_obv, use_container_width=True)
    
    # 斐波那契回撤
    st.subheader("🔺 斐波那契回撤水平")
    if len(highs) > 0 and len(lows) > 0:
        recent_high = max(highs[-20:])  # 最近20期最高价
        recent_low = min(lows[-20:])    # 最近20期最低价
        fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
        
        fib_df = pd.DataFrame(list(fib_levels.items()), columns=['水平', '价格'])
        st.dataframe(fib_df, use_container_width=True)
        
        # 斐波那契图表
        fig_fib = go.Figure()
        fig_fib.add_trace(go.Scatter(x=df['time'], y=df['close'], name="收盘价"))
        for level, price in fib_levels.items():
            fig_fib.add_hline(y=price, line_dash="dash", annotation_text=level)
        fig_fib.update_layout(height=400, title="斐波那契回撤水平")
        st.plotly_chart(fig_fib, use_container_width=True)
    
    # 支撑阻力分析
    st.subheader("⚖️ 自动支撑阻力分析")
    support_levels, resistance_levels = calculate_support_resistance(closes)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**支撑位**")
        for level in support_levels[:3]:  # 只显示前3个
            st.write(f"- {level:.4f}")
    
    with col2:
        st.write("**阻力位**")
        for level in resistance_levels[:3]:  # 只显示前3个
            st.write(f"- {level:.4f}")
    
    # 价格预测
    st.subheader("🔮 简单价格预测")
    direction, confidence = predict_next_movement(st.session_state.minute_data)
    
    if direction > 0:
        st.success(f"预测方向: 📈 上涨 | 置信度: {confidence*100:.1f}%")
    else:
        st.error(f"预测方向: 📉 下跌 | 置信度: {confidence*100:.1f}%")
    
    # 趋势强度分析
    trend_strength = calculate_price_trend(st.session_state.minute_data)
    st.metric("趋势强度", f"{trend_strength:.2f}")

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
    st.subheader("📈 趋势指标分析")
    
    if not st.session_state.get("minute_data"):
        st.warning("请先获取数据")
        return
    
    df = pd.DataFrame(st.session_state.minute_data)
    closes = df['close'].tolist()
    volumes = df['volume'].tolist()
    
    # RSI计算
    rsi_values = calculate_rsi(closes)
    current_rsi = rsi_values[-1] if rsi_values else 50
    
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{current_rsi:.1f}")
    
    # 趋势强度
    trend_strength = calculate_price_trend(st.session_state.minute_data)
    col2.metric("趋势强度", f"{trend_strength:.2f}")
    
    # VWAP
    vwap = calculate_vwap(st.session_state.minute_data)
    if vwap:
        col3.metric("VWAP", f"{vwap:.4f}")
    
    # 价格和RSI图表
    fig = make_subplots(rows=2, cols=1, subplot_titles=('价格走势', 'RSI指标'))
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name='价格'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=rsi_values, name='RSI'), row=2, col=1)
    
    # 添加RSI超买超卖线
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

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
    st.subheader("🕒 港股交易时间")
    
    st.write("""
    ### 港股交易时间安排
    
    **正常交易日（周一至周五）**:
    - 上午盘: 09:30 - 12:00
    - 午间休市: 12:00 - 13:00  
    - 下午盘: 13:00 - 16:00
    
    **注意事项**:
    - 周六、周日及公众假期休市
    - 开盘前竞价时段: 09:00-09:20
    - 收盘竞价时段: 16:00-16:10
    
    **ETF交易特点**:
    - T+0交易：当日可多次买卖
    - 无涨跌幅限制
    - 交易费用相对较低
    """)
    
    # 显示当前交易状态
    status = get_hk_trading_status()
    st.info(f"当前状态: {status['status']} - {status['message']}")

def render_tab_guide():
    st.subheader("👨‍🏫 新手指南")
    
    st.write("""
    ### ETF网格交易策略指南
    
    **什么是网格交易？**
    网格交易是一种在价格波动中获利的策略，通过在不同价格水平设置买入和卖出订单，
    在价格波动时自动执行低买高卖的操作。
    
    **操作步骤**:
    1. **设置基本参数**: 在侧边栏设置本金、ETF代码、当前价格
    2. **获取数据**: 在"数据"标签页获取实时或模拟数据
    3. **生成网格**: 在"策略"标签页查看生成的买卖网格
    4. **回测验证**: 在"回测"标签页测试策略效果
    5. **分析优化**: 使用其他标签页进行深入分析
    
    **风险提示**:
    - 网格策略在单边行情中可能表现不佳
    - 实际交易前请充分回测验证
    - 投资有风险，入市需谨慎
    """)
    
    st.success("💡 提示: 新手建议从模拟数据开始，熟悉策略后再使用真实数据")

# ---------------------------
# 主应用
# ---------------------------

def main():
    st.set_page_config(
        page_title="ETF日内网格策略 - 增强专业版",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用标题和介绍
    st.title("📈 ETF日内T+0网格交易策略 - 增强专业版")
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
        "📊 数据", "🎯 策略", "📈 回测", "🔬 高级分析", "📋 参数分析", 
        "📊 ETF对比", "📈 趋势指标", "🔔 策略信号", "🕒 交易时间", "👨‍🏫 新手指南"
    ])
    
    with tabs[0]:
        render_tab_data()
    with tabs[1]:
        render_tab_strategy()
    with tabs[2]:
        render_tab_backtest()
    with tabs[3]:
        render_tab_advanced_analysis()
    with tabs[4]:
        render_tab_sensitivity()
    with tabs[5]:
        render_tab_etf_compare()
    with tabs[6]:
        render_tab_indicators()
    with tabs[7]:
        render_tab_signals()
    with tabs[8]:
        render_tab_help()
    with tabs[9]:
        render_tab_guide()
    
    # 页脚信息
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>ETF日内网格交易策略系统 - 增强专业版 | 适合港股ETF T+0交易 | 数据来源: Yahoo Finance</p>
    <p>⚠️ 风险提示: 本系统仅供学习参考，实际交易请谨慎决策</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
