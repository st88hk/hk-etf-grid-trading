import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 页面配置（适配日内交易场景）
st.set_page_config(
    page_title="日内T+0网格交易工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 1. 核心工具函数（日内交易适配）
# --------------------------
def parse_volume(volume_input):
    """解析成交量（支持万/亿/k/m单位，适配日内分钟级数据）"""
    if not volume_input or str(volume_input).strip() == "":
        return 0
            
    volume_input = str(volume_input).strip().lower()
    multipliers = {
        'k': 1000,          # 千
        'w': 10000,         # 万
        '万': 10000,        # 中文万
        'm': 1000000,       # 百万
        '兆': 1000000,      # 中文百万
        '亿': 100000000     # 亿
    }
    
    # 提取单位和数值
    unit = None
    for u in multipliers:
        if volume_input.endswith(u):
            unit = u
            num_str = volume_input[:-len(u)].strip()
            break
    if not unit:
        num_str = volume_input
    
    try:
        num = float(num_str)
        return int(round(num * multipliers.get(unit, 1)))
    except (ValueError, TypeError):
        return 0


def calculate_trade_cost(trade_amount, slippage=0.15, is_single_side=True):
    """计算日内交易成本（含滑点，适配高频交易）
    :param slippage: 滑点率（默认0.15%，日内高频典型值）
    :param is_single_side: 是否单边交易（买入/卖出单独计算）
    """
    # 滑点成本：买入价+滑点，卖出价-滑点
    slippage_cost = trade_amount * (slippage / 100)
    
    # 香港ETF手续费（日内高频场景，平台费每笔15港元）
    PLATFORM_FEE = 15  # 每笔固定平台费
    TRADE_FEE_RATE = 0.00565 / 100  # 交易佣金
    SETTLEMENT_FEE_RATE = 0.0042 / 100  # 交收费
    SFC_FEE_RATE = 0.0027 / 100  # 证监会费
    FRC_FEE_RATE = 0.00015 / 100  # 财务汇报局费
    STAMP_DUTY_RATE = 0  # ETF豁免印花税

    # 计算各项费用
    trade_fee = trade_amount * TRADE_FEE_RATE
    settlement_fee = trade_amount * SETTLEMENT_FEE_RATE
    sfc_fee = trade_amount * SFC_FEE_RATE
    frc_fee = trade_amount * FRC_FEE_RATE

    # 单边总成本（含滑点）
    single_side_total = (PLATFORM_FEE + trade_fee + settlement_fee + 
                        sfc_fee + frc_fee + slippage_cost)
    
    # 双边交易（买入+卖出）总成本
    if not is_single_side:
        return round(single_side_total * 2, 2)
    return round(single_side_total, 2)


def calculate_atr(highs, lows, closes, period=14):
    """计算平均真实波幅（ATR），用于动态网格间距"""
    atr_data = []
    for i in range(len(closes)):
        if i == 0:
            tr = highs[i] - lows[i]  # 首日TR=最高价-最低价
        else:
            # TR = max(最高价-最低价, |最高价-前收盘价|, |最低价-前收盘价|)
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr = max(tr1, tr2, tr3)
        atr_data.append(tr)
    
    # 计算ATR（滚动平均）
    atr = []
    for i in range(len(atr_data)):
        if i < period - 1:
            atr.append(None)  # 数据不足时返回None
        else:
            atr_val = np.mean(atr_data[i-period+1:i+1])
            atr.append(round(atr_val, 4))
    return atr


def calculate_intraday_kdj(highs, lows, closes, k_period=6, d_period=2):
    """计算日内专用KDJ（参数6,2,2，比默认更灵敏）"""
    kdj = []
    for i in range(len(closes)):
        if i < k_period - 1:
            kdj.append((None, None, None))  # 数据不足
        else:
            # 计算RSV（未成熟随机值）
            recent_high = max(highs[i - k_period + 1:i + 1])
            recent_low = min(lows[i - k_period + 1:i + 1])
            rsv = (closes[i] - recent_low) / (recent_high - recent_low) * 100 if recent_high != recent_low else 50
            
            # 计算K值（平滑RSV）
            if i == k_period - 1:
                k = rsv  # 初始K值=RSV
            else:
                prev_k = kdj[i-1][0]
                k = (2/3) * prev_k + (1/3) * rsv
            
            # 计算D值（平滑K值）
            if i == k_period - 1:
                d = k  # 初始D值=K值
            else:
                prev_d = kdj[i-1][1]
                d = (2/3) * prev_d + (1/3) * k
            
            j = 3 * k - 2 * d  # J值（反应最快）
            kdj.append((round(k, 2), round(d, 2), round(j, 2)))
    return kdj


def calculate_narrow_bollinger(prices, period=10, num_std=1.5):
    """计算窄幅布林带（日内专用：周期10，标准差1.5，比默认更灵敏）"""
    middle_band = []  # 中轨（MA）
    upper_band = []   # 上轨（MA + 1.5*std）
    lower_band = []   # 下轨（MA - 1.5*std）
    
    for i in range(len(prices)):
        if i < period - 1:
            middle_band.append(None)
            upper_band.append(None)
            lower_band.append(None)
        else:
            # 计算中轨（移动平均）
            ma = np.mean(prices[i-period+1:i+1])
            # 计算标准差
            std = np.std(prices[i-period+1:i+1])
            # 计算上下轨
            upper = ma + num_std * std
            lower = ma - num_std * std
            # 保留4位小数（日内价格波动小）
            middle_band.append(round(ma, 4))
            upper_band.append(round(upper, 4))
            lower_band.append(round(lower, 4))
    return upper_band, middle_band, lower_band


# --------------------------
# 2. 网格策略核心逻辑（日内T+0适配）
# --------------------------
def calculate_dynamic_grid_params(principal, current_price, minute_data, 
                                 grid_count=15, atr_period=14, kdj_period=6):
    """
    计算日内动态网格参数（基于ATR和KDJ）
    :param grid_count: 网格总档数（默认15档，日内高频建议10-20档）
    :param atr_period: ATR计算周期（默认14，日内常用10-15）
    :return: 优化后的网格参数
    """
    # 提取分钟级数据
    highs = [d['high'] for d in minute_data if d['high'] > 0]
    lows = [d['low'] for d in minute_data if d['low'] > 0]
    closes = [d['close'] for d in minute_data if d['close'] > 0]
    if len(closes) < max(atr_period, kdj_period):
        st.warning("数据不足，使用默认网格参数")
        return get_default_grid_params(principal, current_price, grid_count)
    
    # 1. 计算核心指标
    atr = calculate_atr(highs, lows, closes, atr_period)
    latest_atr = atr[-1] if atr[-1] is not None else (max(highs[-5:]) - min(lows[-5:])) / 2
    kdj = calculate_intraday_kdj(highs, lows, closes, kdj_period)
    latest_k, latest_d, _ = kdj[-1] if kdj[-1][0] is not None else (50, 50, 50)
    bollinger_upper, bollinger_mid, bollinger_lower = calculate_narrow_bollinger(closes)
    latest_bollinger_upper = bollinger_upper[-1] if bollinger_upper[-1] is not None else current_price * 1.01
    latest_bollinger_lower = bollinger_lower[-1] if bollinger_lower[-1] is not None else current_price * 0.99

    # 2. 动态网格区间（基于布林带，适配日内窄幅波动）
    grid_upper = latest_bollinger_upper * 1.005  # 上轨+0.5%缓冲
    grid_lower = latest_bollinger_lower * 0.995  # 下轨-0.5%缓冲
    # 确保区间不超过日内最大波动（默认±2%，可调整）
    grid_upper = min(grid_upper, current_price * 1.02)
    grid_lower = max(grid_lower, current_price * 0.98)

    # 3. 动态网格间距（基于ATR，确保日内触发频率）
    # 间距=ATR*0.6/当前价格（0.6为系数，越小间距越密，需>成本占比）
    base_spacing_pct = (latest_atr * 0.6 / current_price) * 100
    # 计算最小安全间距（覆盖双边成本，避免亏损）
    single_trade_amount = (principal * 0.05)  # 单次交易金额（本金5%，日内风控）
    round_trip_cost = calculate_trade_cost(single_trade_amount, is_single_side=False)
    min_safe_spacing_pct = (round_trip_cost / single_trade_amount) * 100 * 1.2  # 加20%安全垫
    # 最终间距：取动态间距和最小安全间距的较大值
    final_spacing_pct = max(base_spacing_pct, min_safe_spacing_pct, 0.2)  # 最小0.2%，避免过密

    # 4. 网格档数调整（基于KDJ超买超卖）
    final_grid_count = grid_count
    if latest_k > 75:  # 超买区，减少卖出档
        final_grid_count = max(10, grid_count - 3)
    elif latest_k < 25:  # 超卖区，减少买入档
        final_grid_count = max(10, grid_count - 3)
    # 确保档数为偶数（买入档=卖出档）
    final_grid_count = final_grid_count if final_grid_count % 2 == 0 else final_grid_count + 1

    return {
        "trend_status": "震荡" if latest_k > 30 and latest_k < 70 else "弱趋势",
        "kdj": (latest_k, latest_d),
        "atr": latest_atr,
        "grid_upper": round(grid_upper, 4),
        "grid_lower": round(grid_lower, 4),
        "spacing_pct": round(final_spacing_pct, 3),
        "grid_count": final_grid_count,
        "single_trade_amount": round(single_trade_amount, 2),
        "round_trip_cost": round_trip_cost
    }


def get_default_grid_params(principal, current_price, grid_count=15):
    """默认网格参数（数据不足时使用）"""
    # 固定区间：当前价格±1.5%（日内典型波动）
    grid_upper = current_price * 1.015
    grid_lower = current_price * 0.985
    # 固定间距：0.3%（日内常用）
    spacing_pct = 0.3
    # 单次交易金额：本金5%
    single_trade_amount = principal * 0.05
    round_trip_cost = calculate_trade_cost(single_trade_amount, is_single_side=False)
    
    return {
        "trend_status": "默认模式",
        "kdj": (50, 50),
        "atr": (current_price * 0.01) / 2,  # 估算ATR
        "grid_upper": round(grid_upper, 4),
        "grid_lower": round(grid_lower, 4),
        "spacing_pct": spacing_pct,
        "grid_count": grid_count if grid_count % 2 == 0 else grid_count + 1,
        "single_trade_amount": round(single_trade_amount, 2),
        "round_trip_cost": round_trip_cost
    }


def generate_intraday_grid(current_price, spacing_pct, grid_count, grid_upper, grid_lower):
    """生成日内网格（买入档=卖出档，适配高频交易）"""
    spacing = spacing_pct / 100  # 间距百分比转小数
    buy_grids = []  # 买入价（低于当前价）
    sell_grids = []  # 卖出价（高于当前价）

    # 生成买入档（从当前价向下，每次减间距）
    current_buy = current_price * (1 - spacing)
    for _ in range(grid_count // 2):
        if current_buy < grid_lower * 0.99:  # 不低于下轨-1%
            break
        buy_grids.append(round(current_buy, 4))
        current_buy *= (1 - spacing)
    # 买入档倒序（低价在前，便于触发）
    buy_grids = sorted(buy_grids, reverse=True)

    # 生成卖出档（从当前价向上，每次加间距）
    current_sell = current_price * (1 + spacing)
    for _ in range(grid_count // 2):
        if current_sell > grid_upper * 1.01:  # 不高于上轨+1%
            break
        sell_grids.append(round(current_sell, 4))
        current_sell *= (1 + spacing)
    # 卖出档正序（高价在后，便于触发）
    sell_grids = sorted(sell_grids)

    return buy_grids, sell_grids


def backtest_intraday_strategy(principal, current_price, buy_grids, sell_grids, minute_data):
    """日内策略回测（基于分钟级数据）"""
    trade_records = []  # 交易记录
    total_cash = principal * 0.5  # 初始现金（50%仓位，日内风控）
    total_shares = 0  # 初始持股
    shares_per_lot = 100  # 港股每手100股
    single_trade_amount = (principal * 0.05)  # 单次交易金额

    # 遍历分钟级数据，模拟交易
    for idx, data in enumerate(minute_data):
        time = data['time']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 1. 检查买入触发（价格跌破买入档）
        for buy_price in buy_grids:
            if low <= buy_price and total_cash >= single_trade_amount:
                # 计算可买股数（按100股整数倍）
                buy_shares = int((single_trade_amount / buy_price) // shares_per_lot * shares_per_lot)
                if buy_shares == 0:
                    continue
                # 计算实际成本（含滑点）
                buy_amount = buy_shares * buy_price
                cost = calculate_trade_cost(buy_amount, is_single_side=True)
                # 更新仓位和现金
                total_shares += buy_shares
                total_cash -= (buy_amount + cost)
                # 记录交易
                trade_records.append({
                    "时间": time,
                    "类型": "买入",
                    "价格(港元)": buy_price,
                    "股数": buy_shares,
                    "金额(港元)": round(buy_amount, 2),
                    "成本(港元)": round(cost, 2),
                    "剩余现金(港元)": round(total_cash, 2),
                    "持仓股数": total_shares
                })
                # 触发后移除该买入档（避免重复触发）
                buy_grids.remove(buy_price)
                break

        # 2. 检查卖出触发（价格突破卖出档）
        for sell_price in sell_grids:
            if high >= sell_price and total_shares >= shares_per_lot:
                # 计算可卖股数（按100股整数倍）
                sell_shares = min(int(total_shares // shares_per_lot * shares_per_lot), 
                                 int(single_trade_amount / sell_price) // shares_per_lot * shares_per_lot)
                if sell_shares == 0:
                    continue
                # 计算实际收益（含滑点）
                sell_amount = sell_shares * sell_price
                cost = calculate_trade_cost(sell_amount, is_single_side=True)
                # 更新仓位和现金
                total_shares -= sell_shares
                total_cash += (sell_amount - cost)
                # 记录交易
                trade_records.append({
                    "时间": time,
                    "类型": "卖出",
                    "价格(港元)": sell_price,
                    "股数": sell_shares,
                    "金额(港元)": round(sell_amount, 2),
                    "成本(港元)": round(cost, 2),
                    "剩余现金(港元)": round(total_cash, 2),
                    "持仓股数": total_shares
                })
                # 触发后移除该卖出档（避免重复触发）
                sell_grids.remove(sell_price)
                break

    # 3. 回测结果计算
    # 最终市值（现金+持仓价值）
    final_holdings_value = total_shares * current_price
    final_total_value = total_cash + final_holdings_value
    # 总收益和收益率
    total_profit = final_total_value - principal
    profit_rate = (total_profit / principal) * 100
    # 交易统计
    total_buy_count = len([r for r in trade_records if r["类型"] == "买入"])
    total_sell_count = len([r for r in trade_records if r["类型"] == "卖出"])
    avg_trade_profit = (total_profit / (total_buy_count + total_sell_count)) if (total_buy_count + total_sell_count) > 0 else 0

    return {
        "trade_records": trade_records,
        "final_total_value": round(final_total_value, 2),
        "total_profit": round(total_profit, 2),
        "profit_rate": round(profit_rate, 4),
        "total_buy_count": total_buy_count,
        "total_sell_count": total_sell_count,
        "avg_trade_profit": round(avg_trade_profit, 2),
        "max_drawdown": calculate_max_drawdown(trade_records, principal)  # 计算最大回撤
    }


def calculate_max_drawdown(trade_records, principal):
    """计算日内最大回撤（风控关键指标）"""
    if not trade_records:
        return 0.0
    # 记录每日净值变化
    net_values = [principal]
    for record in trade_records:
        # 净值=剩余现金+持仓价值（假设持仓按当前交易价计算）
        holdings_value = record["持仓股数"] * record["价格(港元)"]
        net_value = record["剩余现金(港元)"] + holdings_value
        net_values.append(net_value)
    # 计算最大回撤：(峰值-谷值)/峰值
    peak = max(net_values)
    trough = min(net_values[net_values.index(peak):])  # 峰值后的谷值
    max_drawdown = ((peak - trough) / peak) * 100
    return round(max_drawdown, 4)


# --------------------------
# 3. Streamlit界面（日内交易专用）
# --------------------------
def main():
    st.title("日内T+0网格交易策略工具")
    st.write("🔍 适配0.5%-1.5%日内波动率，支持动态网格间距，高频触发优化")
    st.divider()

    # 初始化会话状态（保存数据和参数）
    if "minute_data" not in st.session_state:
        # 生成默认分钟级数据（当日9:30-15:55，5分钟间隔）
        st.session_state.minute_data = generate_default_minute_data()
    if "grid_params" not in st.session_state:
        st.session_state.grid_params = None
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None

    # 侧边栏：参数设置（日内交易专用）
    with st.sidebar:
        st.header("1. 基础交易参数")
        # 本金设置（日内建议1-5万港元，控制风险）
        principal = st.number_input(
            "交易本金（港元）",
            min_value=10000.0,
            max_value=100000.0,
            value=30000.0,
            step=5000.0,
            help="日内交易建议1-5万，单次交易不超过本金5%"
        )
        # 交易标的（ETF代码）
        etf_code = st.text_input(
            "ETF代码（港股）",
            value="02800.HK",  # 恒生ETF示例
            help="选择日均成交额>5亿、波动率0.5%-1.5%的ETF"
        )
        # 当前价格
        current_price = st.number_input(
            f"{etf_code}当前价格（港元）",
            min_value=0.01,
            value=27.5,
            step=0.01,
            format="%.4f",  # 保留4位小数，适配日内小波动
            help="输入最新成交价，精确到0.0001港元"
        )

        st.divider()
        st.header("2. 日内网格参数")
        # 网格类型（动态/固定）
        grid_type = st.radio(
            "网格类型",
            ["动态间距（推荐）", "固定间距"],
            index=0,
            help="动态间距：基于ATR自动适配波动；固定间距：手动设置"
        )
        # 分钟级数据周期（默认5分钟，日内高频常用）
        data_interval = st.selectbox(
            "数据周期",
            [1, 5, 10, 15],
            index=1,
            help="1分钟：超高频；5分钟：平衡型（推荐）；10-15分钟：低频"
        )
        # 网格档数（默认15档）
        grid_count = st.slider(
            "网格总档数（买入档=卖出档）",
            min_value=10,
            max_value=25,
            value=15,
            step=1,
            help="日内建议10-20档，档数越多触发越频繁"
        )
        # 固定间距（仅固定模式显示）
        fixed_spacing_pct = 0.3
        if grid_type == "固定间距":
            fixed_spacing_pct = st.slider(
                "固定网格间距（%）",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f%%",
                help="日内建议0.2%-0.5%，需>双边成本占比"
            )

        st.divider()
        # 操作按钮
        col_calc, col_reset = st.columns(2)
        with col_calc:
            calculate_btn = st.button(
                "📊 计算网格策略",
                use_container_width=True,
                type="primary",
                help="基于输入数据计算网格参数并回测"
            )
        with col_reset:
            reset_btn = st.button(
                "🔄 重置数据",
                use_container_width=True,
                help="重置为默认分钟级数据和参数"
            )
            if reset_btn:
                st.session_state.minute_data = generate_default_minute_data()
                st.session_state.grid_params = None
                st.session_state.backtest_result = None
                st.success("数据已重置为默认值")

    # 主界面：分标签页
    tab1, tab2, tab3 = st.tabs(["📅 分钟级数据", "📈 网格策略", "📊 回测结果"])

    # 标签页1：分钟级数据输入
    with tab1:
        st.subheader(f"日内{data_interval}分钟数据（当日交易时段）")
        st.write("💡 提示：直接编辑表格，成交量支持1000、1k、0.1万等格式；点击【生成默认数据】快速填充")
        
        # 生成表格数据（字典列表，确保列名对应）
        table_data = []
        for data in st.session_state.minute_data:
            # 格式化成交量（万为单位，便于查看）
            vol = data['volume']
            if vol >= 10000:
                vol_str = f"{vol/10000:.2f}万"
            elif vol >= 1000:
                vol_str = f"{vol/1000:.1f}k"
            else:
                vol_str = str(vol)
            table_data.append({
                "时间": data['time'],
                "最高价(港元)": data['high'],
                "最低价(港元)": data['low'],
                "收盘价(港元)": data['close'],
                "成交量": vol_str
            })

        # 可编辑表格
        edited_table = st.data_editor(
            table_data,
            column_config={
                "时间": st.column_config.TextColumn(disabled=False, help="格式：HH:MM，如09:30"),
                "最高价(港元)": st.column_config.NumberColumn(format="%.4f", min_value=0.0001),
                "最低价(港元)": st.column_config.NumberColumn(format="%.4f", min_value=0.0001),
                "收盘价(港元)": st.column_config.NumberColumn(format="%.4f", min_value=0.0001),
                "成交量": st.column_config.TextColumn(help="支持1000、1k、0.1万等格式")
            },
            use_container_width=True,
            hide_index=True,
            key="minute_data_editor"
        )

        # 数据保存按钮
        if st.button("💾 保存数据", use_container_width=True):
            try:
                # 更新分钟级数据到会话状态
                updated_minute_data = []
                for idx, row in enumerate(edited_table):
                    # 解析时间（补全当日日期）
                    time_str = row["时间"].strip()
                    if not time_str or len(time_str.split(":")) != 2:
                        st.warning(f"第{idx+1}行时间格式错误，跳过该条数据")
                        continue
                    # 解析价格（确保合理）
                    high = float(row["最高价(港元)"])
                    low = float(row["最低价(港元)"])
                    close = float(row["收盘价(港元)"])
                    if high < low or close < low or close > high:
                        st.warning(f"第{idx+1}行价格逻辑错误（高价<低价或收盘价超区间），已自动修正")
                        high = max(high, low, close)
                        low = min(high, low, close)
                        close = max(min(close, high), low)
                    # 解析成交量
                    volume = parse_volume(row["成交量"])
                    # 添加到更新列表
                    updated_minute_data.append({
                        "time": time_str,
                        "high": round(high, 4),
                        "low": round(low, 4),
                        "close": round(close, 4),
                        "volume": volume
                    })
                # 保存更新后的数据
                st.session_state.minute_data = updated_minute_data
                st.success(f"成功保存{len(updated_minute_data)}条分钟级数据")
            except Exception as e:
                st.error(f"数据保存失败：{str(e)}")

        # 生成默认数据按钮
        if st.button("🔧 生成默认数据", use_container_width=True):
            st.session_state.minute_data = generate_default_minute_data(current_price=current_price)
            st.rerun()  # 刷新页面显示新数据

    # 标签页2：网格策略计算结果
    with tab2:
        st.subheader("网格策略参数（日内T+0优化）")
        st.write("📌 关键指标：动态间距基于ATR，确保日内触发频率；成本已含滑点")

        # 计算按钮触发后显示结果
        if calculate_btn:
            try:
                # 1. 计算网格参数
                with st.spinner("正在计算网格参数..."):
                    if grid_type == "动态间距（推荐）":
                        # 动态网格（基于ATR）
                        st.session_state.grid_params = calculate_dynamic_grid_params(
                            principal=principal,
                            current_price=current_price,
                            minute_data=st.session_state.minute_data,
                            grid_count=grid_count
                        )
                    else:
                        # 固定网格（手动设置间距）
                        grid_params = get_default_grid_params(principal, current_price, grid_count)
                        grid_params["spacing_pct"] = fixed_spacing_pct
                        grid_params["trend_status"] = "固定模式"
                        st.session_state.grid_params = grid_params

                # 2. 生成网格价格
                grid_params = st.session_state.grid_params
                buy_grids, sell_grids = generate_intraday_grid(
                    current_price=current_price,
                    spacing_pct=grid_params["spacing_pct"],
                    grid_count=grid_params["grid_count"],
                    grid_upper=grid_params["grid_upper"],
                    grid_lower=grid_params["grid_lower"]
                )
                # 保存网格到会话状态
                st.session_state.buy_grids = buy_grids
                st.session_state.sell_grids = sell_grids

                # 3. 显示策略参数（分栏布局，清晰直观）
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 基础配置")
                    st.write(f"**交易标的**：{etf_code}")
                    st.write(f"**交易本金**：{principal:,.0f}港元")
                    st.write(f"**当前价格**：{current_price:.4f}港元")
                    st.write(f"**数据周期**：{data_interval}分钟")
                    st.write(f"**网格类型**：{grid_type}")

                with col2:
                    st.markdown("### 网格核心参数")
                    st.write(f"**网格区间**：{grid_params['grid_lower']:.4f} ~ {grid_params['grid_upper']:.4f}港元")
                    st.write(f"**网格间距**：{grid_params['spacing_pct']:.3f}%")
                    st.write(f"**网格档数**：{grid_params['grid_count']}档（买入{len(buy_grids)}档/卖出{len(sell_grids)}档）")
                    st.write(f"**单次交易金额**：{grid_params['single_trade_amount']:.2f}港元")
                    st.write(f"**双边成本**：{grid_params['round_trip_cost']:.2f}港元（{grid_params['round_trip_cost']/grid_params['single_trade_amount']*100:.3f}%）")

                st.divider()
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("### 市场状态指标")
                    st.write(f"**趋势判断**：{grid_params['trend_status']}")
                    st.write(f"**KDJ（K,D）**：({grid_params['kdj'][0]}, {grid_params['kdj'][1]})")
                    st.write(f"**ATR（平均波幅）**：{grid_params['atr']:.4f}港元")
                    st.write(f"**触发条件**：价格跌破买入档/突破卖出档")

                with col4:
                    st.markdown("### 风控参数")
                    st.write(f"**最大仓位**：≤50%（日内不满仓）")
                    st.write(f"**单次风险**：≤5%本金（避免黑天鹅）")
                    st.write(f"**最小间距**：{grid_params['spacing_pct']:.3f}%（覆盖成本）")
                    st.write(f"**区间限制**：当前价±2%（避免极端行情）")

                st.divider()
                # 显示买入/卖出网格
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    st.markdown(f"### 买入网格（{len(buy_grids)}档）")
                    if buy_grids:
                        buy_df = pd.DataFrame({
                            "买入档位": [f"买{i+1}" for i in range(len(buy_grids))],
                            "买入价格(港元)": buy_grids,
                            "触发条件": ["价格≤该档价格" for _ in buy_grids]
                        })
                        st.dataframe(buy_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("未生成买入网格，请检查网格参数")

                with col_sell:
                    st.markdown(f"### 卖出网格（{len(sell_grids)}档）")
                    if sell_grids:
                        sell_df = pd.DataFrame({
                            "卖出档位": [f"卖{i+1}" for i in range(len(sell_grids))],
                            "卖出价格(港元)": sell_grids,
                            "触发条件": ["价格≥该档价格" for _ in sell_grids]
                        })
                        st.dataframe(sell_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("未生成卖出网格，请检查网格参数")

                # 回测提示
                st.divider()
                if st.button("🚀 开始日内回测", use_container_width=True, type="primary"):
                    with st.spinner("正在进行日内回测..."):
                        backtest_result = backtest_intraday_strategy(
                            principal=principal,
                            current_price=current_price,
                            buy_grids=buy_grids.copy(),
                            sell_grids=sell_grids.copy(),
                            minute_data=st.session_state.minute_data
                        )
                        st.session_state.backtest_result = backtest_result
                        # 切换到回测结果标签页
                        st.switch_page(st_pages[2])  # 需确保标签页顺序正确

            except Exception as e:
                st.error(f"策略计算失败：{str(e)}")
                st.exception(e)  # 显示详细错误信息（调试用）

        # 未点击计算按钮时显示提示
        elif not st.session_state.grid_params:
            st.info("请在左侧边栏设置参数后，点击【计算网格策略】按钮")
        # 已计算过，显示缓存结果
        else:
            # 逻辑同计算按钮触发后（复用代码）
            grid_params = st.session_state.grid_params
            buy_grids = st.session_state.get("buy_grids", [])
            sell_grids = st.session_state.get("sell_grids", [])
            # 显示参数（同计算后逻辑，此处省略重复代码，实际需完整复制）
            st.info("已加载历史计算结果，点击【计算网格策略】可更新参数")

    # 标签页3：回测结果
    with tab3:
        st.subheader("日内T+0策略回测结果")
        st.write("📊 基于输入的分钟级数据，模拟日内交易触发情况")

        # 显示回测结果
        backtest_result = st.session_state.get("backtest_result")
        if backtest_result:
            # 1. 核心收益指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("初始本金(港元)", f"{principal:,.0f}")
            with col2:
                st.metric("最终总市值(港元)", f"{backtest_result['final_total_value']:,.2f}")
            with col3:
                profit_color = "green" if backtest_result["total_profit"] > 0 else "red"
                st.metric("总收益(港元)", f"{backtest_result['total_profit']:,.2f}", 
                         f"{backtest_result['profit_rate']:.4f}%", delta_color=profit_color)
            with col4:
                drawdown_color = "red" if backtest_result["max_drawdown"] > 1 else "orange"
                st.metric("最大回撤(%)", f"{backtest_result['max_drawdown']:.4f}", 
                         delta_color=drawdown_color)

            st.divider()
            # 2. 交易统计
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("总买入次数", backtest_result["total_buy_count"])
            with col_b:
                st.metric("总卖出次数", backtest_result["total_sell_count"])
            with col_c:
                avg_color = "green" if backtest_result["avg_trade_profit"] > 0 else "red"
                st.metric("平均每笔收益(港元)", f"{backtest_result['avg_trade_profit']:.2f}", 
                         delta_color=avg_color)

            st.divider()
            # 3. 交易记录
            st.markdown("### 详细交易记录")
            trade_records = backtest_result["trade_records"]
            if trade_records:
                # 转换为DataFrame便于查看
                trade_df = pd.DataFrame(trade_records)
                # 格式化显示
                st.dataframe(
                    trade_df,
                    column_config={
                        "时间": st.column_config.TextColumn(),
                        "类型": st.column_config.TextColumn(),
                        "价格(港元)": st.column_config.NumberColumn(format="%.4f"),
                        "股数": st.column_config.NumberColumn(),
                        "金额(港元)": st.column_config.NumberColumn(format="%.2f"),
                        "成本(港元)": st.column_config.NumberColumn(format="%.2f"),
                        "剩余现金(港元)": st.column_config.NumberColumn(format="%.2f"),
                        "持仓股数": st.column_config.NumberColumn()
                    },
                    use_container_width=True,
                    hide_index=True
                )
                # 导出交易记录
                csv = trade_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "💾 下载交易记录",
                    data=csv,
                    file_name=f"日内交易记录_{etf_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("未触发任何交易，可能原因：1.网格间距过大；2.价格未触及网格；3.数据不足")

            st.divider()
            # 4. 策略建议
            st.markdown("### 日内交易建议")
            if backtest_result["profit_rate"] > 0.1:
                st.success("✅ 策略回测盈利：建议实盘小仓位试错（本金10%以内）")
            elif backtest_result["profit_rate"] >= 0:
                st.info("⚠️ 策略回测持平：建议优化网格参数（缩小间距/增加档数）")
            else:
                st.error("❌ 策略回测亏损：不建议实盘，需调整参数（如扩大间距/减少档数）")
            
            st.write("💡 实盘注意事项：")
            st.write("1. 日内交易需紧盯行情，避免尾盘单边行情")
            st.write("2. 单次交易不超过本金5%，总仓位不超过50%")
            st.write("3. 若1小时内无交易，可手动缩小间距0.05%-0.1%")
            st.write("4. 对接券商API时，需设置条件单有效期为当日")

        else:
            st.info("请先在【网格策略】标签页计算参数，再点击【开始日内回测】")

    # 底部风险提示
    st.divider()
    st.caption("""
    ⚠️ 风险提示：
    1. 日内交易风险较高，可能面临滑点扩大、流动性不足等问题
    2. 回测结果基于历史数据，不代表未来收益
    3. 实盘前需充分测试，建议从模拟交易开始
    4. 本工具不构成投资建议，交易风险自负
    """)


def generate_default_minute_data(current_price=27.5, interval=5):
    """生成默认分钟级数据（当日9:30-15:55，5分钟间隔）"""
    minute_data = []
    # 生成时间序列（9:30到15:55，5分钟间隔）
    start_time = datetime.strptime("09:30", "%H:%M")
    end_time = datetime.strptime("15:55", "%H:%M")
    current_time = start_time
    while current_time <= end_time:
        # 生成随机价格（围绕当前价±0.3%波动）
        price_offset = np.random.uniform(-0.003, 0.003)
        close_price = current_price * (1 + price_offset)
        # 最高价=收盘价+0.05%-0.1%，最低价=收盘价-0.05%-0.1%
        high_price = close_price * (1 + np.random.uniform(0.0005, 0.001))
        low_price = close_price * (1 - np.random.uniform(0.0005, 0.001))
        # 生成成交量（日内ETF典型成交量：5000-20000股/5分钟）
        volume = int(np.random.uniform(5000, 20000))
        # 添加到数据列表
        minute_data.append({
            "time": current_time.strftime("%H:%M"),
            "high": round(high_price, 4),
            "low": round(low_price, 4),
            "close": round(close_price, 4),
            "volume": volume
        })
        # 时间递增5分钟
        current_time += timedelta(minutes=interval)
    return minute_data


if __name__ == "__main__":
    # 修复Streamlit标签页切换问题（提前定义标签页顺序）
    st_pages = ["📅 分钟级数据", "📈 网格策略", "📊 回测结果"]
    main()
