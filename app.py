import streamlit as st
import numpy as np

# 页面配置
st.set_page_config(
    page_title="香港ETF网格交易策略",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 核心功能函数
# --------------------------
def parse_volume(volume_input):
    """解析成交量（支持单位）"""
    if not volume_input:
        return None
            
    volume_input = str(volume_input).strip().lower()
    multipliers = {
        'k': 1000, 'w': 10000, '万': 10000,
        'm': 1000000, '兆': 1000000, '亿': 100000000
    }
    
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
        multiplier = multipliers.get(unit, 1)
        return int(round(num * multiplier))
    except ValueError:
        return None


def calculate_trade_cost(trade_amount, is_single_side=True):
    """计算交易成本"""
    PLATFORM_FEE = 15
    SETTLEMENT_FEE_RATE = 0.0042 / 100
    TRADE_FEE_RATE = 0.00565 / 100
    SFC_FEE_RATE = 0.0027 / 100
    FRC_FEE_RATE = 0.00015 / 100
    STAMP_DUTY_RATE = 0

    settlement_fee = round(trade_amount * SETTLEMENT_FEE_RATE, 2)
    trade_fee = round(trade_amount * TRADE_FEE_RATE, 2)
    sfc_fee = round(trade_amount * SFC_FEE_RATE, 2)
    frc_fee = round(trade_amount * FRC_FEE_RATE, 2)
    
    single_side_cost = PLATFORM_FEE + settlement_fee + trade_fee + sfc_fee + frc_fee
    return round(single_side_cost * 2 if not is_single_side else single_side_cost, 2)


# 技术指标计算函数
def calculate_ma(prices, period):
    ma = []
    for i in range(len(prices)):
        if i < period - 1:
            ma.append(None)
        else:
            ma.append(round(np.mean(prices[i - period + 1:i + 1]), 2))
    return ma


def calculate_bollinger_bands(prices, period=20, num_std=2):
    middle_band = calculate_ma(prices, period)
    upper_band, lower_band = [], []
    
    for i in range(len(prices)):
        if i < period - 1:
            upper_band.append(None)
            lower_band.append(None)
        else:
            std = np.std(prices[i - period + 1:i + 1])
            upper_band.append(round(middle_band[i] + num_std * std, 2))
            lower_band.append(round(middle_band[i] - num_std * std, 2))
    
    return upper_band, middle_band, lower_band


def calculate_kdj(highs, lows, closes, period=9):
    kdj = []
    for i in range(len(closes)):
        if i < period - 1:
            kdj.append((None, None, None))
        else:
            recent_high = max(highs[i - period + 1:i + 1])
            recent_low = min(lows[i - period + 1:i + 1])
            rsv = (closes[i] - recent_low) / (recent_high - recent_low) * 100 if recent_high != recent_low else 50
            
            k = rsv if i == period - 1 else (2/3) * kdj[i-1][0] + (1/3) * rsv
            d = k if i == period - 1 else (2/3) * kdj[i-1][1] + (1/3) * k
            j = 3 * k - 2 * d
            
            kdj.append((round(k, 2), round(d, 2), round(j, 2)))
    return kdj


def calculate_vwap(highs, lows, closes, volumes):
    vwap = []
    cumulative_volume = 0
    cumulative_price_volume = 0
    
    for i in range(len(closes)):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        price_volume = typical_price * volumes[i]
        cumulative_volume += volumes[i]
        cumulative_price_volume += price_volume
        
        vwap.append(round(cumulative_price_volume / cumulative_volume, 2) if cumulative_volume else round(closes[i], 2))
    return vwap


# 网格策略核心函数
def calculate_optimal_grid_params(principal, current_price, recent_data, grid_count=10):
    highs = [d['high'] for d in recent_data]
    lows = [d['low'] for d in recent_data]
    closes = [d['close'] for d in recent_data]
    volumes = [d['volume'] for d in recent_data]
    
    ma5 = calculate_ma(closes, 5)
    ma20 = calculate_ma(closes, 20)
    upper_band, middle_band, lower_band = calculate_bollinger_bands(closes)
    kdj = calculate_kdj(highs, lows, closes)
    vwap = calculate_vwap(highs, lows, closes, volumes)
    
    latest_ma5 = ma5[-1] if ma5[-1] is not None else current_price
    latest_ma20 = ma20[-1] if ma20[-1] is not None else current_price
    latest_upper = upper_band[-1] if upper_band[-1] is not None else current_price * 1.1
    latest_lower = lower_band[-1] if lower_band[-1] is not None else current_price * 0.9
    latest_k, latest_d, _ = kdj[-1] if kdj[-1][0] is not None else (50, 50, 50)
    latest_vwap = vwap[-1] if vwap[-1] is not None else current_price
    
    # 趋势判断
    trend = "震荡"
    if latest_ma5 > latest_ma20 * 1.01 and closes[-1] > closes[-5]:
        trend = "强上涨"
    elif latest_ma5 < latest_ma20 * 0.99 and closes[-1] < closes[-5]:
        trend = "强下跌"
    elif abs(latest_ma5 - latest_ma20) / latest_ma20 < 0.01:
        trend = "弱震荡"
    
    # 网格参数优化
    grid_upper, grid_lower = latest_upper, latest_lower
    if trend == "强上涨":
        grid_upper *= 1.05
        grid_lower = current_price * 0.98
    elif trend == "强下跌":
        grid_lower *= 0.95
        grid_upper = current_price * 1.02
    
    price_range = grid_upper - grid_lower
    base_spacing = (price_range / grid_count) / current_price * 100
    
    spacing = base_spacing * 0.8 if latest_k > 80 or latest_k < 20 else base_spacing
    
    final_grid_count = max(6, grid_count // 2 * 2) if trend in ["强上涨", "强下跌"] else grid_count
    
    # 最小安全间距
    buy_grid_count = final_grid_count // 2
    single_trade_amount = (principal / 2) / buy_grid_count
    round_trip_cost = calculate_trade_cost(single_trade_amount, is_single_side=False)
    min_safe_spacing = (round_trip_cost / single_trade_amount) * 100 * 1.2
    if spacing < min_safe_spacing:
        spacing = min_safe_spacing
    
    return {
        "trend": trend,
        "grid_upper": round(grid_upper, 2),
        "grid_lower": round(grid_lower, 2),
        "spacing": round(spacing, 2),
        "grid_count": final_grid_count,
        "kdj": (latest_k, latest_d),
        "vwap": latest_vwap,
        "ma_status": "金叉" if latest_ma5 > latest_ma20 else "死叉" if latest_ma5 < latest_ma20 else "缠绕"
    }


def generate_grid_from_params(current_price, spacing_pct, grid_count, grid_upper, grid_lower):
    spacing = spacing_pct / 100
    buy_grids, sell_grids = [], []
    
    current_buy = current_price * (1 - spacing)
    for _ in range(grid_count // 2):
        if current_buy < grid_lower * 0.98:
            break
        buy_grids.append(round(current_buy, 2))
        current_buy *= (1 - spacing)
    
    current_sell = current_price * (1 + spacing)
    for _ in range(grid_count // 2):
        if current_sell > grid_upper * 1.02:
            break
        sell_grids.append(round(current_sell, 2))
        current_sell *= (1 + spacing)
    
    min_count = min(len(buy_grids), len(sell_grids))
    return buy_grids[:min_count], sell_grids[:min_count]


def simulate_trade(principal, current_price, buy_grids, sell_grids, recent_data):
    trade_records = []
    total_buy_funds = principal / 2
    initial_funds = principal / 2
    shares_per_lot = 100

    # 初始底仓
    initial_shares = int(initial_funds / current_price) // shares_per_lot * shares_per_lot
    initial_amount = round(initial_shares * current_price, 2)
    trade_records.append({
        "档位": "起始底仓", "指令": "持仓", "目标价格(港元)": current_price,
        "股数(股)": initial_shares, "成交金额(港元)": initial_amount,
        "单笔净利润(港元)": "-", "累计持仓(股)": initial_shares
    })

    # 买入档
    if buy_grids:
        buy_funds_per_grid = total_buy_funds / len(buy_grids)
        for i, buy_price in enumerate(buy_grids, 1):
            buy_shares = int(buy_funds_per_grid / buy_price) // shares_per_lot * shares_per_lot
            buy_amount = round(buy_shares * buy_price, 2)
            trade_records.append({
                "档位": f"买入{i}", "指令": "买入", "目标价格(港元)": buy_price,
                "股数(股)": buy_shares, "成交金额(港元)": buy_amount,
                "单笔净利润(港元)": "-", "累计持仓(股)": trade_records[-1]["累计持仓(股)"] + buy_shares
            })

    # 卖出档
    if sell_grids and len(trade_records) > 1:
        sell_shares_per_grid = trade_records[1]["股数(股)"]
        for i, sell_price in enumerate(sell_grids, 1):
            sell_amount = round(sell_shares_per_grid * sell_price, 2)
            buy_amount = round(sell_shares_per_grid * buy_grids[i-1], 2) if i-1 < len(buy_grids) else 0
            round_trip_cost = calculate_trade_cost(buy_amount, is_single_side=False)
            profit = round((sell_amount - buy_amount) - round_trip_cost, 2)
            trade_records.append({
                "档位": f"卖出{i}", "指令": "卖出", "目标价格(港元)": sell_price,
                "股数(股)": sell_shares_per_grid, "成交金额(港元)": sell_amount,
                "单笔净利润(港元)": profit if profit > 0 else f"亏损{abs(profit)}",
                "累计持仓(股)": trade_records[-1]["累计持仓(股)"] - sell_shares_per_grid
            })

    # 回测分析
    closes = [d['close'] for d in recent_data]
    grid_upper = max(sell_grids) if sell_grids else current_price * 1.1
    grid_lower = min(buy_grids) if buy_grids else current_price * 0.9
    break_upper_count = sum(1 for p in closes if p > grid_upper)
    break_lower_count = sum(1 for p in closes if p < grid_lower)
    trigger_count = sum(1 for p in closes for price in buy_grids + sell_grids if abs(p - price) < 0.01)

    return trade_records, {
        "break_count": break_upper_count + break_lower_count,
        "trigger_count": trigger_count,
        "grid_upper": grid_upper,
        "grid_lower": grid_lower
    }


# --------------------------
# Streamlit界面布局（稳定输入版）
# --------------------------
def main():
    st.title("香港ETF网格交易策略工具")
    st.write("基于MA、布林带、KDJ指标的智能网格策略，支持简化输入和可视化分析")
    st.divider()

    # 初始化历史数据和临时编辑数据
    if 'recent_data' not in st.session_state:
        current_price = 27.5
        st.session_state.recent_data = []
        for i in range(20):
            base_price = current_price * (1 + (i-10)*0.002)
            high = round(base_price * (1 + np.random.uniform(0.005, 0.01)), 2)
            low = round(base_price * (1 - np.random.uniform(0.005, 0.01)), 2)
            close = round((high + low) / 2 * (1 + np.random.uniform(-0.002, 0.002)), 2)
            volume = int(np.random.uniform(800000, 1200000))
            st.session_state.recent_data.append({
                "day": i + 1,
                "high": high, 
                "low": low, 
                "close": close, 
                "volume": volume
            })
    
    # 存储临时编辑的数据，避免实时刷新导致丢失
    if 'temp_edited_data' not in st.session_state:
        st.session_state.temp_edited_data = None

    # 左侧参数输入
    with st.sidebar:
        st.header("基础参数设置")
        principal = st.number_input("交易本金（港元）", value=100000.0, step=10000.0)
        etf_code = st.text_input("ETF代码", value="02800.HK")
        current_price = st.number_input("当前价格（港元）", value=27.5, step=0.1)
        
        st.divider()
        if st.button("生成默认历史数据", use_container_width=True):
            st.session_state.recent_data = []
            for i in range(20):
                base_price = current_price * (1 + (i-10)*0.002)
                high = round(base_price * (1 + np.random.uniform(0.005, 0.01)), 2)
                low = round(base_price * (1 - np.random.uniform(0.005, 0.01)), 2)
                close = round((high + low) / 2 * (1 + np.random.uniform(-0.002, 0.002)), 2)
                volume = int(np.random.uniform(800000, 1200000))
                st.session_state.recent_data.append({
                    "day": i + 1,
                    "high": high, 
                    "low": low, 
                    "close": close, 
                    "volume": volume
                })
            st.session_state.temp_edited_data = None  # 清空临时数据
            st.success("已生成默认数据")

        st.divider()
        calculate_btn = st.button("计算网格策略", use_container_width=True, type="primary")

    # 主界面：历史数据表格（添加手动保存机制）
    st.subheader("历史数据输入（最近20个交易日）")
    st.write("提示：编辑完成后点击【保存数据】按钮，避免输入丢失")

    # 创建表格数据
    table_data = []
    for item in st.session_state.recent_data:
        vol = item['volume']
        vol_str = f"{vol:,}" if vol < 10000 else f"{vol//10000}万"
        table_data.append({
            "日期": f"第{item['day']}天",
            "最高价": item['high'],
            "最低价": item['low'],
            "收盘价": item['close'],
            "成交量": vol_str
        })

    # 显示可编辑表格（使用临时数据或原始数据）
    edited_data = st.data_editor(
        st.session_state.temp_edited_data if st.session_state.temp_edited_data is not None else table_data,
        column_config={
            "日期": st.column_config.TextColumn(disabled=True),
            "最高价": st.column_config.NumberColumn(format="%.2f"),
            "最低价": st.column_config.NumberColumn(format="%.2f"),
            "收盘价": st.column_config.NumberColumn(format="%.2f"),
            "成交量": st.column_config.TextColumn()
        },
        use_container_width=True,
        hide_index=True,
        key="data_editor"  # 添加唯一key确保状态稳定
    )

    # 保存按钮：手动触发数据保存，解决回车丢失问题
    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("💾 保存数据", use_container_width=True):
            try:
                # 先保存到临时变量
                st.session_state.temp_edited_data = edited_data
                
                # 再更新到核心数据
                for i in range(len(edited_data)):
                    vol_input = edited_data[i]["成交量"]
                    volume = parse_volume(vol_input) or st.session_state.recent_data[i]['volume']
                    
                    st.session_state.recent_data[i] = {
                        "day": st.session_state.recent_data[i]['day'],
                        "high": float(edited_data[i]["最高价"]),
                        "low": float(edited_data[i]["最低价"]),
                        "close": float(edited_data[i]["收盘价"]),
                        "volume": volume
                    }
                st.success("数据保存成功！")
            except Exception as e:
                st.error(f"保存失败：{str(e)}")
    
    with col_reset:
        if st.button("🔄 重置编辑", use_container_width=True):
            st.session_state.temp_edited_data = None
            st.success("已重置为上次保存的数据")

    # 计算并显示结果
    if calculate_btn:
        with st.spinner("正在计算策略..."):
            try:
                recent_data_for_calc = [
                    {
                        "high": item['high'],
                        "low": item['low'],
                        "close": item['close'],
                        "volume": item['volume']
                    }
                    for item in st.session_state.recent_data
                ]
                
                grid_params = calculate_optimal_grid_params(principal, current_price, recent_data_for_calc)
                buy_grids, sell_grids = generate_grid_from_params(
                    current_price, 
                    grid_params['spacing'], 
                    grid_params['grid_count'],
                    grid_params['grid_upper'],
                    grid_params['grid_lower']
                )
                trade_records, backtest_result = simulate_trade(
                    principal, current_price, buy_grids, sell_grids, recent_data_for_calc
                )

                # 显示结果（与之前相同）
                st.divider()
                st.header("策略分析结果")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("核心参数")
                    st.write(f"**交易标的**：{etf_code}")
                    st.write(f"**当前价格**：{current_price}港元")
                    st.write(f"**总本金**：{principal:,.0f}港元")
                
                with col2:
                    st.subheader("网格配置")
                    st.write(f"**网格档数**：{grid_params['grid_count']}档（{grid_params['grid_count']//2}买{grid_params['grid_count']//2}卖）")
                    st.write(f"**网格间距**：{grid_params['spacing']}%")
                    st.write(f"**价格区间**：{grid_params['grid_lower']}~{grid_params['grid_upper']}港元")

                st.subheader("市场状态分析")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**趋势判断**：{grid_params['trend']}")
                    st.write(f"**MA5与MA20**：{grid_params['ma_status']}")
                with col2:
                    st.write(f"**KDJ状态**：K={grid_params['kdj'][0]}, D={grid_params['kdj'][1]}")
                    st.write(f"**成交量加权价**：{grid_params['vwap']}港元")

                total_initial = trade_records[0]["成交金额(港元)"] if trade_records else 0
                total_buy = sum(r["成交金额(港元)"] for r in trade_records if r["指令"] == "买入") if trade_records else 0
                
                st.subheader("资金分配")
                st.write(f"**初始底仓**：{total_initial:,.2f}港元（占本金{total_initial/principal*100:.1f}%）")
                st.write(f"**买入备用金**：{total_buy:,.2f}港元（占本金{total_buy/principal*100:.1f}%）")

                st.subheader("交易清单")
                if trade_records:
                    st.dataframe(trade_records, use_container_width=True)

                total_profit = sum(r["单笔净利润(港元)"] for r in trade_records 
                                 if isinstance(r["单笔净利润(港元)"], (int, float))) if trade_records else 0
                
                st.subheader("回测与操作建议")
                st.write(f"**历史触发交易**：{backtest_result['trigger_count']}次")
                st.write(f"**历史突破网格**：{backtest_result['break_count']}次")
                st.write(f"**收益预估**：{total_profit:,.2f}港元（若全部成交）")

                if grid_params['trend'] in ["强上涨", "强下跌"]:
                    st.warning("⚠️ 当前处于强趋势市，建议降低仓位至50%或暂停网格")
                elif grid_params['kdj'][0] > 80:
                    st.info("提示：KDJ超买，可优先执行卖出档交易")
                elif grid_params['kdj'][0] < 20:
                    st.info("提示：KDJ超卖，可优先执行买入档交易")
                else:
                    st.success("✅ 当前为震荡市，适合正常执行网格策略")

                st.divider()
                st.caption("""
                ⚠️ 风险提示：  
                1. 指标仅为辅助，实际交易需结合实时行情  
                2. 强趋势市下网格策略风险较高，建议严格止损  
                3. 本工具不构成投资建议，风险自负
                """)

            except Exception as e:
                st.error(f"计算出错：{str(e)}")


if __name__ == "__main__":
    main()
