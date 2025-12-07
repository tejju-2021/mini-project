import streamlit as st
import yfinance as yf
import ta
import pandas as pd
import re
from datetime import datetime

# ---------------- PAGE CONFIG + TITLE ----------------
st.set_page_config(
    page_title="AI Stock Chatbot",
    page_icon="📱",
    layout="centered"
)

# Custom Header with Centered Logo + Title
# Fixed Header Logo + Title
st.markdown("""
    <div style='padding-top: 25px; text-align: center;'>
        <span style='font-size: 65px;'>🤖📱🤖</span>
        <h1 style='margin-top: -10px; font-size: 40px;'>
            BullBear AI
        </h1>
        <p style='font-size: 20px; color: #c8c8c8;'>
            Smart Dual-Trend Stock Analyzer
        </p>
    </div>
""", unsafe_allow_html=True)



# --- Simple UI polish (card style + signal color) ---
st.markdown(
    """
    <style>
    .signal-badge {
        padding: 2px 6px;
        border-radius: 4px;
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .signal-buy { background-color: #16a34a; }   /* green */
    .signal-sell { background-color: #dc2626; }  /* red */
    .signal-hold { background-color: #ca8a04; }  /* amber */
    .chat-card {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        background-color: #1118270d;
        border: 1px solid #37415133;
        margin-bottom: 0.5rem;
    }
    /* Mobile friendly tweaks */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def format_signal_badge(text: str) -> str:
    t = text.upper()
    if "BUY" in t:
        cls = "signal-buy"
    elif "SELL" in t:
        cls = "signal-sell"
    else:
        cls = "signal-hold"
    return f'<span class="signal-badge {cls}">{text}</span>'


# ---------------- QUICK ACTIONS (GOOD FOR MOBILE) ----------------
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⭐ Best Buys", use_container_width=True):
            st.session_state.quick_action = "best"
        else:
            if "quick_action" not in st.session_state:
                st.session_state.quick_action = None
    with c2:
        if st.button("📊 Sample: Reliance", use_container_width=True):
            st.session_state.quick_action = "sample_reliance"
    with c3:
        if st.button("⚖️ Compare TCS vs Reliance", use_container_width=True):
            st.session_state.quick_action = "compare_tcs_rel"

# ---------------- COMPANY MAP + WATCHLIST ----------------
company_map = {
    # 🏦 Indian Large & Mid Caps (NSE)
    "reliance": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hcl": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "titan": "TITAN.NS",
    "airtel": "BHARTIARTL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "axis bank": "AXISBANK.NS",
    "axis": "AXISBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "kotak": "KOTAKBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj fin": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "indusind": "INDUSINDBK.NS",
    "pnb": "PNB.NS",
    "canara bank": "CANBK.NS",
    "bandhan": "BANDHANBNK.NS",
    "l&t": "LT.NS",
    "larsen": "LT.NS",
    "lt": "LT.NS",
    "adani": "ADANIENT.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "jsw steel": "JSWSTEEL.NS",
    "maruti": "MARUTI.NS",
    "mahindra": "M&M.NS",
    "m&m": "M&M.NS",
    "eicher": "EICHERMOT.NS",
    "heromoto": "HEROMOTOCO.NS",
    "power grid": "POWERGRID.NS",
    "itc": "ITC.NS",
    "asian paints": "ASIANPAINT.NS",
    "hul": "HINDUNILVR.NS",
    "ultracemco": "ULTRACEMCO.NS",
    "cipla": "CIPLA.NS",
    "dr reddy": "DRREDDY.NS",
    "sun pharma": "SUNPHARMA.NS",
    "coal india": "COALINDIA.NS",
    "ongc": "ONGC.NS",
    "ntpc": "NTPC.NS",

    # 🌎 US Tech & Popular Stocks
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "nflx": "NFLX",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
    "paypal": "PYPL",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "uber": "UBER",
    "coca cola": "KO",
    "pepsi": "PEP",
    "mcd": "MCD",
    "visa": "V",
    "mastercard": "MA",
}

watchlist = list(dict.fromkeys(company_map.values()))  # de-duplicate


# ---------------- SYMBOL RESOLUTION ----------------
def resolve_symbol(user_text: str):
    text = user_text.lower().strip()

    # 1️⃣ Check fixed name mappings first (Indian + US)
    for name, ticker in company_map.items():
        if name in text:
            return ticker

    # 2️⃣ Extract possible ticker-like words
    tokens = re.findall(r"[A-Za-z]{2,10}(?:\.[A-Za-z]{1,4})?", text)
    if not tokens:
        return None

    cand = tokens[-1].upper()

    # ❌ Prevent accidentally detecting "buy" or "best" as ticker
    if cand in ["BUY", "BEST", "STOCK", "STOCKS"]:
        return None

    # 3️⃣ Validate ticker by test-download of last few days
    test = yf.download(cand, period="5d", interval="1d", progress=False)
    if test is not None and not test.empty:
        return cand  # valid ticker

    return None  # ❌ invalid ticker like "vgjewj"


# ---------------- DATA HELPERS ----------------
def get_price_df(symbol: str, period: str, interval: str):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([c for c in col if c]) if isinstance(col, tuple) else col
            for col in data.columns
        ]

    close_candidates = [c for c in data.columns if "close" in c.lower()]
    if not close_candidates:
        return None

    close_col = close_candidates[0]
    data["Close"] = pd.to_numeric(data[close_col], errors="coerce")
    data = data.dropna(subset=["Close"])
    if data.empty:
        return None

    return data


def compute_trend(data: pd.DataFrame):
    close = data["Close"]

    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()
    data["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_calc = ta.trend.MACD(close)
    data["MACD"] = macd_calc.macd()
    data["MACD_Signal"] = macd_calc.macd_signal()

    data = data.dropna()
    if data.empty:
        return None

    latest = data.iloc[-1]

    score = 0
    if latest["RSI"] < 30:
        score += 1
    if latest["Close"] > latest["SMA50"]:
        score += 1
    if latest["Close"] > latest["SMA200"]:
        score += 1
    if latest["Close"] > latest["EMA20"]:
        score += 1
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 1

    if score >= 3:
        signal = "BUY ✅"
    elif score <= 1 or latest["RSI"] > 70:
        signal = "SELL ❌"
    else:
        signal = "HOLD 🤔"

    return {
        "price": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "SMA50": float(latest["SMA50"]),
        "SMA200": float(latest["SMA200"]),
        "EMA20": float(latest["EMA20"]),
        "MACD": float(latest["MACD"]),
        "MACD_Signal": float(latest["MACD_Signal"]),
        "signal": signal,
        "score": int(score),
        "timestamp": data.index[-1],
        "data": data,
    }


def analyze_stock_dual(symbol: str):
    try:
        daily_df = get_price_df(symbol, period="400d", interval="1d")
        hourly_df = get_price_df(symbol, period="60d", interval="1h")

        if daily_df is None and hourly_df is None:
            return None

        daily = compute_trend(daily_df) if daily_df is not None else None
        hourly = compute_trend(hourly_df) if hourly_df is not None else None

        if daily is None and hourly is None:
            return None

        combined_score = 0
        if daily:
            combined_score += daily["score"] * 2
        if hourly:
            combined_score += hourly["score"]

        if combined_score >= 7:
            combined_signal = "BUY ✅ Strong (Daily + 1H agree)"
        elif combined_score <= 2:
            combined_signal = "SELL ❌ Weak / Risky"
        else:
            combined_signal = "HOLD 🤔 Mixed / Neutral"

        latest_ts = hourly["timestamp"] if hourly else (daily["timestamp"] if daily else None)
        price = daily["price"] if daily else (hourly["price"] if hourly else None)

        return {
            "symbol": symbol,
            "price": round(price, 2) if price is not None else None,
            "daily": daily,
            "hourly": hourly,
            "combined_signal": combined_signal,
            "combined_score": combined_score,
            "timestamp": latest_ts,
        }

    except Exception as e:
        return {"error": str(e)}


def parse_compare_symbols(text: str):
    text = text.lower()
    parts = re.split(r"\bvs\b|\bversus\b|\band\b|\&|\,", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 1]

    cleaned_parts = []
    for p in parts:
        if "compare" in p:
            p = p.replace("compare", "").strip()
        if p:
            cleaned_parts.append(p)

    found = []
    for chunk in cleaned_parts:
        sym = resolve_symbol(chunk)
        if sym and sym not in found:
            found.append(sym)
        if len(found) == 2:
            break

    if len(found) == 2:
        return found[0], found[1]
    return None, None


# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = None

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT (also handle quick action shortcuts) ----------------
user_input = st.chat_input(
    "Ask about a stock, say 'best stocks to buy', 'compare TCS and Reliance', or 'should I buy now?'..."
)

# Map quick buttons to synthetic queries
if "quick_action" in st.session_state:
    if st.session_state.quick_action == "best":
        user_input = "best stocks to buy"
    elif st.session_state.quick_action == "sample_reliance":
        user_input = "analyze reliance"
    elif st.session_state.quick_action == "compare_tcs_rel":
        user_input = "compare tcs and reliance"
    st.session_state.quick_action = None  # reset

lower = user_input.lower().strip() if user_input else None


def bot_say(text: str, html: bool = False):
    st.session_state.messages.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        if html:
            st.markdown(f'<div class="chat-card">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(text)


# ---------------- MAIN LOGIC ----------------
# Greeting first
if user_input and lower in ["hi", "hello", "hey", "start", "help"]:
    bot_say(
        "👋 Hello! Welcome to AI Stock Helper Chatbot.\n\n"
        "You can ask me things like:\n"
        "• `Analyze Reliance`\n"
        "• `Analyze AAPL`\n"
        "• `Best stocks to buy`\n"
        "• `Compare TCS and Reliance`\n"
        "• `Should I buy now?`\n\n"
        "Try any option 😊"
    )
    st.session_state.stop_processing = True
else:
    st.session_state.stop_processing = False

if user_input and not st.session_state.get("stop_processing", False):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lower = user_input.lower().strip()

    # Mode 0: Follow-up
    if any(p in lower for p in ["should i buy", "good to buy", "what should i do", "buy now", "sell now"]):
        last = st.session_state.last_analysis
        sym = st.session_state.last_symbol

        if last and "error" not in last and sym:
            daily = last.get("daily")
            hourly = last.get("hourly")
            ts = last["timestamp"].strftime("%Y-%m-%d %H:%M") if last["timestamp"] is not None else "N/A"

            msg = f"📌 Last analyzed: **{sym}**\n\n"
            if daily:
                msg += (
                    "📈 **Daily (Investor View)**\n"
                    f"- Signal: **{daily['signal']}**\n"
                    f"- Price: `{round(daily['price'],2)}`\n"
                    f"- RSI: `{round(daily['RSI'],2)}`\n"
                    f"- SMA50 / SMA200: `{round(daily['SMA50'],2)}` / `{round(daily['SMA200'],2)}`\n\n"
                )
            if hourly:
                msg += (
                    "⌛ **1H (Short-Term View)**\n"
                    f"- Signal: **{hourly['signal']}**\n"
                    f"- RSI: `{round(hourly['RSI'],2)}`\n"
                    f"- EMA20: `{round(hourly['EMA20'],2)}`\n"
                    f"- MACD vs Signal: `{round(hourly['MACD'],2)}` / `{round(hourly['MACD_Signal'],2)}`\n\n"
                )

            msg += (
                f"🎯 **Overall:** {last['combined_signal']}\n"
                f"🕒 Data as of: *{ts}*\n\n"
                "⚠ This is **technical view only**, not financial advice."
            )
        else:
            msg = (
                "🤔 I don't have any recent stock analysis stored.\n\n"
                "First ask something like:\n"
                "- `Analyze TCS`\n"
                "- `Tell me about Reliance`\n"
                "- `Analyze Google`"
            )
        bot_say(msg)

    # Compare mode
    elif "compare" in lower:
        sym1, sym2 = parse_compare_symbols(lower)

        if not sym1 or not sym2:
            bot_say(
                "⚠️ I couldn't detect two valid stocks to compare.\n\n"
                "Try like:\n"
                "- `compare TCS and Reliance`\n"
                "- `compare AAPL vs MSFT`"
            )
        else:
            bot_say(f"📊 Comparing **{sym1}** and **{sym2}** (Daily + 1H)...")

            res1 = analyze_stock_dual(sym1)
            res2 = analyze_stock_dual(sym2)

            if not res1 or "error" in res1 or not res2 or "error" in res2:
                bot_say("⚠️ Unable to fetch data for one or both symbols.")
            else:
                daily1, daily2 = res1["daily"], res2["daily"]

                mode = st.selectbox(
                    "Choose comparison mode",
                    [
                        "Technical Score (Daily×2 + 1H)",
                        "Daily Trend Strength",
                        "RSI Risk (Lower RSI Better)",
                    ],
                    key="compare_mode",
                )

                winner = None
                reason = ""

                if mode.startswith("Technical"):
                    s1 = res1["combined_score"]
                    s2 = res2["combined_score"]
                    if s1 > s2:
                        winner, reason = sym1, f"higher combined score ({s1} vs {s2})"
                    elif s2 > s1:
                        winner, reason = sym2, f"higher combined score ({s2} vs {s1})"
                elif mode.startswith("Daily"):
                    s1 = daily1["score"] if daily1 else 0
                    s2 = daily2["score"] if daily2 else 0
                    if s1 > s2:
                        winner, reason = sym1, f"stronger daily trend ({s1} vs {s2})"
                    elif s2 > s1:
                        winner, reason = sym2, f"stronger daily trend ({s2} vs {s1})"
                else:
                    r1 = daily1["RSI"] if daily1 else 999
                    r2 = daily2["RSI"] if daily2 else 999
                    if r1 < r2:
                        winner, reason = sym1, f"less overbought (RSI {r1:.2f} vs {r2:.2f})"
                    elif r2 < r1:
                        winner, reason = sym2, f"less overbought (RSI {r2:.2f} vs {r1:.2f})"

                table_md = f"""
| Metric | {sym1} | {sym2} |
|--------|--------|--------|
| Price | `{res1['price']}` | `{res2['price']}` |
| Daily Signal | {daily1['signal'] if daily1 else 'N/A'} | {daily2['signal'] if daily2 else 'N/A'} |
| Daily RSI | `{round(daily1['RSI'],2) if daily1 else 'N/A'}` | `{round(daily2['RSI'],2) if daily2 else 'N/A'}` |
| Daily Score | `{daily1['score'] if daily1 else 'N/A'}` | `{daily2['score'] if daily2 else 'N/A'}` |
| Overall Signal | {res1['combined_signal']} | {res2['combined_signal']} |
| Combined Score | `{res1['combined_score']}` | `{res2['combined_score']}` |
"""
                with st.chat_message("assistant"):
                    st.markdown("### 📊 Side-by-Side Comparison")
                    st.markdown(table_md)

                    if winner:
                        st.markdown(
                            f"🏆 **Based on _{mode}_, better pick now:** **{winner}**  \n"
                            f"📝 Reason: {reason}"
                        )
                    else:
                        st.markdown("⚖️ Both look similar under this mode. No clear winner.")

    else:
        symbol = resolve_symbol(user_input)

        if symbol == "BUY":
            symbol = None

        if symbol:
            bot_say(f"📊 Analyzing **{symbol}** with Daily + 1H trends ...")
            result = analyze_stock_dual(symbol)

            if result and "error" not in result:
                st.session_state.last_analysis = result
                st.session_state.last_symbol = symbol

                daily = result["daily"]
                hourly = result["hourly"]
                ts = result["timestamp"].strftime("%Y-%m-%d %H:%M") if result["timestamp"] is not None else "N/A"

                sig_html = format_signal_badge(result["combined_signal"])

                text = (
                    f"<h3>🧾 Analysis for <b>{symbol}</b></h3>"
                    f"<p>💰 <b>Price</b>: <code>{result['price']}</code></p>"
                    f"<p>🎯 Overall Technical View: {sig_html}</p>"
                )

                if daily:
                    text += (
                        "<h4>📈 Daily Trend (Investor View)</h4>"
                        f"- Signal: <b>{daily['signal']}</b><br>"
                        f"- RSI: <code>{round(daily['RSI'],2)}</code><br>"
                        f"- SMA50 / SMA200: <code>{round(daily['SMA50'],2)}</code> / "
                        f"<code>{round(daily['SMA200'],2)}</code><br>"
                        f"- EMA20: <code>{round(daily['EMA20'],2)}</code><br><br>"
                    )
                else:
                    text += "<p>❌ No sufficient daily data.</p>"

                if hourly:
                    text += (
                        "<h4>⌛ 1H Trend (Short-Term / Swing View)</h4>"
                        f"- Signal: <b>{hourly['signal']}</b><br>"
                        f"- RSI: <code>{round(hourly['RSI'],2)}</code><br>"
                        f"- EMA20: <code>{round(hourly['EMA20'],2)}</code><br>"
                        f"- MACD / Signal: <code>{round(hourly['MACD'],2)}</code> / "
                        f"<code>{round(hourly['MACD_Signal'],2)}</code><br><br>"
                    )
                else:
                    text += "<p>❌ No sufficient 1H intraday data.</p>"

                text += (
                    f"<p>🕒 <i>As of {ts}</i></p>"
                    "<p>💡 You can now ask: <b>`Should I buy now?`</b></p>"
                )

                with st.chat_message("assistant"):
                    st.markdown(f'<div class="chat-card">{text}</div>', unsafe_allow_html=True)

                    chart_source = daily["data"] if daily else (hourly["data"] if hourly else None)
                    if chart_source is not None:
                        st.markdown("#### 📉 Price with Moving Averages")
                        chart_df = chart_source[["Close"]].copy()
                        chart_df["SMA50"] = chart_df["Close"].rolling(50).mean()
                        chart_df["SMA200"] = chart_df["Close"].rolling(200).mean()
                        st.line_chart(chart_df.tail(200))
                    else:
                        st.markdown("⚠️ Not enough data to draw chart.")

                st.session_state.messages.append({"role": "assistant", "content": f"Analysis for {symbol}"})

            else:
                bot_say(f"⚠️ Could not fetch data for `{symbol}`.")

        elif any(x in lower for x in ["best", "recommend", "suggest", "good stocks", "best stocks"]):
            greeting = "📊 Fetching latest recommendations from the watchlist (Daily trend)..."
            bot_say(greeting)

            buy_candidates = []
            result_text = "## 📊 Latest Stock Recommendations (Watchlist)\n\n"

            for stock in watchlist:
                res = analyze_stock_dual(stock)
                if res is None or "error" in res:
                    err = res.get("error", "No data") if res else "No data"
                    result_text += f"⚠️ **{stock}**: {err}\n\n"
                    continue

                daily = res["daily"]
                if not daily:
                    result_text += f"⚠️ **{stock}**: No daily data\n\n"
                    continue

                result_text += (
                    f"**{stock}** → Daily: **{daily['signal']}**, Overall: **{res['combined_signal']}**\n"
                    f"- Price: `{round(daily['price'],2)}`\n"
                    f"- RSI: `{round(daily['RSI'],2)}`\n"
                    f"- SMA50 / SMA200: `{round(daily['SMA50'],2)}` / `{round(daily['SMA200'],2)}`\n\n"
                )

                if "BUY" in daily["signal"]:
                    buy_candidates.append((stock, daily["score"], daily["RSI"]))

            if buy_candidates:
                buy_candidates.sort(key=lambda x: (-x[1], x[2]))
                best = [s[0] for s in buy_candidates]
                result_text += (
                    "\n👉 **Top BUY candidates by daily trend strength:**\n"
                    + ", ".join(best)
                )
            else:
                result_text += "\n🤔 No strong BUY signals found right now (based on daily trend)."

            bot_say(result_text)

        else:
            bot_say(
                "😅 I’m not sure what you meant.\n\n"
                "Try one of these:\n"
                "• `Analyze Google`\n"
                "• `Analyze Reliance`\n"
                "• `Best stocks to buy`\n"
                "• `Compare TCS and Reliance`\n"
                "• `Should I buy now?`\n\n"
                "I’m here to help! 😊"
            )
