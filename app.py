"""
MBA Finance: Bull vs. Bear AI Investment Committee Debate
Single-file Streamlit app powered by Google Gemini (google-genai) + yfinance.
"""

import io
import math
import numbers
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL = "gemini-2.5-flash"
MAX_DOC_CHARS = 60_000

st.set_page_config(
    page_title="Bull vs. Bear Investment Committee",
    page_icon="⚖️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
BULL_SYSTEM = (
    "You are the Bull Analyst on an MBA-level Investment Committee. Your job is to make "
    "the strongest possible, rigorous, data-backed BUY case. Ground every claim in the "
    "supplied live market metrics and the uploaded document (quote the figure, or say "
    "'per the uploaded notes'). Never invent financial figures that were not provided. "
    "If you rely on general business knowledge, label it clearly as such. If a metric is "
    "N/A, say so instead of guessing. Be persuasive but intellectually honest."
)

BEAR_SYSTEM = (
    "You are the Bear Analyst on an MBA-level Investment Committee. Your job is to "
    "rigorously rebut the Bull Analyst and expose the real risks: stretched valuation, "
    "leverage and debt burden, margin pressure, weak or low-quality free cash flow, "
    "growth deceleration, and anything else the data reveals. Respond directly to the "
    "Bull's specific claims. Ground every point in the supplied live metrics and the "
    "uploaded document. Never invent financial figures that were not provided. If you "
    "rely on general knowledge, label it clearly. If a metric is N/A, say so."
)

JUDGE_SYSTEM = (
    "You are the Portfolio Manager chairing an MBA-level Investment Committee. You are "
    "impartial and decisive. Weigh the Bull and Bear arguments against the hard data, "
    "call out weak or unsupported claims on either side, and deliver a clear, "
    "actionable verdict. Never invent financial figures that were not provided."
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _num(x):
    """Return a finite float, or None for missing/non-numeric/NaN/inf values."""
    if isinstance(x, bool):
        return None
    if isinstance(x, numbers.Real):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None
    return None


def currency_symbol(cur: str) -> str:
    symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}
    if not cur:
        return ""
    return symbols.get(cur, f"{cur} ")


def fmt_large(v, cur: str) -> str:
    if v is None:
        return "N/A"
    sign = "-" if v < 0 else ""
    a = abs(v)
    sym = currency_symbol(cur)
    if cur == "INR":
        cr = a / 1e7
        if cr >= 1e5:
            return f"{sign}{sym}{cr / 1e5:,.2f} Lakh Cr"
        return f"{sign}{sym}{cr:,.0f} Cr"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{sign}{sym}{a / div:,.2f}{suffix}"
    return f"{sign}{sym}{a:,.0f}"


def fmt_ratio(v) -> str:
    return "N/A" if v is None else f"{v:,.2f}"


def fmt_pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=600, show_spinner=False)
def fetch_metrics(ticker: str) -> dict:
    """Fetch live fundamentals from Yahoo Finance. Raises ValueError if ticker is invalid."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    price = (
        _num(info.get("currentPrice"))
        or _num(info.get("regularMarketPrice"))
        or _num(info.get("previousClose"))
    )
    if price is None:
        try:
            price = _num(t.fast_info["last_price"])
        except Exception:
            price = None
    if price is None:
        raise ValueError(
            f"No market data found for '{ticker}'. Check the symbol "
            "(Indian NSE stocks need the .NS suffix, e.g. RELIANCE.NS)."
        )

    fcf = _num(info.get("freeCashflow"))
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
                series = cf.loc["Free Cash Flow"].dropna()
                if not series.empty:
                    fcf = _num(series.iloc[0])
        except Exception:
            fcf = None

    summary = (info.get("longBusinessSummary") or "").strip()
    if len(summary) > 900:
        summary = summary[:900].rsplit(" ", 1)[0] + "…"

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "N/A",
        "industry": info.get("industry") or "N/A",
        "exchange": info.get("fullExchangeName") or info.get("exchange") or "N/A",
        "currency": info.get("currency") or "",
        "summary": summary,
        "price": price,
        "prev_close": _num(info.get("previousClose")),
        "market_cap": _num(info.get("marketCap")),
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "revenue_growth": _num(info.get("revenueGrowth")),
        "profit_margins": _num(info.get("profitMargins")),
        "debt_to_equity": _num(info.get("debtToEquity")),
        "free_cash_flow": fcf,
    }


def format_metrics(m: dict) -> list:
    """Return list of (label, display_value, delta_or_None, help_text)."""
    cur = m["currency"]
    sym = currency_symbol(cur)

    delta = None
    if m["price"] is not None and m["prev_close"]:
        delta = f"{(m['price'] / m['prev_close'] - 1) * 100:+.2f}% vs prev. close"

    de = m["debt_to_equity"]
    de_display = "N/A" if de is None else f"{de / 100:.2f}x"

    return [
        ("Current Price", f"{sym}{m['price']:,.2f}", delta, "Latest price from Yahoo Finance."),
        ("Market Cap", fmt_large(m["market_cap"], cur), None, "Total equity market value."),
        ("Trailing P/E", fmt_ratio(m["trailing_pe"]), None, "Price / last-12-month earnings."),
        ("Forward P/E", fmt_ratio(m["forward_pe"]), None, "Price / next-12-month expected earnings."),
        ("Revenue Growth", fmt_pct(m["revenue_growth"]), None, "Year-over-year revenue growth (most recent quarter)."),
        ("Profit Margin", fmt_pct(m["profit_margins"]), None, "Net income / revenue."),
        ("Debt-to-Equity", de_display, None, "Total debt / shareholder equity (Yahoo reports this as a percentage; shown here as a multiple)."),
        ("Free Cash Flow", fmt_large(m["free_cash_flow"], cur), None, "Operating cash flow minus capex."),
    ]


def extract_document(uploaded) -> tuple:
    """Return (text, display_name) from an uploaded PDF or CSV. Raises on unreadable files."""
    if uploaded is None:
        return "", ""

    name = uploaded.name
    raw = uploaded.getvalue()

    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise ValueError("The PDF is password-protected.")
        parts = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                parts.append(f"[Page {i}]\n{txt.strip()}")
        text = "\n\n".join(parts)
        if not text.strip():
            raise ValueError("No extractable text found in the PDF (it may be a scanned image).")
    else:
        df = None
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            text = raw.decode("utf-8", errors="replace")
        else:
            text = (
                f"CSV with {len(df)} rows x {len(df.columns)} columns.\n\n"
                + df.to_csv(index=False)
            )

    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS] + "\n\n[... document truncated for length ...]"
    return text, name


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #
def call_gemini(client, system_instruction: str, prompt: str, temperature: float = 0.7, retries: int = 3) -> str:
    """Call Gemini with simple retry/backoff on transient errors."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
            )
            text = (resp.text or "").strip()
            if text:
                return text
            last_err = RuntimeError(
                "Gemini returned an empty response (it may have been blocked by safety filters)."
            )
        except genai_errors.APIError as e:
            last_err = e
            if getattr(e, "code", None) not in (429, 500, 502, 503, 504):
                raise
        if attempt < retries - 1:
            time.sleep(2 * (2 ** attempt))
    raise last_err


def build_data_block(m: dict, doc_text: str, doc_name: str) -> str:
    rows = format_metrics(m)
    metric_lines = "\n".join(f"- {label}: {value}" for label, value, _, _ in rows)
    doc_section = (
        f"UPLOADED DOCUMENT ({doc_name}):\n\"\"\"\n{doc_text}\n\"\"\""
        if doc_text
        else "UPLOADED DOCUMENT: None provided."
    )
    return (
        f"COMPANY: {m['name']} (ticker: {m['ticker']})\n"
        f"Sector: {m['sector']} | Industry: {m['industry']} | Exchange: {m['exchange']} | "
        f"Currency: {m['currency'] or 'N/A'}\n"
        f"Business summary: {m['summary'] or 'N/A'}\n\n"
        f"LIVE MARKET METRICS (Yahoo Finance, as of {datetime.now().strftime('%Y-%m-%d %H:%M')}):\n"
        f"{metric_lines}\n\n"
        f"{doc_section}"
    )


def bull_prompt(data_block: str, ticker: str) -> str:
    return f"""{data_block}

TASK: Present the strongest data-backed BUY case for {ticker} to the Investment Committee.

Format your answer in Markdown, 350-500 words, using exactly this structure:
### 🐂 Bull Thesis: <one-line headline>
**1. <Pillar title>** - evidence with specific numbers from the data above
**2. <Pillar title>** - evidence with specific numbers
**3. <Pillar title>** - evidence with specific numbers
**4. Valuation Justification** - why the current multiples are reasonable or attractive
**Catalysts (next 12-24 months):** 2-3 bullet points
**What must go right:** one short sentence
"""


def bear_prompt(data_block: str, ticker: str, bull_text: str) -> str:
    return f"""{data_block}

THE BULL ANALYST'S ARGUMENT:
\"\"\"
{bull_text}
\"\"\"

TASK: Rebut the Bull Analyst's case for {ticker} and make the Bear case.

Format your answer in Markdown, 350-500 words, using exactly this structure:
### 🐻 Bear Rebuttal: <one-line headline>
**Rebuttal to the Bull's key claims:** directly challenge 2-3 of the Bull's specific points, with numbers.
**Valuation Risk:** analyze Trailing P/E, Forward P/E and what is priced in.
**Balance Sheet / Debt Risk:** analyze Debt-to-Equity and Free Cash Flow quality.
**Margin & Growth Risk:** analyze profit margins and revenue growth sustainability.
**What the Bull is ignoring:** one short paragraph.
**Downside trigger:** one short sentence on what would break the thesis.
"""


def judge_prompt(data_block: str, ticker: str, bull_text: str, bear_text: str) -> str:
    return f"""{data_block}

BULL ANALYST'S CASE:
\"\"\"
{bull_text}
\"\"\"

BEAR ANALYST'S REBUTTAL:
\"\"\"
{bear_text}
\"\"\"

TASK: As Portfolio Manager, weigh both sides and issue the final verdict on {ticker}.

Format your answer in Markdown using exactly this structure (the first line MUST follow the
"Conviction Score" format shown, with a single integer from 1 to 10, where 1 = strongest
conviction to avoid/sell and 10 = strongest conviction to buy):

**Conviction Score: X/10** - <one-line meaning of this score>

### ⚖️ Debate Scorecard
- **Valuation:** who argued better and why (1-2 sentences)
- **Balance Sheet & Cash Flow:** who argued better and why
- **Growth & Margins:** who argued better and why

### 📈 Outlook Scenarios
- 🟢 **Bull Case (xx% probability):** what happens and what drives it
- 🟡 **Base Case (xx% probability):** what happens and what drives it
- 🔴 **Bear Case (xx% probability):** what happens and what drives it
(The three probabilities must sum to 100%.)

### 🧭 Recommendation
One short paragraph: your stance (e.g., Accumulate / Hold / Avoid), position-sizing guidance, and key condition for changing your mind.

### 🎯 #1 Metric to Watch
**<Metric name>** - why it matters most for this stock, its current reading from the data, and what level or direction would confirm the bull vs. bear case.

Stay under 450 words. Do not invent figures that were not provided.
"""


# --------------------------------------------------------------------------- #
# Output layer
# --------------------------------------------------------------------------- #
def render_metrics(m: dict) -> None:
    st.subheader(f"{m['name']} ({m['ticker']})")
    st.caption(f"{m['sector']} · {m['industry']} · {m['exchange']} · Currency: {m['currency'] or 'N/A'}")

    rows = format_metrics(m)
    for start in (0, 4):
        cols = st.columns(4)
        for col, (label, value, delta, help_text) in zip(cols, rows[start:start + 4]):
            with col:
                with st.container(border=True):
                    st.metric(label, value, delta=delta, help=help_text)


def extract_conviction(judge_text: str):
    match = re.search(r"Conviction Score[^0-9]{0,12}(\d{1,2}(?:\.\d)?)", judge_text, re.IGNORECASE)
    if not match:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None
    return max(1.0, min(10.0, score))


def build_memo(r: dict) -> str:
    m = r["metrics"]
    rows = format_metrics(m)
    table = "\n".join(f"| {label} | {value} |" for label, value, _, _ in rows)
    doc_line = (
        f"Supporting document analyzed: `{r['doc_name']}`" if r["doc_name"] else "Supporting document: none"
    )
    return f"""# Investment Committee Memo: {m['name']} ({m['ticker']})

*Generated: {r['generated_at']} · Model: {MODEL}*
*{m['sector']} · {m['industry']} · {m['exchange']}*

> **Disclaimer:** AI-generated content for educational purposes only. This is not investment advice.

## Live Market Snapshot

| Metric | Value |
|---|---|
{table}

{doc_line}

---

## 🐂 Bull Analyst

{r['bull']}

---

## 🐻 Bear Analyst

{r['bear']}

---

## ⚖️ Portfolio Manager's Verdict

{r['judge']}
"""


def render_result(r: dict) -> None:
    render_metrics(r["metrics"])
    if r["doc_name"]:
        st.caption(f"📎 Supporting document analyzed: {r['doc_name']}")

    st.divider()
    st.header("🗣️ The Debate")
    col_bull, col_bear = st.columns(2, gap="large")
    with col_bull:
        with st.container(border=True):
            st.markdown("## 🐂 Bull Analyst")
            st.markdown(r["bull"])
    with col_bear:
        with st.container(border=True):
            st.markdown("## 🐻 Bear Analyst")
            st.markdown(r["bear"])

    st.divider()
    st.header("⚖️ Portfolio Manager's Verdict")
    with st.container(border=True):
        score = extract_conviction(r["judge"])
        if score is not None:
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Conviction Score", f"{score:g}/10")
            with c2:
                st.write("")
                st.progress(score / 10.0)
        st.markdown(r["judge"])

    st.download_button(
        label="📥 Download Memo (.md)",
        data=build_memo(r),
        file_name=f"{r['metrics']['ticker'].replace('.', '_')}_investment_memo_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
        type="primary",
    )
    st.caption("AI-generated for educational purposes only. Not investment advice.")


def render_intro() -> None:
    st.info(
        "👈 Enter a ticker in the sidebar (optionally upload a PDF/CSV of financial notes) and click "
        "**Run Investment Committee Debate**."
    )
    st.markdown(
        """
**How the committee works**
1. **Live data**: key fundamentals are pulled from Yahoo Finance.
2. **🐂 Bull Analyst** builds the strongest data-backed buy case.
3. **🐻 Bear Analyst** rebuts the Bull and highlights valuation, debt and margin risks.
4. **⚖️ Portfolio Manager** weighs both sides, issues a Conviction Score (1-10), a Bull/Base/Bear outlook, and names the #1 metric to watch.
        """
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_debate(api_key: str, ticker: str, uploaded) -> None:
    st.session_state.pop("result", None)

    if not api_key:
        st.error("Please provide your Gemini API key in the sidebar (or add GEMINI_API_KEY to Streamlit secrets).")
        return
    if not ticker:
        st.error("Please enter a stock ticker.")
        return

    # 1) Live metrics
    try:
        with st.spinner(f"Fetching live metrics for {ticker}…"):
            metrics = fetch_metrics(ticker)
    except Exception as e:
        st.error(f"Could not fetch market data for **{ticker}**: {e}")
        return

    render_metrics(metrics)

    # 2) Optional uploaded document
    doc_text, doc_name = "", ""
    if uploaded is not None:
        try:
            doc_text, doc_name = extract_document(uploaded)
        except Exception as e:
            st.warning(f"Could not read the uploaded file ({e}). Continuing without it.")

    # 3) Three sequential Gemini calls
    try:
        client = genai.Client(api_key=api_key)
        data_block = build_data_block(metrics, doc_text, doc_name)

        with st.status("Investment Committee is in session…", expanded=True) as status:
            st.write("🐂 Bull Analyst is building the buy case…")
            bull = call_gemini(client, BULL_SYSTEM, bull_prompt(data_block, ticker), temperature=0.7)
            st.write("✅ Bull case delivered.")

            st.write("🐻 Bear Analyst is preparing the rebuttal…")
            bear = call_gemini(client, BEAR_SYSTEM, bear_prompt(data_block, ticker, bull), temperature=0.7)
            st.write("✅ Bear rebuttal delivered.")

            st.write("⚖️ Portfolio Manager is deliberating…")
            judge = call_gemini(
                client, JUDGE_SYSTEM, judge_prompt(data_block, ticker, bull, bear), temperature=0.4
            )
            st.write("✅ Verdict issued.")
            status.update(label="Debate complete", state="complete", expanded=False)
    except genai_errors.APIError as e:
        st.error(f"Gemini API error: {e}")
        return
    except Exception as e:
        st.error(f"The debate could not be completed: {e}")
        return

    st.session_state["result"] = {
        "metrics": metrics,
        "bull": bull,
        "bear": bear,
        "judge": judge,
        "doc_name": doc_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.rerun()


def get_secret_key() -> str:
    try:
        key = st.secrets["GEMINI_API_KEY"]
        return str(key).strip() if key else ""
    except Exception:
        return ""


def main() -> None:
    st.title("⚖️ Bull vs. Bear: AI Investment Committee")
    st.caption("MBA Finance debate tool · Live data from Yahoo Finance · Powered by Google Gemini 2.5 Flash")

    with st.sidebar:
        st.header("⚙️ Setup")

        secret_key = get_secret_key()
        if secret_key:
            api_key = secret_key
            st.success("Gemini API key loaded from secrets.")
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey. "
                "Or set GEMINI_API_KEY in .streamlit/secrets.toml.",
            ).strip()

        ticker = (
            st.text_input(
                "Stock Ticker",
                value="AAPL",
                help="Use Yahoo Finance symbols. Indian NSE stocks use the .NS suffix "
                "(e.g., RELIANCE.NS, TCS.NS, INFY.NS); BSE stocks use .BO.",
            )
            .strip()
            .upper()
        )
        st.caption("💡 Indian NSE stocks work too, e.g. `RELIANCE.NS`.")

        uploaded = st.file_uploader(
            "Financial notes / reports (optional)",
            type=["pdf", "csv"],
            help="PDF text is extracted with pypdf; CSVs are passed to the analysts as tables.",
        )

        run_clicked = st.button(
            "Run Investment Committee Debate",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        st.caption("Educational use only. Not investment advice.")

    if run_clicked:
        run_debate(api_key, ticker, uploaded)

    result = st.session_state.get("result")
    if result:
        render_result(result)
    elif not run_clicked:
        render_intro()


main()
