import json
from pathlib import Path
import pandas as pd
import streamlit as st

# =========================
# CONFIGURATION
# =========================
RESULTS_DIR = Path("results")
DAILY_DIR = RESULTS_DIR / "daily"
RUN_LOG = RESULTS_DIR / "run_log.jsonl"
LATEST = RESULTS_DIR / "latest.csv"
DIFF_FILE = RESULTS_DIR / "daily_diff.json"

st.set_page_config(page_title="AlphaBot Quant Screener V7.7", layout="wide")
st.title("AlphaBot Quant Screener V7.7")

# =========================
# LOAD DATA
# =========================
runs = []
if RUN_LOG.exists():
    try:
        with open(RUN_LOG, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        runs.append(json.loads(line))
                    except: pass
    except: pass

run_df = pd.DataFrame(runs) if runs else pd.DataFrame()

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

# 1. Date Selection (Look for V7 first, then legacy V6)
v7_files = sorted(DAILY_DIR.glob("alpha_v7_results_*.csv"), key=lambda x: x.name)
v6_files = sorted(DAILY_DIR.glob("alpha_v6_results_*.csv"), key=lambda x: x.name)
all_files = v7_files + v6_files
# Sort all by date descending (latest first)
all_files = sorted(all_files, key=lambda x: x.name)[::-1]

date_options = ["(latest)"] + [p.name.replace("alpha_v7_results_", "").replace("alpha_v6_results_", "").replace(".csv", "") for p in all_files]
# Deduplicate dates
date_options = list(dict.fromkeys(date_options))

choice = st.sidebar.selectbox("Select date", date_options)

# 2. Filters
top_n = st.sidebar.slider("Top N", 5, 50, 20)
min_alpha = st.sidebar.slider("Min Alpha Score", 0, 100, 0)

# =========================
# HELPER FUNCTIONS
# =========================
def load_results(selected):
    if selected == "(latest)":
        if LATEST.exists():
            return pd.read_csv(LATEST)
        return None

    # Try V7 format first
    path = DAILY_DIR / f"alpha_v7_results_{selected}.csv"
    if path.exists():
        return pd.read_csv(path)
    
    # Fallback to V6 format
    path = DAILY_DIR / f"alpha_v6_results_{selected}.csv"
    if path.exists():
        return pd.read_csv(path)
        
    return None

df = load_results(choice)

# =========================
# SECTION 1: MODEL HEALTH
# =========================
st.subheader("Model Health")

if not run_df.empty:
    # Deterministic Deduplication
    run_df["run_idx"] = range(len(run_df))
    run_df = run_df.sort_values(["date", "run_idx"])
    run_df = run_df.drop_duplicates(subset=["date"], keep="last")
    
    last = run_df.iloc[-1]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last run date", str(last.get("date", "")))
    c2.metric("Runtime (min)", round(float(last.get("runtime_sec", 0)) / 60.0, 1))
    c3.metric("Universe", int(last.get("universe", 0)))
    c4.metric("Eligible", int(last.get("eligible", 0)))

    # Exclusions Chart
    st.write("Recent Exclusion Trends")
    ex_rows = []
    for _, r in run_df.iterrows():
        ex = r.get("exclusions", {}) or {}
        row = {"date": r["date"], **ex}
        ex_rows.append(row)
    
    if ex_rows:
        ex_df = pd.DataFrame(ex_rows).fillna(0).set_index("date")
        totals = ex_df.sum().sort_values(ascending=False)
        top_cols = list(totals.head(8).index)
        if top_cols:
            st.bar_chart(ex_df[top_cols].iloc[-5:])
else:
    st.info("No run log found.")

# =========================
# SECTION 2: TRACKING
# =========================
if choice == "(latest)" and DIFF_FILE.exists():
    try:
        with open(DIFF_FILE, "r") as f:
            diff = json.load(f)
            
        new_entrants = diff.get("new_entrants", [])
        if new_entrants:
            st.success(f"New to Top 20 Today: {', '.join(new_entrants)}")
            
    except Exception as e:
        st.error(f"Error loading diffs: {e}")

# =========================
# SECTION 3: RANKINGS
# =========================
st.subheader("Top Picks")

if df is None or df.empty:
    st.warning("No results file found yet.")
else:
    # Sector Filter
    sector_options = ["(all)"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_choice = st.sidebar.selectbox("Sector filter", sector_options)

    view = df.copy()
    if sector_choice != "(all)":
        view = view[view["Sector"] == sector_choice]

    # Alpha Filter
    if "Alpha_Score" in view.columns:
        view = view[view["Alpha_Score"] >= min_alpha]
        view = view.sort_values("Alpha_Score", ascending=False)

    # Display Table
    st.write(f"Showing top {top_n} for: {choice}")
    
    # Columns to display (Cleaned up list)
    cols_to_show = [
        "Ticker", "Alpha_Score", "Fundamental_Score", "Sentiment_Score", 
        "Streak_Days", "Analyst_Count", "Rec_BuyRatio", "Sector"
    ]
    
    final_cols = [c for c in cols_to_show if c in view.columns]
    
    st.dataframe(view[final_cols].head(top_n), use_container_width=True)

    # Composition Chart
    st.write("Sector composition (top list)")
    if "Sector" in view.columns:
        st.bar_chart(view.head(top_n)["Sector"].value_counts())
