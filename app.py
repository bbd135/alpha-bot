import json
from pathlib import Path
import pandas as pd
import streamlit as st

RESULTS_DIR = Path("results")
DAILY_DIR = RESULTS_DIR / "daily"
RUN_LOG = RESULTS_DIR / "run_log.jsonl"
LATEST = RESULTS_DIR / "latest.csv"
DIFF_FILE = RESULTS_DIR / "daily_diff.json"

st.set_page_config(page_title="AlphaBot Quant Screener V8.12", layout="wide")
st.title("AlphaBot Quant Screener V8.12")

# LOAD DATA
runs = []
if RUN_LOG.exists():
    try:
        with open(RUN_LOG, "r") as f:
            for line in f:
                if line.strip():
                    try: runs.append(json.loads(line))
                    except: pass
    except: pass
run_df = pd.DataFrame(runs) if runs else pd.DataFrame()

# SIDEBAR
st.sidebar.header("Controls")
v8_files = sorted(DAILY_DIR.glob("alpha_v8_results_*.csv"), key=lambda x: x.name)
all_files = sorted(v8_files, key=lambda x: x.name)[::-1]
date_options = ["(latest)"] + [p.name.replace("alpha_v8_results_", "").replace(".csv", "") for p in all_files]
date_options = list(dict.fromkeys(date_options))
choice = st.sidebar.selectbox("Select date", date_options)

sort_option = st.sidebar.radio("Sort By:", ["Alpha Score", "Fundamental Score"])
top_n = st.sidebar.slider("Top N", 5, 50, 20)
min_alpha = st.sidebar.slider("Min Alpha Score", 0, 100, 0)

# HELPER
def load_results(selected, mode):
    # Determine which file suffix to look for
    file_prefix = "alpha_v8_results_" if mode == "Alpha Score" else "alpha_v8_candidates_"
    
    if selected == "(latest)":
        if mode == "Alpha Score":
            if LATEST.exists(): return pd.read_csv(LATEST)
        else:
            cand_latest = RESULTS_DIR / "latest_candidates.csv"
            if cand_latest.exists(): return pd.read_csv(cand_latest)
        return None

    target_file = DAILY_DIR / f"{file_prefix}{selected}.csv"
    
    if target_file.exists():
        return pd.read_csv(target_file)
    
    # Fallback: If Candidates file missing, load Survivors file
    if mode == "Fundamental Score":
        fallback = DAILY_DIR / f"alpha_v8_results_{selected}.csv"
        if fallback.exists():
            st.toast("Fundamental view not available for this date. Showing Survivors.", icon="⚠️")
            return pd.read_csv(fallback)
            
    return None

df = load_results(choice, sort_option)

# MODEL HEALTH
st.subheader("Model Health")
if not run_df.empty:
    run_df["run_idx"] = range(len(run_df))
    run_df = run_df.sort_values(["date", "run_idx"]).drop_duplicates(subset=["date"], keep="last")
    last = run_df.iloc[-1]
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Date", str(last.get("date", "")))
    c2.metric("Runtime", f"{round(float(last.get('runtime_sec', 0)) / 60.0, 1)}m")
    c3.metric("Stage 1: Global", int(last.get("stage_1_pop", 0)))
    c4.metric("Stage 2: Select", int(last.get("stage_2_candidates", 0)))
    c5.metric("Final: Quality", int(last.get("final_survivors", 0)))

    # EXCLUSIONS GRAPH
    st.write("Recent Exclusion Trends")
    ex_rows = []
    for _, r in run_df.tail(5).iterrows():
        ex = r.get("exclusions", {})
        if ex:
            row = {"date": r["date"], **ex}
            ex_rows.append(row)
    
    if ex_rows:
        ex_df = pd.DataFrame(ex_rows).fillna(0).set_index("date")
        st.bar_chart(ex_df)
    else:
        st.caption("No exclusion data available for recent runs.")

# DIFFS
if choice == "(latest)" and DIFF_FILE.exists():
    try:
        with open(DIFF_FILE, "r") as f:
            diff = json.load(f)
        new_entrants = diff.get("new_entrants", [])
        if new_entrants: st.success(f"New to Top 20: {', '.join(new_entrants)}")
    except: pass

# RANKINGS
st.subheader("Top Picks")
if df is None or df.empty:
    st.warning("No results found.")
else:
    sector_options = ["(all)"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_choice = st.sidebar.selectbox("Sector filter", sector_options)
    
    view = df.copy()
    if sector_choice != "(all)": view = view[view["Sector"] == sector_choice]
    
    # Conditional Filtering & Sorting
    if sort_option == "Alpha Score":
        if "Alpha_Score" in view.columns:
            view = view[view["Alpha_Score"] >= min_alpha]
        sort_col = "Alpha_Score"
    else:
        # V8.12 FIX: Handle column name difference in fallback files
        if "Fundamental_Score" in view.columns:
            sort_col = "Fundamental_Score"
        elif "FundamentalPlusRev_Score" in view.columns:
            sort_col = "FundamentalPlusRev_Score"
        else:
            sort_col = None # Should fail gracefully
        
    if sort_col and sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=False)
    
    st.write(f"Showing top {top_n} sorted by {sort_option}")
    
    # Dynamic Columns based on View
    if sort_option == "Alpha Score":
        cols = ["Ticker", "Name", "Alpha_Score", "FundamentalPlusRev_Score", 
                "Value_Grade", "Growth_Grade", "Prof_Grade", "Mom_Grade", "Rev_Grade",
                "Val_Fail", "Mom_Fail", "Sentiment_Score", "Analyst_Count", "Sector"]
    else:
        # Candidates view
        cols = ["Ticker", "Name", "Fundamental_Score", 
                "Value_Grade", "Growth_Grade", "Prof_Grade", "Mom_Grade",
                "Sector", "Price", "PE_Rank"]

    final_cols = [c for c in cols if c in view.columns]
    st.dataframe(view[final_cols].head(top_n), use_container_width=True)
    
    if "Sector" in view.columns:
        st.bar_chart(view.head(top_n)["Sector"].value_counts())
