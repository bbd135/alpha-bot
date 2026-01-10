import json
from pathlib import Path
import pandas as pd
import streamlit as st

RESULTS_DIR = Path("results")
DAILY_DIR = RESULTS_DIR / "daily"
RUN_LOG = RESULTS_DIR / "run_log.jsonl"
LATEST = RESULTS_DIR / "latest.csv"

st.set_page_config(page_title="Alpha Screener Dashboard", layout="wide")

st.title("📈 Alpha Screener Dashboard")

# --------- Load run log (health) ----------
runs = []
if RUN_LOG.exists():
    with open(RUN_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except:
                pass

run_df = pd.DataFrame(runs) if runs else pd.DataFrame()

# --------- Sidebar controls ----------
st.sidebar.header("Controls")

# Choose date file (or latest)
daily_files = sorted(DAILY_DIR.glob("alpha_v5_results_*.csv"))
date_options = ["(latest)"] + [p.name.replace("alpha_v5_results_", "").replace(".csv", "") for p in daily_files[::-1]]

choice = st.sidebar.selectbox("Select date", date_options)

top_n = st.sidebar.slider("Top N", 5, 50, 20)
min_alpha = st.sidebar.slider("Min Alpha Score", 0, 100, 0)

# --------- Load selected results ----------
def load_results(selected):
    if selected == "(latest)":
        if LATEST.exists():
            return pd.read_csv(LATEST)
        return None

    path = DAILY_DIR / f"alpha_v5_results_{selected}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

df = load_results(choice)

# --------- Model health section ----------
st.subheader("🩺 Model Health")

if not run_df.empty:
    run_df = run_df.sort_values("date")
    last = run_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last run date", str(last.get("date", "")))
    c2.metric("Runtime (min)", round(float(last.get("runtime_sec", 0)) / 60.0, 1))
    c3.metric("Universe", int(last.get("universe", 0)))
    c4.metric("Eligible", int(last.get("eligible", 0)))

    st.write("**Runtime over time**")
    st.line_chart(run_df.set_index("date")["runtime_sec"] / 60.0)

    st.write("**Eligible count over time**")
    st.line_chart(run_df.set_index("date")["eligible"])

    # Exclusions over time (top reasons only to keep it readable)
    st.write("**Exclusions (top reasons)**")
    ex_rows = []
    for _, r in run_df.iterrows():
        ex = r.get("exclusions", {}) or {}
        row = {"date": r["date"], **ex}
        ex_rows.append(row)
    ex_df = pd.DataFrame(ex_rows).fillna(0).set_index("date")

    # keep only top 8 by total
    totals = ex_df.sum().sort_values(ascending=False)
    top_cols = list(totals.head(8).index)
    if top_cols:
        st.area_chart(ex_df[top_cols])

else:
    st.info("No run log found yet. After the first GitHub Action run, health charts will appear.")

# --------- Results section ----------
st.subheader("🏆 Picks")

if df is None or df.empty:
    st.warning("No results file found yet. Wait for the first scheduled run, or run it locally once and push.")
else:
    # optional filters
    sector_options = ["(all)"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_choice = st.sidebar.selectbox("Sector filter", sector_options)

    view = df.copy()
    if sector_choice != "(all)":
        view = view[view["Sector"] == sector_choice]

    if "Alpha_Score" in view.columns:
        view = view[view["Alpha_Score"] >= min_alpha]
        view = view.sort_values("Alpha_Score", ascending=False)

    st.write(f"Showing top **{top_n}** for: **{choice}**")
    cols = [c for c in ["Ticker", "Alpha_Score", "Fundamental_Score", "Sentiment_Score", "Streak_Days", "Sector", "SubIndustry"] if c in view.columns]
    st.dataframe(view[cols].head(top_n), use_container_width=True)

    st.write("**Sector composition (top list)**")
    if "Sector" in view.columns:
        st.bar_chart(view.head(top_n)["Sector"].value_counts())

