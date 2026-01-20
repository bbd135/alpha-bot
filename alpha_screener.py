import os
import io
import json
import math
import time
import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# =========================
# CONFIGURATION & SETUP
# =========================
RUN_START = time.time()
load_dotenv()

API_KEY = os.getenv("FINNHUB_API_KEY")
if not API_KEY:
    print("⚠️  Warning: FINNHUB_API_KEY not found. Script may fail.")

BASE_URL = "https://finnhub.io/api/v1"

# FOLDER SETUP
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DAILY_DIR = RESULTS_DIR / "daily"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# FILES
GICS_CACHE_FILE = DATA_DIR / "sp1500_gics_map.json"
RUN_LOG_FILE = RESULTS_DIR / "run_log.jsonl"
LATEST_CSV = RESULTS_DIR / "latest.csv"
LATEST_CANDIDATES_CSV = RESULTS_DIR / "latest_candidates.csv"
DIFF_FILE = RESULTS_DIR / "daily_diff.json"

# STAGE 1 FILTERS (The Census)
MIN_PRICE = 10.0
MIN_MKTCAP_MUSD = 500.0
EXCLUDE_NON_US = True
EXCLUDE_REITS = True

# STAGE 2 FILTERS (The Interview)
MIN_ANALYSTS = 5
STAGE_2_CUTOFF = 500

# SENTIMENT
FLOOR_BUY_RATIO = 0.65       
STREAK_ON = 0.80             
STREAK_OFF = 0.75            
MAX_STREAK_DAYS = 90         
DAYS_INTO_SOFT_CAP = 0.25    

# SCORING WEIGHTS
WEIGHT_FUNDAMENTALS = 0.80   
WEIGHT_SENTIMENT = 0.20      

FUND_WEIGHTS = {
    "Value_S": 0.25,     
    "Growth_S": 0.20,
    "Prof_S": 0.25,      
    "Mom_S": 0.20,       
    "Rev_S": 0.10        
}

MOM_WEIGHTS = {
    "Range": 0.40,
    "12M": 0.60
}

# DYNAMIC BLEND CONSTANT
K_BLEND = 20

# GATING CONSTANTS
VAL_THRESHOLD = 0.35      
MOM_THRESHOLD = 0.30      
PENALTY_STRENGTH = 0.5    

BAYESIAN_K = 5               
GLOBAL_BUY_AVG = 0.55        

# API SETTINGS
MAX_CALLS_PER_MIN = float(os.getenv("FINNHUB_MAX_CALLS_PER_MIN", "290"))
MIN_INTERVAL = 60.0 / MAX_CALLS_PER_MIN

print("\n--- ALPHA-BOT V8.12 (DISTINCT SCORES) ---\n")

# =========================
# CLASSES
# =========================

class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        now = time.time()
        dt = now - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()

class GICSManager:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.mapping = self._load_or_scrape()

    def _load_or_scrape(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    print("Loaded GICS from cache.")
                    return json.load(f)
            except:
                pass
        return self._scrape_wikipedia()

    def _scrape_wikipedia(self):
        print("Scraping Wikipedia GICS Data...")
        mapping = {}
        urls = [
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        ]
        headers = {"User-Agent": "Mozilla/5.0"}
        
        def find_col(df, needles):
            for c in df.columns:
                if any(n in str(c).lower() for n in needles): return c
            return None

        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=20)
                tables = pd.read_html(io.StringIO(r.text))
                for df in tables:
                    sym_col = find_col(df, ["symbol", "ticker"])
                    sec_col = find_col(df, ["sector"])
                    if sym_col and sec_col:
                        sub_col = find_col(df, ["sub-industry", "sub industry"])
                        for _, row in df.iterrows():
                            sym = str(row[sym_col]).strip().upper().replace(".", "-")
                            mapping[sym] = {
                                "Sector": str(row[sec_col]),
                                "SubIndustry": str(row[sub_col]) if sub_col else "Unknown"
                            }
                        break
            except Exception as e:
                print(f"Error {url}: {e}")

        with open(self.cache_file, "w") as f:
            json.dump(mapping, f)
        return mapping

    def get_gics(self, ticker):
        return self.mapping.get(ticker, {"Sector": "Unknown", "SubIndustry": "Unknown"})
    
    def get_universe(self):
        return sorted(list(self.mapping.keys()))

class ExclusionTracker:
    def __init__(self):
        self.stats = {}
    def log(self, reason):
        self.stats[reason] = self.stats.get(reason, 0) + 1

# =========================
# HELPERS
# =========================
def safe_num(x):
    try:
        return float(x) if x is not None else math.nan
    except:
        return math.nan

limiter = RateLimiter(MIN_INTERVAL)
session = requests.Session()

def finnhub_get(path, params):
    params["token"] = API_KEY
    for attempt in range(3):
        limiter.wait()
        try:
            r = session.get(f"{BASE_URL}{path}", params=params, timeout=15)
            if r.status_code == 200: return r.json()
            if r.status_code == 429: time.sleep(2 ** attempt)
        except:
            time.sleep(1)
    return None

def winsorize_series_by_group(df, target_col, group_col, lower=0.05, upper=0.95):
    # Guard against tiny groups
    def clip_group(x):
        x2 = x.dropna()
        if len(x2) < 5: return x
        return x.clip(lower=x2.quantile(lower), upper=x2.quantile(upper))
    return df.groupby(group_col)[target_col].transform(clip_group)

def dynamic_sector_rank(df, value_col, higher_is_better, use_fixed_sector=False):
    if df[value_col].isnull().all(): return pd.Series(50, index=df.index)
    
    sector_rank = df.groupby("Sector")[value_col].rank(pct=True).fillna(0.5)
    
    if use_fixed_sector:
        if not higher_is_better: sector_rank = 1.0 - sector_rank
        return (sector_rank * 100.0).clip(0, 100)
        
    sub_rank = df.groupby("SubIndustry")[value_col].rank(pct=True).fillna(sector_rank)
    sub_counts = df.groupby("SubIndustry")[value_col].transform("count")
    w_sector = K_BLEND / (sub_counts + K_BLEND)
    
    blended = (w_sector * sector_rank) + ((1.0 - w_sector) * sub_rank)
    
    if not higher_is_better: blended = 1.0 - blended
    return (blended * 100.0).clip(0, 100)

def pct_to_letter(pct_series):
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.93, 0.97, 1.0000001]
    labels = ["F","D-","D","D+","C-","C","C+","B-","B","B+","A-","A","A+"]
    return pd.cut(pct_series.clip(0,1), bins=bins, labels=labels, include_lowest=True)

def get_shrunk_ratio(row):
    try:
        sb = float(row.get("strongBuy", 0) or 0)
        b  = float(row.get("buy", 0) or 0)
        h  = float(row.get("hold", 0) or 0)
        s  = float(row.get("sell", 0) or 0)
        ss = float(row.get("strongSell", 0) or 0)
        tot = sb + b + h + s + ss
        
        if tot == 0: return None
        
        raw_ratio = (sb + b) / tot
        return (tot / (tot + BAYESIAN_K)) * raw_ratio + (BAYESIAN_K / (tot + BAYESIAN_K)) * GLOBAL_BUY_AVG
    except:
        return None

def calculate_asymptotic_score(streak_days):
    k = 0.021
    return 100.0 * (1.0 - math.exp(-k * streak_days))

def calculate_streak_shrunk(rec_list, today_date):
    if not rec_list: return 0.0, 0.0, 0.0
    
    r0 = get_shrunk_ratio(rec_list[0])
    if r0 is None: return 0.0, 0.0, 0.0
    
    r1 = get_shrunk_ratio(rec_list[1]) if len(rec_list) > 1 else None
    
    was_strong_prev = (r1 is not None and r1 >= STREAK_ON)
    
    if was_strong_prev:
        if r0 < STREAK_OFF: return 0.0, 0.0, 0.0
    else:
        if r0 < STREAK_ON: return 0.0, 0.0, 0.0
        
    try:
        p_date = datetime.strptime(rec_list[0].get("period", ""), "%Y-%m-%d").date()
        days_into = max(0, (today_date - p_date).days)
        days_into = min(days_into, 45)
    except:
        days_into = 0
    
    days_val = days_into * DAYS_INTO_SOFT_CAP
    
    period_streak = 1 
    for row in rec_list[1:]:
        r = get_shrunk_ratio(row)
        if r is None or r < STREAK_ON: 
            break
        period_streak += 1
        
    past_periods = max(0, period_streak - 1)
    
    total_days = (past_periods * 30) + days_val
    capped_days = min(total_days, MAX_STREAK_DAYS)
    score = calculate_asymptotic_score(capped_days)
    
    return float(total_days), float(capped_days), float(score)

# =========================
# MAIN LOGIC
# =========================
gics = GICSManager(GICS_CACHE_FILE)
tracker = ExclusionTracker()
universe = gics.get_universe()

today = datetime.now(timezone.utc).date()
today_str = today.strftime("%Y-%m-%d")

# --- STAGE 1: THE CENSUS ---
print(f"--- STAGE 1: CENSUS ({len(universe)} tickers) ---")
pop_rows = []

for i, sym in enumerate(universe):
    if i % 100 == 0: print(f"Census {i}/{len(universe)}...", end="\r")
    
    p2 = finnhub_get("/stock/profile2", {"symbol": sym})
    if not p2 or not p2.get("ticker"):
        tracker.log("API_Fail_Profile"); continue
    if EXCLUDE_NON_US and (p2.get("country") or "").upper() != "US":
        tracker.log("Non_US"); continue
    if safe_num(p2.get("marketCapitalization")) < MIN_MKTCAP_MUSD:
        tracker.log("Small_Cap"); continue

    company_name = p2.get("name", sym)
    gics_data = gics.get_gics(sym)
    
    if EXCLUDE_REITS:
        if "REAL ESTATE" in gics_data["Sector"].upper() and "REIT" in gics_data["SubIndustry"].upper():
            tracker.log("REIT_GICS"); continue

    q = finnhub_get("/quote", {"symbol": sym})
    price = safe_num(q.get("c")) if q else math.nan
    
    if pd.isna(price) or price <= 0.01:
        tracker.log("Bad_Price"); continue
    if price < MIN_PRICE:
        tracker.log("Low_Price"); continue

    metric = (finnhub_get("/stock/metric", {"symbol": sym, "metric": "all"}) or {}).get("metric", {})
    
    eps_ttm = safe_num(metric.get("epsGrowthTTMYoy"))
    eps_3y  = safe_num(metric.get("epsGrowth3Y"))
    eps_5y  = safe_num(metric.get("epsGrowth5Y"))
    eps_g = eps_ttm if not pd.isna(eps_ttm) else (eps_3y if not pd.isna(eps_3y) else eps_5y)
    
    pe = safe_num(metric.get("peBasicExclExtraTTM"))
    ps = safe_num(metric.get("psTTM"))
    
    ev_ebitda_raw = safe_num(metric.get("evEbitdaTTM"))
    if pd.isna(ev_ebitda_raw):
        ev_ebitda_used = np.nan
    elif ev_ebitda_raw <= 0:
        ev_ebitda_used = 1e6 
    else:
        ev_ebitda_used = ev_ebitda_raw

    rev_g = safe_num(metric.get("revenueGrowthTTMYoy"))
    if pd.isna(rev_g): rev_g = safe_num(metric.get("revenueGrowth5Y"))
    
    roe = safe_num(metric.get("roeTTM"))
    op_margin_raw = safe_num(metric.get("operatingMarginTTM"))
    op_margin = (op_margin_raw / 100.0) if (not pd.isna(op_margin_raw) and abs(op_margin_raw) > 1.0) else op_margin_raw

    roic = safe_num(metric.get("roiTTM"))

    bvps = safe_num(metric.get("bookValuePerShareAnnual"))
    pb_ratio = (price / bvps) if (not pd.isna(bvps) and bvps > 0) else np.nan

    h52 = safe_num(metric.get("52WeekHigh"))
    l52 = safe_num(metric.get("52WeekLow"))
    
    if pd.isna(price) or pd.isna(h52) or pd.isna(l52) or h52 <= l52:
        mom_range = np.nan
    else:
        rng = (price - l52) / (h52 - l52)
        mom_range = max(0.0, min(1.0, rng))
    
    mom_12m = safe_num(metric.get("52WeekPriceReturnDaily"))

    pop_rows.append({
        "Ticker": sym, 
        "Name": company_name,
        "Sector": gics_data["Sector"], 
        "SubIndustry": gics_data["SubIndustry"],
        "Price": price,
        "PE_Rank": pe if (pe and pe > 0) else 1e6,
        "PS_Rank": ps if (ps and ps > 0) else 1e6,
        "EV_EBITDA_Used": ev_ebitda_used,
        "PB_Ratio": pb_ratio, 
        "EPS_Growth": eps_g,
        "Rev_Growth": rev_g,
        "ROE": roe,
        "Op_Margin": op_margin, 
        "ROIC": roic,
        "Mom_Range": mom_range,
        "Mom_12M": mom_12m
    })

df_pop = pd.DataFrame(pop_rows)
print(f"\nCensus Complete. Population size: {len(df_pop)}")

# --- STAGE 1 SCORING ---

winsor_cols = ["EPS_Growth", "Rev_Growth", "Mom_12M", "Mom_Range", "ROE", "Op_Margin", "ROIC"]
for c in winsor_cols:
    df_pop[f"{c}_Win"] = winsorize_series_by_group(df_pop, c, "Sector")

# 1. VALUE
v_pe = dynamic_sector_rank(df_pop, "PE_Rank", False, use_fixed_sector=True)
v_ps = dynamic_sector_rank(df_pop, "PS_Rank", False, use_fixed_sector=True)
v_ev = dynamic_sector_rank(df_pop, "EV_EBITDA_Used", False, use_fixed_sector=True)
v_pb = dynamic_sector_rank(df_pop, "PB_Ratio", False, use_fixed_sector=True)

mask_fin = df_pop["Sector"] == "Financials"
v_ev[mask_fin] = np.nan 
v_ps[mask_fin] = np.nan 
v_pb[~mask_fin] = np.nan 

df_pop["Value_S"] = pd.concat([v_pe, v_ps, v_ev, v_pb], axis=1).mean(axis=1).fillna(50)

# 2. GROWTH
g_eps = dynamic_sector_rank(df_pop, "EPS_Growth_Win", True, use_fixed_sector=False)
g_rev = dynamic_sector_rank(df_pop, "Rev_Growth_Win", True, use_fixed_sector=False)
df_pop["Growth_S"] = pd.concat([g_eps, g_rev], axis=1).mean(axis=1).fillna(50)

# 3. PROFITABILITY
p_roe = dynamic_sector_rank(df_pop, "ROE_Win", True, use_fixed_sector=True)
p_margin = dynamic_sector_rank(df_pop, "Op_Margin_Win", True, use_fixed_sector=True)
p_roic = dynamic_sector_rank(df_pop, "ROIC_Win", True, use_fixed_sector=True)

mask_roic_missing = df_pop["ROIC"].isna()
p_roic[mask_roic_missing] = np.nan
p_margin[mask_fin] = np.nan
p_roic[mask_fin] = np.nan

df_pop["Prof_S"] = pd.concat([p_roe, p_margin, p_roic], axis=1).mean(axis=1).fillna(50)

mask_thin = df_pop["Op_Margin_Win"].notna() & (df_pop["Op_Margin_Win"] < 0.05) & (~mask_fin)
df_pop.loc[mask_thin, "Prof_S"] = df_pop.loc[mask_thin, "Prof_S"].clip(upper=60)

# 4. MOMENTUM
m1 = dynamic_sector_rank(df_pop, "Mom_Range_Win", True, use_fixed_sector=False).fillna(50)
m2 = dynamic_sector_rank(df_pop, "Mom_12M_Win", True, use_fixed_sector=False).fillna(50)
df_pop["Mom_S"] = (m1 * MOM_WEIGHTS["Range"]) + (m2 * MOM_WEIGHTS["12M"])

df_pop["Rev_S"] = 50.0 

# Aggregates
df_pop["Fundamental_Score"] = (
    df_pop["Value_S"] * FUND_WEIGHTS["Value_S"] +
    df_pop["Growth_S"] * FUND_WEIGHTS["Growth_S"] +
    df_pop["Prof_S"]  * FUND_WEIGHTS["Prof_S"] +
    df_pop["Mom_S"]   * FUND_WEIGHTS["Mom_S"] +
    df_pop["Rev_S"]   * FUND_WEIGHTS["Rev_S"]
)

# Population Percentiles
df_pop["Value_Pct"] = df_pop.groupby("Sector")["Value_S"].rank(pct=True)
df_pop["Growth_Pct"] = df_pop.groupby("Sector")["Growth_S"].rank(pct=True)
df_pop["Prof_Pct"] = df_pop.groupby("Sector")["Prof_S"].rank(pct=True)
df_pop["Mom_Pct"] = df_pop.groupby("Sector")["Mom_S"].rank(pct=True)

df_pop["Value_Grade"] = pct_to_letter(df_pop["Value_Pct"])
df_pop["Growth_Grade"] = pct_to_letter(df_pop["Growth_Pct"])
df_pop["Prof_Grade"] = pct_to_letter(df_pop["Prof_Pct"])
df_pop["Mom_Grade"] = pct_to_letter(df_pop["Mom_Pct"])

# --- V8.12: DEDUPE & SAVE CANDIDATES (Fundamental View) ---
df_pop_sorted = df_pop.sort_values("Fundamental_Score", ascending=False).reset_index(drop=True)

# V8.12 Fix: Dedupe BEFORE Head to ensure full list
df_pop_sorted["Name_Key"] = df_pop_sorted["Name"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
df_pop_deduped = df_pop_sorted.drop_duplicates(subset=["Name_Key"], keep="first")

candidates_export = df_pop_deduped.head(STAGE_2_CUTOFF).copy()

CANDIDATES_FILENAME = f"alpha_v8_candidates_{today_str}.csv"
cand_path = DAILY_DIR / CANDIDATES_FILENAME
candidates_export.to_csv(cand_path, index=False)
candidates_export.to_csv(LATEST_CANDIDATES_CSV, index=False)
print(f"Saved Candidates (Fundamental View) to {cand_path}")

# --- STAGE 2: THE INTERVIEW ---
candidates = candidates_export.copy()

print(f"\n--- STAGE 2: INTERVIEW ({len(candidates)} Candidates) ---")

final_rows = []

for j, (_, row) in enumerate(candidates.iterrows(), start=1):
    sym = row["Ticker"]
    if j % 50 == 0: print(f"Interviewing {j}/{len(candidates)}...", end="\r")
    
    rec = finnhub_get("/stock/recommendation", {"symbol": sym})
    if not rec or not isinstance(rec, list) or len(rec) == 0:
        tracker.log("Stage2_NoRec"); continue
    
    rec.sort(key=lambda x: x.get("period", ""), reverse=True)
    d = rec[0]
    
    sb = float(d.get("strongBuy", 0) or 0)
    b  = float(d.get("buy", 0) or 0)
    h  = float(d.get("hold", 0) or 0)
    s  = float(d.get("sell", 0) or 0)
    ss = float(d.get("strongSell", 0) or 0)
    
    total_analysts = sb + b + h + s + ss
    
    if total_analysts <= 0:
        tracker.log("Stage2_ZeroAnalysts"); continue
    
    if total_analysts < MIN_ANALYSTS:
        tracker.log("Stage2_LowAnalysts"); continue
        
    raw_buy_ratio = (sb + b) / total_analysts
    
    N = total_analysts
    K = BAYESIAN_K
    shrunk_ratio = (N / (N + K)) * raw_buy_ratio + (K / (N + K)) * GLOBAL_BUY_AVG

    if shrunk_ratio < FLOOR_BUY_RATIO:
        tracker.log("Stage2_LowBuyRatio"); continue

    streak_days_raw, streak_days_capped, streak_score_val = calculate_streak_shrunk(rec, today)
    
    rec_change = math.nan
    target_date = today - timedelta(days=90)
    best_row = None
    best_diff = 10**9
    for r in rec:
        try:
            r_date = datetime.strptime(r.get("period",""), "%Y-%m-%d").date()
            if r_date > target_date: continue
            if sum(float(r.get(k, 0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"]) <= 0: continue
            diff = abs((r_date - target_date).days)
            if diff < best_diff:
                best_diff = diff
                best_row = r
        except: continue
        
    if best_row:
        t_old = sum(float(best_row.get(k, 0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"])
        if t_old > 0:
            old_r = (float(best_row.get("strongBuy", 0) or 0) + float(best_row.get("buy", 0) or 0)) / t_old
            rec_change = raw_buy_ratio - old_r
    
    full_row = row.to_dict()
    full_row["Rev_Change"] = rec_change
    full_row["Rec_BuyRatio"] = raw_buy_ratio
    full_row["Shrunk_Ratio"] = shrunk_ratio
    full_row["Analyst_Count"] = total_analysts
    full_row["Streak_Days"] = streak_days_capped
    full_row["Streak_Score"] = streak_score_val
    
    final_rows.append(full_row)

df_final = pd.DataFrame(final_rows)
print(f"\nInterview Complete. Survivors: {len(df_final)}")

# --- FINAL SCORING ---
if not df_final.empty:
    df_final["Rev_Change_Win"] = winsorize_series_by_group(df_final, "Rev_Change", "Sector")
    df_final["Rev_S"] = dynamic_sector_rank(df_final, "Rev_Change_Win", True, use_fixed_sector=False).fillna(50)
    
    df_final["Rev_Pct"] = df_final.groupby("Sector")["Rev_S"].rank(pct=True)
    df_final["Rev_Grade"] = pct_to_letter(df_final["Rev_Pct"])

    # V8.12: Rename to FundamentalPlusRev_Score
    df_final["FundamentalPlusRev_Score"] = (
        df_final["Value_S"] * FUND_WEIGHTS["Value_S"] +
        df_final["Growth_S"] * FUND_WEIGHTS["Growth_S"] +
        df_final["Prof_S"]  * FUND_WEIGHTS["Prof_S"] +
        df_final["Mom_S"]   * FUND_WEIGHTS["Mom_S"] +
        df_final["Rev_S"]   * FUND_WEIGHTS["Rev_S"]
    )
    
    df_final["Consensus_Score"] = (df_final["Shrunk_Ratio"] * 100.0).clip(0, 100)
    df_final["Sentiment_Score"] = (df_final["Consensus_Score"]*0.6) + (df_final["Streak_Score"]*0.4)
    
    # Use new FundamentalPlusRev_Score for Alpha Calculation
    df_final["Alpha_Score"] = (df_final["FundamentalPlusRev_Score"]*WEIGHT_FUNDAMENTALS) + (df_final["Sentiment_Score"]*WEIGHT_SENTIMENT)
    
    # --- V8.9 SMOOTH GATES (V8.12 Clipped) ---
    val_gap = (VAL_THRESHOLD - df_final["Value_Pct"]).clip(lower=0)
    val_penalty_mult = (1.0 - (val_gap * PENALTY_STRENGTH)).clip(lower=0.0) # V8.12 Safety Clip
    
    mom_gap = (MOM_THRESHOLD - df_final["Mom_Pct"]).clip(lower=0)
    mom_penalty_mult = (1.0 - (mom_gap * PENALTY_STRENGTH)).clip(lower=0.0) # V8.12 Safety Clip
    
    df_final["Alpha_Score"] = df_final["Alpha_Score"] * val_penalty_mult * mom_penalty_mult
    
    df_final["Val_Fail"] = df_final["Value_Pct"] < VAL_THRESHOLD
    df_final["Mom_Fail"] = df_final["Mom_Pct"] < MOM_THRESHOLD
    
    mask_neg_mom = (df_final["Mom_12M"].notna()) & (df_final["Mom_12M"] < 0)
    df_final.loc[mask_neg_mom, "Alpha_Score"] = df_final.loc[mask_neg_mom, "Alpha_Score"].clip(upper=50)

    df_final["Min_Fund_Factor"] = df_final[["Value_S", "Growth_S", "Prof_S", "Mom_S", "Rev_S"]].min(axis=1)
    mask_lopsided = df_final["Min_Fund_Factor"] < 20
    df_final.loc[mask_lopsided, "Alpha_Score"] = df_final.loc[mask_lopsided, "Alpha_Score"] * 0.50

    mask_neg_eps = (df_final["EPS_Growth"].notna()) & (df_final["EPS_Growth"] < 0)
    df_final.loc[mask_neg_eps, "Alpha_Score"] = df_final.loc[mask_neg_eps, "Alpha_Score"] * 0.50
    
    mask_nan_eps = df_final["EPS_Growth"].isna()
    df_final.loc[mask_nan_eps, "Alpha_Score"] = df_final.loc[mask_nan_eps, "Alpha_Score"] * 0.75

    # Dedupe Final (Already deduped at stage 1, but good practice to keep)
    df_final = df_final.sort_values("Alpha_Score", ascending=False).reset_index(drop=True)
    
    OUTPUT_FILENAME = f"alpha_v8_results_{today_str}.csv"
    daily_path = DAILY_DIR / OUTPUT_FILENAME
    df_final.to_csv(daily_path, index=False)
    df_final.to_csv(LATEST_CSV, index=False)
    print(f"Saved Survivors (Alpha View) to {daily_path}")
    
    daily_files = sorted(DAILY_DIR.glob("alpha_v8_results_*.csv"), key=lambda p: p.name)
    if not daily_files: daily_files = sorted(DAILY_DIR.glob("alpha_v7_results_*.csv"), key=lambda p: p.name)
    
    diff_data = {"new_entrants": [], "movers": []}
    if len(daily_files) >= 2:
        prev_file = daily_files[-2]
        try:
            prev_df = pd.read_csv(prev_file)
            prev_df = prev_df.sort_values("Alpha_Score", ascending=False).reset_index(drop=True)
            top_today = df_final.head(20)["Ticker"].tolist()
            top_prev = prev_df.head(20)["Ticker"].tolist()
            new_entrants = [t for t in top_today if t not in top_prev]
            diff_data["new_entrants"] = new_entrants
        except: pass
    with open(DIFF_FILE, "w") as f: json.dump(diff_data, f)

if df_final.empty:
    print("No survivors after Stage 2.")

# LOGGING
runtime_sec = int(time.time() - RUN_START)
ev_missing_pct = 0.0
if "df_pop" in locals() and not df_pop.empty and "EV_EBITDA_Used" in df_pop.columns:
    ev_missing_pct = float(df_pop["EV_EBITDA_Used"].isna().mean())

log_obj = {
    "date": today_str,
    "runtime_sec": runtime_sec,
    "universe": len(universe),
    "stage_1_pop": len(df_pop) if "df_pop" in locals() else 0,
    "stage_2_candidates": len(candidates) if "candidates" in locals() else 0,
    "final_survivors": len(df_final) if "df_final" in locals() else 0,
    "exclusions": tracker.stats,
    "data_quality": {"ev_missing_pct": round(ev_missing_pct * 100.0, 2)}
}
with open(RUN_LOG_FILE, "a") as f:
    f.write(json.dumps(log_obj) + "\n")
