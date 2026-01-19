import os
import io
import json
import math
import time
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
DIFF_FILE = RESULTS_DIR / "daily_diff.json"

# FILTERS
MIN_PRICE = 10.0
MIN_MKTCAP_MUSD = 500.0
EXCLUDE_NON_US = True
EXCLUDE_REITS = True
MIN_ANALYSTS = 5

# SENTIMENT (V7.7 Strict & Clamped)
FLOOR_BUY_RATIO = 0.65       # Hard floor: Shrunk Score < 65% = Kicked out
STREAK_ON = 0.80             # Streak Start: Shrunk Score >= 80%
STREAK_OFF = 0.75            # Hysteresis: Shrunk Score >= 75%
MAX_STREAK_DAYS = 90         # Hard Cap for scoring AND display
DAYS_INTO_SOFT_CAP = 0.25    # Multiplier for days into current month

# SCORING WEIGHTS (V7.7: Quality Focus)
WEIGHT_FUNDAMENTALS = 0.80   
WEIGHT_SENTIMENT = 0.20      

FUND_WEIGHTS = {
    "Value_S": 0.20,
    "Growth_S": 0.20,
    "Prof_S": 0.25,      # Quality is King
    "Mom_S": 0.15,
    "Rev_S": 0.20        # Reduced Proxy Weight
}

MOM_WEIGHTS = {
    "Range": 0.40,
    "12M": 0.60
}

BAYESIAN_K = 5               
GLOBAL_BUY_AVG = 0.55        

# API SETTINGS
# V7.7: Paid Plan Speed (290 calls/min)
MAX_CALLS_PER_MIN = float(os.getenv("FINNHUB_MAX_CALLS_PER_MIN", "290"))
MIN_INTERVAL = 60.0 / MAX_CALLS_PER_MIN

print("\n--- ALPHA-BOT V7.7 (CORRECTED KEYS) ---\n")

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

def blended_rank(df, value_col, group_col, higher_is_better):
    K = 20
    if df[value_col].isnull().all(): return pd.Series(50, index=df.index)
    
    global_rank = df[value_col].rank(pct=True)
    sector_rank = df.groupby("Sector")[value_col].rank(pct=True).fillna(global_rank)
    
    sub_groups = df.groupby(group_col)[value_col]
    sub_rank = sub_groups.rank(pct=True).fillna(sector_rank)
    sub_counts = sub_groups.transform("count")
    
    w = sub_counts / (sub_counts + K)
    blended = (w * sub_rank) + ((1 - w) * sector_rank)
    
    if not higher_is_better: blended = 1.0 - blended
    return (blended * 100.0).clip(0, 100)

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
    
    # 1. Hysteresis Check
    r0 = get_shrunk_ratio(rec_list[0])
    if r0 is None: return 0.0, 0.0, 0.0
    
    r1 = get_shrunk_ratio(rec_list[1]) if len(rec_list) > 1 else None
    
    was_strong_prev = (r1 is not None and r1 >= STREAK_ON)
    
    if was_strong_prev:
        if r0 < STREAK_OFF: return 0.0, 0.0, 0.0
    else:
        if r0 < STREAK_ON: return 0.0, 0.0, 0.0
        
    # 2. Days Into Month
    try:
        p_date = datetime.strptime(rec_list[0].get("period", ""), "%Y-%m-%d").date()
        days_into = max(0, (today_date - p_date).days)
        days_into = min(days_into, 45)
    except:
        days_into = 0
    
    days_val = days_into * DAYS_INTO_SOFT_CAP
    
    # 3. Past Periods
    period_streak = 1 
    for row in rec_list[1:]:
        r = get_shrunk_ratio(row)
        if r is None or r < STREAK_ON: 
            break
        period_streak += 1
        
    past_periods = max(0, period_streak - 1)
    
    total_days = (past_periods * 30) + days_val
    
    # Clamp before scoring
    capped_days = min(total_days, MAX_STREAK_DAYS)
    score = calculate_asymptotic_score(capped_days)
    
    return float(total_days), float(capped_days), float(score)

# =========================
# MAIN LOGIC
# =========================
gics = GICSManager(GICS_CACHE_FILE)
tracker = ExclusionTracker()
universe = gics.get_universe()

print(f"Scanning {len(universe)} tickers...")
rows = []
today = datetime.now(timezone.utc).date()
today_str = today.strftime("%Y-%m-%d")

# Note: We will name output files V7 to match new versioning
OUTPUT_FILENAME = f"alpha_v7_results_{today_str}.csv"

for i, sym in enumerate(universe):
    if i % 100 == 0: print(f"Processing {i}/{len(universe)}...", end="\r")
    
    # 1. Profile
    p2 = finnhub_get("/stock/profile2", {"symbol": sym})
    if not p2 or not p2.get("ticker"):
        tracker.log("API_Fail_Profile"); continue
    if EXCLUDE_NON_US and (p2.get("country") or "").upper() != "US":
        tracker.log("Non_US"); continue
    if safe_num(p2.get("marketCapitalization")) < MIN_MKTCAP_MUSD:
        tracker.log("Small_Cap"); continue

    # 2. GICS
    gics_data = gics.get_gics(sym)
    if EXCLUDE_REITS:
        if "REAL ESTATE" in gics_data["Sector"].upper() and "REIT" in gics_data["SubIndustry"].upper():
            tracker.log("REIT_GICS"); continue

    # 3. Price
    q = finnhub_get("/quote", {"symbol": sym})
    if not q or "c" not in q:
        tracker.log("API_Fail_Quote"); continue
        
    price = safe_num(q.get("c"))
    if pd.isna(price) or price <= 0.01:
        tracker.log("API_Fail_Quote_BadData"); continue
    if price < MIN_PRICE:
        tracker.log("Low_Price"); continue

    # 4. Analyst
    rec = finnhub_get("/stock/recommendation", {"symbol": sym})
    if not rec or not isinstance(rec, list) or len(rec) == 0:
        tracker.log("API_Fail_Rec"); continue
    
    rec.sort(key=lambda x: x.get("period", ""), reverse=True)
    d = rec[0]
    
    sb, b = float(d.get("strongBuy",0)), float(d.get("buy",0))
    total_analysts = sb + b + float(d.get("hold",0)) + float(d.get("sell",0)) + float(d.get("strongSell",0))
    
    if total_analysts == 0:
        tracker.log("API_Fail_Rec_Zero"); continue
        
    raw_buy_ratio = (sb + b) / total_analysts
    
    N = total_analysts
    K = BAYESIAN_K
    shrunk_ratio = (N / (N + K)) * raw_buy_ratio + (K / (N + K)) * GLOBAL_BUY_AVG

    # Smart Lookback (90 days)
    rec_change = math.nan
    target_date = today - timedelta(days=90)
    best_row = None
    best_diff = 10**9
    
    for row in rec:
        try:
            row_date = datetime.strptime(row.get("period",""), "%Y-%m-%d").date()
            if row_date > target_date: continue 
            
            t_old = sum(float(row.get(k, 0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"])
            if t_old <= 0: continue
            
            diff = (target_date - row_date).days
            if diff < best_diff:
                best_diff = diff
                best_row = row
        except: continue
        
    if best_row is None:
        best_diff = 10**9
        for row in rec:
            try:
                row_date = datetime.strptime(row.get("period",""), "%Y-%m-%d").date()
                t_old = sum(float(row.get(k, 0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"])
                if t_old <= 0: continue
                
                diff = abs((row_date - target_date).days)
                if diff < best_diff:
                    best_diff = diff
                    best_row = row
            except: continue
            
    if best_row is not None:
        t_old = sum(float(best_row.get(k, 0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"])
        old_r = (float(best_row.get("strongBuy",0) or 0) + float(best_row.get("buy",0) or 0)) / t_old
        rec_change = raw_buy_ratio - old_r

    if total_analysts < MIN_ANALYSTS:
        tracker.log("No_Analyst_Coverage"); continue
        
    if shrunk_ratio < FLOOR_BUY_RATIO:
        tracker.log("Below_Floor_Ratio"); continue
    
    streak_days_raw, streak_days_capped, streak_score_val = calculate_streak_shrunk(rec, today)

    # 5. Metrics
    metric = (finnhub_get("/stock/metric", {"symbol": sym, "metric": "all"}) or {}).get("metric", {})
    
    # SMART EPS LOGIC
    eps_ttm = safe_num(metric.get("epsGrowthTTMYoy"))
    eps_3y  = safe_num(metric.get("epsGrowth3Y"))
    eps_5y  = safe_num(metric.get("epsGrowth5Y"))
    
    eps_g = eps_ttm
    eps_src = "TTM_YoY"
    if pd.isna(eps_g):
        eps_g = eps_3y
        eps_src = "3Y"
    if pd.isna(eps_g):
        eps_g = eps_5y
        eps_src = "5Y"
    if pd.isna(eps_g): eps_src = "NA"
    
    # 6. Data Row (V7.7 PRO METRICS - CORRECTED KEYS)
    pe = safe_num(metric.get("peBasicExclExtraTTM"))
    ps = safe_num(metric.get("psTTM"))
    
    # EV/EBITDA Logic (Key: evEbitdaTTM)
    ev_ebitda_raw = safe_num(metric.get("evEbitdaTTM"))
    ev_ebitda_used = np.nan
    ev_src = "Missing"
    
    if not pd.isna(ev_ebitda_raw):
        if ev_ebitda_raw <= 0:
            ev_ebitda_used = 1e6 # Penalize invalid/negative
            ev_src = "Invalid<=0"
        else:
            ev_ebitda_used = ev_ebitda_raw
            ev_src = "TTM"

    rev_g = safe_num(metric.get("revenueGrowthTTMYoy"))
    if pd.isna(rev_g): rev_g = safe_num(metric.get("revenueGrowth5Y"))
    
    roe = safe_num(metric.get("roeTTM"))
    op_margin = safe_num(metric.get("operatingMarginTTM"))
    
    # ROIC Logic (Key: roiTTM)
    roic = safe_num(metric.get("roiTTM"))
    
    roic_src = "TTM" if not pd.isna(roic) else "Missing"

    h52 = safe_num(metric.get("52WeekHigh"))
    l52 = safe_num(metric.get("52WeekLow"))
    
    # MOMENTUM RANGE FIX
    if pd.isna(price) or pd.isna(h52) or pd.isna(l52) or h52 <= l52:
        mom_range = np.nan
    else:
        rng = (price - l52) / (h52 - l52)
        mom_range = max(0.0, min(1.0, rng))
    
    mom_12m = safe_num(metric.get("52WeekPriceReturnDaily"))

    rows.append({
        "Ticker": sym, "Sector": gics_data["Sector"], "SubIndustry": gics_data["SubIndustry"],
        "Price": price,
        "PE_Rank": pe if (pe and pe > 0) else 1e6,
        "PS_Rank": ps if (ps and ps > 0) else 1e6,
        "EV_EBITDA": ev_ebitda_raw,
        "EV_EBITDA_Used": ev_ebitda_used,
        "EV_EBITDA_Source": ev_src,
        "EPS_Growth": eps_g,
        "EPS_Growth_Source": eps_src,
        "EPS_Growth_TTMYoY": eps_ttm,
        "EPS_Growth_3Y": eps_3y,
        "EPS_Growth_5Y": eps_5y,
        "Rev_Growth": rev_g,
        "ROE": roe,
        "Op_Margin": op_margin,
        "ROIC": roic,
        "ROIC_Source": roic_src,
        "Mom_Range": mom_range,
        "Mom_12M": mom_12m,
        "Rev_Change": rec_change,
        "Rec_BuyRatio": raw_buy_ratio,
        "Shrunk_Ratio": shrunk_ratio,
        "Analyst_Count": total_analysts,
        "Streak_Days": streak_days_capped, 
        "Streak_Days_Raw": streak_days_raw,
        "Streak_Score": streak_score_val
    })

# =========================
# SCORING & OUTPUT
# =========================
if rows:
    df = pd.DataFrame(rows)
    print(f"\nScoring {len(df)} eligible stocks...")
    
    # 1. VALUE
    v_pe = blended_rank(df,"PE_Rank","SubIndustry",False)
    v_ps = blended_rank(df,"PS_Rank","SubIndustry",False)
    v_ev = blended_rank(df,"EV_EBITDA_Used","SubIndustry",False)
    
    df["Value_S"] = pd.concat([v_pe, v_ps, v_ev], axis=1).mean(axis=1).fillna(50)
    
    # 2. GROWTH
    g_eps = blended_rank(df,"EPS_Growth","SubIndustry",True)
    g_rev = blended_rank(df,"Rev_Growth","SubIndustry",True)
    
    df["Growth_S"] = pd.concat([g_eps, g_rev], axis=1).mean(axis=1).fillna(50)
    
    # 3. PROFITABILITY
    p_roe = blended_rank(df,"ROE","SubIndustry",True)
    p_margin = blended_rank(df,"Op_Margin","SubIndustry",True)
    p_roic = blended_rank(df,"ROIC","SubIndustry",True)
    
    mask_roic_missing = df["ROIC"].isna()
    p_roic[mask_roic_missing] = np.nan
    
    df["Prof_S"] = pd.concat([p_roe, p_margin, p_roic], axis=1).mean(axis=1).fillna(50)
    
    # Smart Margin Floor (Normalize %)
    op_clean = df["Op_Margin"].apply(lambda x: x / 100.0 if (not pd.isna(x) and x > 1.0) else x)
    mask_thin_margin = op_clean.notna() & (op_clean < 0.05)
    df.loc[mask_thin_margin, "Prof_S"] = df.loc[mask_thin_margin, "Prof_S"].clip(upper=60)
    
    # 4. MOMENTUM
    m1 = blended_rank(df,"Mom_Range","SubIndustry",True).fillna(50)
    m2 = blended_rank(df,"Mom_12M","SubIndustry",True).fillna(50)
    df["Mom_S"] = (m1 * MOM_WEIGHTS["Range"]) + (m2 * MOM_WEIGHTS["12M"])
    
    # 5. REVISIONS
    df["Rev_S"] = blended_rank(df,"Rev_Change","SubIndustry",True).fillna(50)

    # Aggregates
    df["Fundamental_Score"] = (
        df["Value_S"] * FUND_WEIGHTS["Value_S"] +
        df["Growth_S"] * FUND_WEIGHTS["Growth_S"] +
        df["Prof_S"]  * FUND_WEIGHTS["Prof_S"] +
        df["Mom_S"]   * FUND_WEIGHTS["Mom_S"] +
        df["Rev_S"]   * FUND_WEIGHTS["Rev_S"]
    )
    
    df["Consensus_Score"] = blended_rank(df, "Shrunk_Ratio", "SubIndustry", True).fillna(50)
    
    df["Sentiment_Score"] = (df["Consensus_Score"]*0.6) + (df["Streak_Score"]*0.4)
    df["Alpha_Score"] = (df["Fundamental_Score"]*WEIGHT_FUNDAMENTALS) + (df["Sentiment_Score"]*WEIGHT_SENTIMENT)
    
    # --- V7.7 GATES & PENALTIES ---
    df["Alpha_Score_PreGates"] = df["Alpha_Score"] 
    
    # 1. Momentum Gate
    mask_neg_mom = (df["Mom_12M"].notna()) & (df["Mom_12M"] < 0)
    df.loc[mask_neg_mom, "Alpha_Score"] = df.loc[mask_neg_mom, "Alpha_Score"].clip(upper=50)

    # 2. Min Factor Gate
    df["Min_Fund_Factor"] = df[["Value_S", "Growth_S", "Prof_S", "Mom_S", "Rev_S"]].min(axis=1)
    mask_lopsided = df["Min_Fund_Factor"] < 20
    df.loc[mask_lopsided, "Alpha_Score"] = df.loc[mask_lopsided, "Alpha_Score"] * 0.50

    # 3. EPS Penalties
    mask_neg_eps = (df["EPS_Growth"].notna()) & (df["EPS_Growth"] < 0)
    df.loc[mask_neg_eps, "Alpha_Score"] = df.loc[mask_neg_eps, "Alpha_Score"] * 0.50
    
    mask_nan_eps = df["EPS_Growth"].isna()
    df.loc[mask_nan_eps, "Alpha_Score"] = df.loc[mask_nan_eps, "Alpha_Score"] * 0.75
    
    # 4. Audit Flags
    df["Momentum_Fail"] = mask_neg_mom
    df["Lopsided_Fail"] = mask_lopsided
    df["Neg_EPS_Fail"] = mask_neg_eps
    df["Missing_EPS_Fail"] = mask_nan_eps
    
    df = df.sort_values("Alpha_Score", ascending=False).reset_index(drop=True)
    
    daily_path = DAILY_DIR / OUTPUT_FILENAME
    df.to_csv(daily_path, index=False)
    df.to_csv(LATEST_CSV, index=False)
    
    print(f"Saved to {daily_path}")

    # --- TRACKING ---
    daily_files = sorted(DAILY_DIR.glob("alpha_v7_results_*.csv"), key=lambda p: p.name)
    if not daily_files:
        daily_files = sorted(DAILY_DIR.glob("alpha_v6_results_*.csv"), key=lambda p: p.name)

    diff_data = {"new_entrants": [], "movers": []}
    
    if len(daily_files) >= 2:
        prev_file = daily_files[-2]
        try:
            prev_df = pd.read_csv(prev_file)
            prev_df = prev_df.sort_values("Alpha_Score", ascending=False).reset_index(drop=True)
            
            top_today = df.head(20)["Ticker"].tolist()
            top_prev = prev_df.head(20)["Ticker"].tolist()
            
            new_entrants = [t for t in top_today if t not in top_prev]
            diff_data["new_entrants"] = new_entrants
            
            rank_today = {t: i for i, t in enumerate(df["Ticker"], start=1)}
            rank_prev = {t: i for i, t in enumerate(prev_df["Ticker"], start=1)}
            
            movers = []
            for t in top_today:
                if t in rank_prev:
                    change = rank_prev[t] - rank_today[t] 
                    movers.append({"ticker": t, "change": int(change)})
                else:
                    movers.append({"ticker": t, "change": "New"})
            
            diff_data["movers"] = movers
            
            with open(DIFF_FILE, "w") as f:
                json.dump(diff_data, f)
                
        except Exception as e:
            print(f"Diff generation failed: {e}")

# LOGGING
runtime_sec = int(time.time() - RUN_START)

ev_missing_pct = 0.0
roic_missing_pct = 0.0

if "df" in locals() and isinstance(df, pd.DataFrame) and not df.empty:
    if "EV_EBITDA_Used" in df.columns:
        ev_missing_pct = float(df["EV_EBITDA_Used"].isna().mean())
    if "ROIC" in df.columns:
        roic_missing_pct = float(df["ROIC"].isna().mean())

log_obj = {
    "date": today_str,
    "runtime_sec": runtime_sec,
    "universe": len(universe),
    "eligible": len(rows),
    "exclusions": tracker.stats,
    "data_quality": {
        "ev_missing_pct": round(ev_missing_pct * 100.0, 2),
        "roic_missing_pct": round(roic_missing_pct * 100.0, 2)
    }
}
with open(RUN_LOG_FILE, "a") as f:
    f.write(json.dumps(log_obj) + "\n")
