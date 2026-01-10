import os
import io
import json
import math
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# =========================
# CONFIGURATION & SETUP
# =========================
RUN_START = time.time()
load_dotenv()

# GitHub Secrets sometimes come in without quotes, this handles both
API_KEY = os.getenv("FINNHUB_API_KEY")
if not API_KEY:
    print("⚠️  Warning: FINNHUB_API_KEY not found. Script may fail.")

BASE_URL = "https://finnhub.io/api/v1"

# FOLDER SETUP (Cloud Friendly)
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DAILY_DIR = RESULTS_DIR / "daily"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# FILE PATHS
HISTORY_FILE = DATA_DIR / "alpha_picks_history.json"
GICS_CACHE_FILE = DATA_DIR / "sp1500_gics_map.json"
RUN_LOG_FILE = RESULTS_DIR / "run_log.jsonl"
LATEST_CSV = RESULTS_DIR / "latest.csv"

# FILTERS
MIN_PRICE = 10.0
MIN_MKTCAP_MUSD = 500.0
EXCLUDE_NON_US = True
EXCLUDE_REITS = True
MIN_ANALYSTS = 5

# SENTIMENT
FLOOR_BUY_RATIO = 0.30
STREAK_BUY_RATIO = 0.55
MAX_STREAK_DAYS = 75

# WEIGHTS
WEIGHT_FUNDAMENTALS = 0.70
WEIGHT_SENTIMENT = 0.30

# API SETTINGS
MAX_CALLS_PER_MIN = float(os.getenv("FINNHUB_MAX_CALLS_PER_MIN", "55"))
MIN_INTERVAL = 60.0 / MAX_CALLS_PER_MIN

print("\n--- ☁️ ALPHA-BOT CLOUD RUNNER (V5.2) ☁️ ---\n")

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

class HistoryManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load()
        self.today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def update_streak(self, ticker, buy_ratio):
        record = self.data.get(ticker, {"streak": 0, "last_processed": "", "last_increment": ""})
        streak = record.get("streak", 0)
        last_processed = record.get("last_processed", "")

        if last_processed == self.today_str:
            return streak

        if buy_ratio < FLOOR_BUY_RATIO:
            self.data[ticker] = {
                "streak": 0,
                "last_processed": self.today_str,
                "last_increment": record.get("last_increment", "")
            }
            return 0
        elif buy_ratio >= STREAK_BUY_RATIO:
            new_streak = streak + 1
            self.data[ticker] = {
                "streak": new_streak,
                "last_processed": self.today_str,
                "last_increment": self.today_str
            }
            return new_streak
        else:
            self.data[ticker] = {
                "streak": streak,
                "last_processed": self.today_str,
                "last_increment": record.get("last_increment", "")
            }
            return streak

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f)

class GICSManager:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.mapping = self._load_or_scrape()

    def _load_or_scrape(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    print("✅ Loaded GICS from cache.")
                    return json.load(f)
            except:
                pass
        return self._scrape_wikipedia()

    def _scrape_wikipedia(self):
        print("🌍 Scraping Wikipedia GICS Data...")
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
                print(f"⚠️ Error {url}: {e}")

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

limiter = RateLimiter(60.0 / 55.0)
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

# =========================
# MAIN LOGIC
# =========================
history = HistoryManager(HISTORY_FILE)
gics = GICSManager(GICS_CACHE_FILE)
tracker = ExclusionTracker()
universe = gics.get_universe()

print(f"Scanning {len(universe)} tickers...")
rows = []
today = datetime.now(timezone.utc).date()
today_str = today.strftime("%Y-%m-%d")

for i, sym in enumerate(universe):
    if i % 50 == 0: print(f"Processing {i}/{len(universe)}...", end="\r")
    
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
    price = safe_num(q.get("c")) if q else math.nan
    if pd.isna(price) or price < MIN_PRICE:
        tracker.log("Low_Price"); continue

    # 4. Analyst
    rec = finnhub_get("/stock/recommendation", {"symbol": sym})
    rec_total, buy_ratio, rec_change = 0, 0.0, math.nan
    
    if rec and isinstance(rec, list) and len(rec) >= 1:
        rec.sort(key=lambda x: x.get("period", ""), reverse=True)
        d = rec[0]
        sb, b = float(d.get("strongBuy",0)), float(d.get("buy",0))
        total = sb + b + float(d.get("hold",0)) + float(d.get("sell",0)) + float(d.get("strongSell",0))
        if total > 0:
            rec_total = total
            buy_ratio = (sb + b) / total
            # Rec change logic
            target = today - timedelta(days=90)
            for row in rec:
                try:
                    if datetime.strptime(row.get("period",""), "%Y-%m-%d").date() <= target:
                        d_old = row
                        t_old = sum([float(d_old.get(k,0)) for k in ["strongBuy","buy","hold","sell","strongSell"]])
                        if t_old > 0:
                            old_r = (float(d_old.get("strongBuy",0))+float(d_old.get("buy",0)))/t_old
                            rec_change = buy_ratio - old_r
                        break
                except: continue

    if rec_total < MIN_ANALYSTS:
        tracker.log("No_Analyst_Coverage"); continue
    if buy_ratio < FLOOR_BUY_RATIO:
        history.update_streak(sym, buy_ratio)
        tracker.log("Below_Floor_Ratio"); continue
    
    current_streak = history.update_streak(sym, buy_ratio)

    # 5. Metrics
    metric = (finnhub_get("/stock/metric", {"symbol": sym, "metric": "all"}) or {}).get("metric", {})
    
    # 6. Data Row
    pe = safe_num(metric.get("peBasicExclExtraTTM"))
    ps = safe_num(metric.get("psTTM"))
    rev_g = safe_num(metric.get("revenueGrowthTTMYoy"))
    if pd.isna(rev_g): rev_g = safe_num(metric.get("revenueGrowth5Y"))
    
    h52 = safe_num(metric.get("52WeekHigh"))
    l52 = safe_num(metric.get("52WeekLow"))
    rng = (price - l52)/(h52 - l52) if (h52 > l52) else math.nan

    rows.append({
        "Ticker": sym, "Sector": gics_data["Sector"], "SubIndustry": gics_data["SubIndustry"],
        "Price": price,
        "PE_Rank": pe if (pe and pe > 0) else 1e6,
        "PS_Rank": ps if (ps and ps > 0) else 1e6,
        "EPS_Growth": safe_num(metric.get("epsGrowth5Y")),
        "Rev_Growth": rev_g,
        "ROE": safe_num(metric.get("roeTTM")),
        "Op_Margin": safe_num(metric.get("operatingMarginTTM")),
        "Mom_Range": max(0.0, min(1.0, rng)),
        "Mom_3M": safe_num(metric.get("3MonthPriceReturnDaily")),
        "Mom_12M": safe_num(metric.get("52WeekPriceReturnDaily")),
        "Rev_Change": rec_change,
        "Rec_BuyRatio": buy_ratio,
        "Streak_Days": current_streak
    })

history.save()

# =========================
# SCORING & OUTPUT
# =========================
if rows:
    df = pd.DataFrame(rows)
    print(f"\n✅ Scoring {len(df)} eligible stocks...")
    
    # Ranks
    df["Value_S"] = pd.concat([blended_rank(df,"PE_Rank","SubIndustry",False), 
                               blended_rank(df,"PS_Rank","SubIndustry",False)], axis=1).fillna(50).mean(axis=1)
    
    df["Growth_S"] = pd.concat([blended_rank(df,"EPS_Growth","SubIndustry",True), 
                                blended_rank(df,"Rev_Growth","SubIndustry",True)], axis=1).fillna(50).mean(axis=1)
    
    df["Prof_S"] = pd.concat([blended_rank(df,"ROE","SubIndustry",True), 
                              blended_rank(df,"Op_Margin","SubIndustry",True)], axis=1).fillna(50).mean(axis=1)
    
    m1 = blended_rank(df,"Mom_Range","SubIndustry",True).fillna(50)
    m2 = blended_rank(df,"Mom_3M","SubIndustry",True).fillna(50)
    m3 = blended_rank(df,"Mom_12M","SubIndustry",True).fillna(50)
    df["Mom_S"] = (m1 * 0.4) + (m2 * 0.4) + (m3 * 0.2)
    
    df["Rev_S"] = blended_rank(df,"Rev_Change","SubIndustry",True).fillna(50)

    # Aggregates
    fund_cols = ["Value_S", "Growth_S", "Prof_S", "Mom_S", "Rev_S"]
    df["Fundamental_Score"] = df[fund_cols].mean(axis=1)
    
    df["Consensus_Score"] = blended_rank(df,"Rec_BuyRatio","SubIndustry",True).fillna(50)
    df["Streak_Score"] = (df["Streak_Days"].clip(upper=MAX_STREAK_DAYS)/MAX_STREAK_DAYS)*100
    df["Sentiment_Score"] = (df["Consensus_Score"]*0.6) + (df["Streak_Score"]*0.4)
    
    df["Alpha_Score"] = (df["Fundamental_Score"]*WEIGHT_FUNDAMENTALS) + (df["Sentiment_Score"]*WEIGHT_SENTIMENT)
    
    # Gate
    df["Min_Fund_Factor"] = df[fund_cols].min(axis=1)
    df.loc[df["Min_Fund_Factor"] < 25, "Alpha_Score"] = df.loc[df["Min_Fund_Factor"] < 25, "Alpha_Score"].clip(upper=60)
    
    df = df.sort_values("Alpha_Score", ascending=False)
    
    # Save Results
    daily_path = DAILY_DIR / f"alpha_v5_results_{today_str}.csv"
    df.to_csv(daily_path, index=False)
    df.to_csv(LATEST_CSV, index=False)
    
    print(f"✅ Saved to {daily_path}")

# LOGGING
runtime_sec = int(time.time() - RUN_START)
log_obj = {
    "date": today_str,
    "runtime_sec": runtime_sec,
    "universe": len(universe),
    "eligible": len(rows),
    "exclusions": tracker.stats
}
with open(RUN_LOG_FILE, "a") as f:
    f.write(json.dumps(log_obj) + "\n")
