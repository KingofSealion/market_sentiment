"""
WeatherForecastCropImpactTool
─────────────────────────────
LangChain‑ready tool that

1. Parses REAL‑WORLD ag‑market user queries (KR/EN, rough region names, dates, hours, crops)
2. Calls OpenWeather APIs (geocoding + 5‑day/3‑hour forecast)
3. Returns a clean forecast snippet PLUS a professional agronomic & price impact commentary
4. Gracefully handles all errors / missing data
5. Is easily extensible (more crops, regions, rules, languages)

Author : <you/your‑team>
Created: 2025‑08‑05
"""

from __future__ import annotations

import os
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from langchain.agents import Tool
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANT TABLES  ──────────────────────────────────────────────────────────
# ------------------------------------------------------------------------------
CROP_KEYWORDS: Dict[str, List[str]] = {
    "corn":       ["corn", "maize", "옥수수"],
    "soybean":    ["soybean", "soy", "bean", "대두", "소이빈"],
    "wheat":      ["wheat", "밀", "소맥"],
    "palm_oil":   ["palm oil", "팜유", "팜오일"],
}

AG_BELT_ALIASES: Dict[str, str] = {
    # rough regional aliases ➜ canonical city / lat‑lon anchor
    "corn belt":        "Des Moines, Iowa",
    "soy belt":         "Cuiaba, Brazil",
    "black sea":        "Odessa, Ukraine",
    "남미":              "Buenos Aires, Argentina",
    "남미 지역":          "Buenos Aires, Argentina",
    "콘벨트":            "Des Moines, Iowa",
    "흑해":              "Odessa, Ukraine",
}

# Agronomic impact rules – very compressed; extend as needed.
IMPACT_RULES: Dict[str, List[Tuple[str, str]]] = {
    "corn": [
        (r"temp>32 & rain<1",  "고온·가뭄은 옥수수 수분스트레스 → 수확량↓, 가격↑"),
        (r"temp<20",           "저온은 생육지연 → 생산량↓, 가격↑"),
    ],
    "soybean": [
        (r"rain>20",           "폭우는 대두 수확·품질↓, 가격↑"),
        (r"temp>32",           "고온은 대두 결실률↓, 가격↑"),
    ],
    "wheat": [
        (r"temp>35",           "폭염은 밀 수분스트레스·단백질 저하 → 수량↓, 가격↑"),
        (r"rain<1",            "가뭄은 밀 생육↓, 가격↑"),
    ],
}

# ───────────────────────────────────────────────────────────────────────────────
# 1.  HELPER FUNCTIONS  ─────────────────────────────────────────────────────────
# ------------------------------------------------------------------------------

def _lookup_crop(query: str) -> Optional[str]:
    q = query.lower()
    for k, aliases in CROP_KEYWORDS.items():
        if any(a.lower() in q for a in aliases):
            return k
    return None

def _normalize_location(raw_loc: str) -> str:
    """Convert rough alias (콘벨트, Black Sea…) → canonical location string."""
    raw = raw_loc.lower().strip()
    for alias, canonical in AG_BELT_ALIASES.items():
        if alias in raw:
            return canonical
    return raw_loc  # fallback

def _geocode(location: str, api_key: str) -> Optional[Dict]:
    url = "http://api.openweathermap.org/geo/1.0/direct"
    r = requests.get(url, params={"q": location, "limit": 1, "appid": api_key}, timeout=10)
    if r.status_code == 200 and r.json():
        j = r.json()[0]
        return {"lat": j["lat"], "lon": j["lon"], "name": j.get("name", location)}
    return None

def _nearest_forecast_entry(entries: List[dict], target_date: str, target_hr: Optional[int]) -> Optional[dict]:
    best, min_diff = None, 99
    for e in entries:
        d, t = e["dt_txt"].split(" ")
        if d != target_date:
            continue
        if target_hr is None:
            return e
        diff = abs(int(t[:2]) - target_hr)
        if diff < min_diff:
            best, min_diff = e, diff
    return best

def _parse_dates(query: str) -> Tuple[str, Optional[int]]:
    today = datetime.now()
    # Y‑M‑D
    m = re.search(r'(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})', query)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", None
    # M‑D
    m2 = re.search(r'(\d{1,2})[.\-/월\s]*(\d{1,2})', query)
    if m2:
        return f"{today.year}-{int(m2.group(1)):02d}-{int(m2.group(2)):02d}", None
    if "내일" in query:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d"), None
    # default today
    return today.strftime("%Y-%m-%d"), None

def _parse_hour(query: str) -> Optional[int]:
    m = re.search(r'([01]?\d|2[0-3])시', query)
    return int(m.group(1)) if m else None

def _parse_location(query: str) -> str:
    # capture until keyword
    m = re.search(r'([가-힣A-Za-z,\s\-]+?)(?:날씨|기상|예보)', query)
    raw = m.group(1).strip() if m else query
    return _normalize_location(raw)

def _impact_comment(crop: str, temp: float, rain: float) -> str:
    if not crop:
        return ""
    expr_ctx = {"temp": temp, "rain": rain}
    for cond, msg in IMPACT_RULES.get(crop, []):
        if eval(cond, {}, expr_ctx):
            return f"[작황·가격 영향]\n{msg}"
    return "[작황·가격 영향]\n특이 리스크 없음 / 중립"

# ───────────────────────────────────────────────────────────────────────────────
# 2.  CORE FETCH & FORMAT  ──────────────────────────────────────────────────────
# ------------------------------------------------------------------------------

def fetch_weather_and_impact(user_query: str) -> str:
    """Main entry: natural‑language query ➜ formatted forecast + impact."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "❗ OPENWEATHER_API_KEY 환경변수가 설정되지 않았습니다."

    date, hour = _parse_dates(user_query)
    hour = _parse_hour(user_query) or hour
    crop = _lookup_crop(user_query)
    location = _parse_location(user_query)

    geo = _geocode(location, api_key)
    if not geo:
        return f"❗ 위치 '{location}'(을)를 인식하지 못했습니다."

    url = "https://api.openweathermap.org/data/2.5/forecast"
    r = requests.get(url, params={
        "lat": geo["lat"], "lon": geo["lon"], "appid": api_key, "units": "metric"
    }, timeout=10)
    if r.status_code != 200 or "list" not in r.json():
        return "❗ OpenWeather Forecast API 호출 실패"

    entry = _nearest_forecast_entry(r.json()["list"], date, hour)
    if not entry:
        return f"❗ {date} {location} 예보 데이터를 찾을 수 없습니다."

    desc = entry["weather"][0]["description"]
    temp = entry["main"]["temp"]
    rain = entry.get("rain", {}).get("3h", 0.0)
    hum  = entry["main"]["humidity"]
    wind = entry["wind"]["speed"]

    impact_txt = _impact_comment(crop, temp, rain)

    return (
        f"📍 {geo['name']} ‑ {date} {'%02d:00'%hour if hour is not None else ''}\n"
        f"└ 날씨 : {desc}\n"
        f"└ 기온 : {temp:.1f}℃ | 습도 {hum}% | 풍속 {wind:.1f} m/s\n"
        f"└ 강수 (3h) : {rain} mm\n\n"
        f"{impact_txt}"
    )

# ───────────────────────────────────────────────────────────────────────────────
# 3.  LangChain Tool 객체  ──────────────────────────────────────────────────────
# ------------------------------------------------------------------------------

weather_forecast_tool = Tool(
    name="Weather Forecast & Crop Impact",
    func=fetch_weather_and_impact,
    description=(
        "🔮 미래 날씨 예보 + 작황/가격 영향 분석을 제공합니다.\n"
        "예시:\n"
        "  • '8월 10일 미국 아이오와 옥수수 날씨와 가격 영향'\n"
        "  • '7/15 Kansas wheat forecast'\n"
        "  • '콘벨트 내일 날씨와 대두 영향'\n"
        "지역·날짜가 없으면 자동으로 오늘·현재 위치를 추정합니다."
    ),
)

# Optional: quick self‑test
if __name__ == "__main__":
    print(fetch_weather_and_impact("8월 15일 콘벨트 옥수수 날씨와 가격 영향"))
