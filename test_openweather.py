# ────────────────────────────────────────────────────────────
# tools.py 에 한 번에 붙여넣기: Weather + Impact (ONE-FILE)
# ────────────────────────────────────────────────────────────

from __future__ import annotations
import os, re, requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.agents import Tool

# (추가) geopy for fuzzy geocoding
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

load_dotenv()
OW_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
TIMEOUT_SEC = 8

# ─────── 1) CONSTANTS ────────────────────────────────────────────────
CROP_ALIASES: Dict[str, List[str]] = {
    "corn":     ["corn", "옥수수", "maize"],
    "soybean":  ["soybean", "beans", "대두", "soya"],
    "soymeal":  ["soybean meal","대두박"],
    "soyoil":   ["soybean oil","대두유"],
    "wheat":    ["wheat","밀","소맥"],
    "palm_oil": ["palm oil","팜유","팜오일"],
}

# Belts & Regions aliases → primary lat/lon
REGION_ANCHORS: Dict[str, Tuple[str,float,float]] = {
    "corn belt":    ("Des Moines, US", 41.59, -93.60),
    "콘벨트":        ("Des Moines, US", 41.59, -93.60),
    "black sea":    ("Odessa, UA",     46.48,  30.73),
    "흑해":          ("Odessa, UA",     46.48,  30.73),
    "brazil":       ("Rondonópolis, BR",-16.47, -54.64),
    "브라질":        ("Rondonópolis, BR",-16.47, -54.64),
    "argentina":    ("Rosario, AR",    -32.95, -60.66),
    "아르헨티나":     ("Rosario, AR",    -32.95, -60.66),
    "south america":("Rondonópolis, BR",-16.47, -54.64),
    "남미":          ("Rondonópolis, BR",-16.47, -54.64),
    "malaysia":     ("Kuantan, MY",      3.80, 103.33),
    "indonesia":    ("Medan, ID",        3.59,  98.67),
}

IMPACT_RULES: Dict[str, List[Tuple[str,str]]] = {
    "corn":     [("temp>32 and rain<1","고온·가뭄 → 스트레스 → 수확↓, 가격↑"),
                 ("temp<20",           "저온 생육지연 → 수확↓, 가격↑")],
    "soybean":  [("rain>25",           "폭우 침수 → 품질↓, 가격↑"),
                 ("temp>33",           "고온 결실불량 → 수확↓, 가격↑")],
    "soymeal":  [("rule:'soybean'",    "대두 원료와 연동")],
    "soyoil":   [("rule:'soybean'",    "대두 원료와 연동")],
    "wheat":    [("temp>35",           "폭염 스트레스 → 수확↓, 가격↑"),
                 ("rain<1",            "가뭄 → 수확↓, 가격↑")],
    "palm_oil": [("rain<2",            "가뭄 → 2~3개월 후 수확↓, 가격↑"),
                 ("rain>30",           "침수·물류 차질 → 가격↑")],
}

# geopy Nominatim
_GEOL = Nominatim(user_agent="agri-weather-tool", timeout=TIMEOUT_SEC)

# ─────── 2) PARSE USER INTENT ─────────────────────────────────────────
def _find_crop(text:str)->Optional[str]:
    tl=text.lower()
    for k,aliases in CROP_ALIASES.items():
        if any(a in tl for a in aliases):
            return k
    return None

def _parse_date_hour(text:str)->Tuple[str,Optional[int]]:
    today=datetime.now().date()
    ymd=re.search(r'(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})', text)
    if ymd:
        d=datetime(int(ymd[1]),int(ymd[2]),int(ymd[3])).date()
    else:
        md=re.search(r'(\d{1,2})[.\-/월\s]*(\d{1,2})', text)
        if md:
            d=datetime(today.year,int(md[1]),int(md[2])).date()
        elif "내일" in text:
            d=today+timedelta(days=1)
        else:
            d=today
    hr=None
    hm=re.search(r'([01]?\d|2[0-3])\s*시', text)
    if hm: hr=int(hm[1])
    return d.isoformat(), hr

def _parse_location(text:str)->str:
    tl=text.lower()
    for alias in REGION_ANCHORS:
        if alias in tl:
            return alias
    m=re.search(r'([A-Za-z가-힣,\s\-]+?)(?:날씨|예보|forecast)', text)
    return m.group(1).strip() if m else text

def _wants_impact(text:str)->bool:
    return any(w in text.lower() for w in ("영향","impact","작황","price","가격"))

# ─────── 3) GEOCODING ─────────────────────────────────────────────────────
def _geocode(loc:str)->Optional[Tuple[str,float,float]]:
    key=loc.lower()
    if key in REGION_ANCHORS:
        return REGION_ANCHORS[key]
    # 1) OpenWeather geocode
    try:
        r=requests.get("http://api.openweathermap.org/geo/1.0/direct",
                       params={"q":loc,"limit":1,"appid":OW_API_KEY},
                       timeout=TIMEOUT_SEC)
        if r.status_code==200 and r.json():
            j=r.json()[0]
            return j.get("name",loc), j["lat"], j["lon"]
    except: pass
    # 2) geopy fallback
    try:
        locobj=_GEOL.geocode(loc)
        if locobj:
            return locobj.address, locobj.latitude, locobj.longitude
    except GeocoderServiceError:
        pass
    return None

# ─────── 4) FORECAST & IMPACT LOGIC ────────────────────────────────────────
def weather_forecast_and_impact(q:str)->str:
    if not OW_API_KEY:
        return "❗ OPENWEATHER_API_KEY가 설정되지 않았습니다."

    date, hr = _parse_date_hour(q)
    crop      = _find_crop(q)
    loc_token = _parse_location(q)
    do_impact = _wants_impact(q) and crop is not None

    geo=_geocode(loc_token)
    if not geo:
        return f"❗ 위치 '{loc_token}' 인식에 실패했습니다."
    name, lat, lon = geo

    # fetch forecast
    try:
        r=requests.get("https://api.openweathermap.org/data/2.5/forecast",
                       params={"lat":lat,"lon":lon,"appid":OW_API_KEY,"units":"metric"},
                       timeout=TIMEOUT_SEC)
        r.raise_for_status()
        fc_list=r.json().get("list",[])
    except Exception as e:
        return f"❗ 기상 API 오류: {e}"

    # pick entry
    entry=None
    if fc_list:
        # find exact date/hour
        best,dmin=None,99
        for e in fc_list:
            d,t=e["dt_txt"].split(" ")
            if d!=date: continue
            if hr is None:
                entry=e; break
            gap=abs(int(t[:2])-hr)
            if gap<dmin: best, dmin = e, gap
        entry = entry or best

    if not entry:
        return f"❗ {date} {name} 예보 데이터가 없습니다."

    # parse weather
    desc=entry["weather"][0]["description"]
    temp=entry["main"]["temp"]
    hum =entry["main"]["humidity"]
    wind=entry["wind"]["speed"]
    rain=entry.get("rain",{}).get("3h",0.0)
    tlabel=f"{hr:02d}:00" if hr is not None else entry["dt_txt"].split()[1][:5]

    # build response
    lines=[
        f"📍 {name} | {date} {tlabel}",
        f" └ 날씨 : {desc}",
        f" └ 기온 : {temp:.1f}℃ | 습도 {hum}% | 풍속 {wind:.1f} m/s",
        f" └ 강수 : {rain} mm (3h)",
    ]
    if do_impact:
        # impact calc
        ctx={"temp":temp,"rain":rain}
        msg="특이 리스크 없음 / 중립"
        for cond,im in IMPACT_RULES.get(crop,[]):
            if cond.startswith("rule:'"):
                base=cond.split("'")[1]
                # reuse soybean logic
                for cnd,im2 in IMPACT_RULES.get(base,[]):
                    if eval(cnd,{},ctx): msg=im2
            else:
                if eval(cond,{},ctx): msg=im
        lines.append(f" └ 품목 : {crop}")
        lines.append(f" └ 영향 : {msg}")
    return "\n".join(lines)

# ─────── 5) TOOL OBJECT ──────────────────────────────────────────────────
weather_forecast_tool = Tool(
    name="Weather Forecast & Crop Impact",
    func=weather_forecast_and_impact,
    description=(
        "미래 날씨 예보 + (선택적) 작황·가격 영향 분석\n"
        "→ '8월 15일 콘벨트 옥수수 날씨', '내일 흑해 밀 날씨 영향'? 등"
    )
)
