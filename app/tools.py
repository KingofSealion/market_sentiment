import os
import re
import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# [수정/신규] 날짜, 품목, 숫자(수량)를 한 번에 파싱하는 통합 유틸리티 함수
def parse_query_details(query: str) -> Dict[str, Any]:
    """
    사용자 질의에서 날짜(범위 포함), 품목, 특정 숫자(수량) 정보를 한번에 추출합니다.
    - parse_date_and_commodity와 CommodityCalculator.parse_dates 기능을 통합 및 개선했습니다.
    """
    q_lower = query.lower()
    result = {
        "dates": [],          # 추출된 날짜 (최대 2개)
        "commodity_name": None, # 품목 영문명
        "value": None,        # 추출된 숫자 (수량)
        "unit": None          # 추출된 숫자의 단위
    }

    # 1. 품목명 추출 (한글 -> 영문 매핑)
    commodity_mapping = {
        "옥수수": "Corn", "corn": "Corn",
        "대두": "Soybean", "soybean": "Soybean",
        "대두박": "Soybean Meal", "soybean meal": "Soybean Meal",
        "대두유": "Soybean Oil", "soybean oil": "Soybean Oil",
        "소맥": "Wheat", "밀": "Wheat", "wheat": "Wheat",
        "팜오일": "Palm Oil", "팜유": "Palm Oil", "palm oil": "Palm Oil"
    }
    for korean_name, english_name in commodity_mapping.items():
        if korean_name.lower() in q_lower:
            result["commodity_name"] = english_name
            break

    # 2. 날짜 추출 (상대 날짜, 절대 날짜, 기간 모두 처리)
    today = datetime.now()
    dates = []
    
    # 절대 날짜 (YYYY-MM-DD, YYYY.MM.DD 등)
    # 정규식 수정: '월'과 '일' 사이 공백도 허용
    abs_dates = re.findall(r'(\d{4})[년.\-\s]+(\d{1,2})[월.\-\s]+(\d{1,2})일?', query)
    for d in abs_dates:
        y, m, day = map(int, d)
        try:
            dates.append(datetime(y, m, day).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    
    # 월/일만 있는 경우 (올해 년도 사용)
    # 정규식 수정: '년'이 없는 경우만 매칭되도록 Negative lookbehind 사용
    month_day_dates = re.findall(r'(?<!\d{4}년\s)(\d{1,2})[월.\-\s]+(\d{1,2})일?', query)
    for d in month_day_dates:
        m, day = map(int, d)
        # 이미 위에서 YYYY-MM-DD 형태로 파싱된 경우는 제외
        temp_date_str = f"{today.year}-{m:02d}-{day:02d}"
        if temp_date_str not in dates:
            try:
                dates.append(datetime(today.year, m, day).strftime("%Y-%m-%d"))
            except ValueError:
                continue

    # 상대적 날짜
    if "오늘" in query or "today" in q_lower:
        dates.append(today.strftime("%Y-%m-%d"))
    if "어제" in query or "yesterday" in q_lower:
        dates.append((today - timedelta(days=1)).strftime("%Y-%m-%d"))
    if "내일" in query or "tomorrow" in q_lower:
        dates.append((today + timedelta(days=1)).strftime("%Y-%m-%d"))
    
    # "n일 전/후"
    match_days_ago = re.findall(r'(\d+)\s*일\s*(전|뒤|후)', query)
    for n, direction in match_days_ago:
        n_days = int(n)
        if direction == "전":
            dates.append((today - timedelta(days=n_days)).strftime("%Y-%m-%d"))
        else: # 후, 뒤
            dates.append((today + timedelta(days=n_days)).strftime("%Y-%m-%d"))

    # 중복 제거 후 최대 2개까지 저장
    result["dates"] = sorted(list(set(dates)))[:2]

    # '최근' 키워드 처리
    if "최근" in query or "최신" in query or "가장 최근" in query:
        result["date_range"] = "recent"
    
    # 3. 숫자와 단위 추출 (수량 변환용)
    # 예: "150.5 부셸", "95.2 million acres", "200 톤", "150 부셸은"
    value_match = re.search(r"([\d\.\,]+)\s*(million)?\s*(bushels? per acre|bu/acre|bushels?|부셸|bu|톤|ton|acres?|에이커|헥타르|hectare)", q_lower)
    if value_match:
        value_str = value_match.group(1).replace(",", "")
        val = float(value_str)
        
        if value_match.group(2) and "million" in value_match.group(2):
            val *= 1_000_000
            
        result["value"] = val
        result["unit"] = value_match.group(3).strip()

    return result


def batch(iterable, batch_size=500):
    """리스트를 batch_size씩 잘라서 반환하는 유틸 함수"""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


# [수정] 새로운 통합 파서(parse_query_details)를 사용하도록 로직 변경
def sql_query_tool(query: str) -> str:
    """
    SQL을 사용하여 정확한 날짜+품목 기반 데이터를 검색합니다.
    """
    try:
        # 새로운 통합 파서로 쿼리 분석
        parsed = parse_query_details(query)
        
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            port=os.environ.get("DB_PORT")
        )
        cursor = conn.cursor()
        
        results = []
        
        # 1. 일일 시장 요약 데이터 검색
        summary_query = """
        SELECT dms.date, c.name as commodity_name, dms.daily_sentiment_score, 
               dms.daily_reasoning, dms.daily_keywords, dms.analyzed_news_count
        FROM daily_market_summary dms
        JOIN commodities c ON dms.commodity_id = c.id
        WHERE 1=1
        """
        params = []
        
        # 파싱된 날짜가 있으면 첫 번째 날짜를 기준으로 검색
        if parsed["dates"]:
            summary_query += " AND dms.date = %s"
            params.append(parsed["dates"][0])
        elif parsed.get("date_range") == "recent":
            # 먼저 최신 날짜를 조회
            cursor.execute("SELECT MAX(date) FROM daily_market_summary")
            latest_date = cursor.fetchone()[0]
            
            # 우선순위별 쿼리 의도 분석
            q_lower = query.lower()
            
            # 1. 명시적 기간 키워드 (최우선)
            period_keywords = ["일주일", "한주일", "7일", "일주", "한달", "30일", "며칠", "몇일"]
            has_period = any(keyword in q_lower for keyword in period_keywords)
            
            # 2. 변화/비교/트렌드 키워드
            trend_keywords = ["변화", "추이", "트렌드", "비교", "동향", "흐름", "패턴", "움직임"]
            has_trend = any(keyword in q_lower for keyword in trend_keywords)
            
            # 3. 단일 시점 키워드
            single_keywords = ["현재", "지금", "오늘", "얼마", "몇점"]
            has_single = any(keyword in q_lower for keyword in single_keywords)
            
            if has_period or (has_trend and not has_single):
                # 기간 데이터 또는 트렌드 요청 - 최신일부터 7일간
                summary_query += " AND dms.date >= %s - INTERVAL '6 days' AND dms.date <= %s"
                params.extend([latest_date, latest_date])
            else:
                # 단일 최신 데이터 요청
                summary_query += " AND dms.date = %s"
                params.append(latest_date)
        
        if parsed["commodity_name"]:
            summary_query += " AND c.name = %s"
            params.append(parsed["commodity_name"])
        
        summary_query += " ORDER BY dms.date DESC LIMIT 10"
        
        cursor.execute(summary_query, tuple(params))
        summary_results = cursor.fetchall()
        
        if summary_results:
            # 요약 데이터의 기간 정보 명시
            if len(summary_results) > 1:
                first_date = summary_results[-1][0]  # 가장 오래된 날짜
                last_date = summary_results[0][0]    # 가장 최신 날짜
                results.append(f"=== 일일 시장 요약 (분석기간: {first_date} ~ {last_date}) ===")
            else:
                summary_date = summary_results[0][0]
                results.append(f"=== 일일 시장 요약 (분석일자: {summary_date}) ===")
            
            for row in summary_results:
                date, commodity, score, reasoning, keywords, news_count = row
                results.append(f"날짜: {date}")
                results.append(f"품목: {commodity}")
                results.append(f"감정점수: {score}")
                results.append(f"시장 동향: {reasoning}")
                results.append(f"주요 키워드: {keywords}")
                results.append(f"분석된 뉴스 수: {news_count}")
                results.append("---")
        
        # 2. 개별 뉴스 분석 데이터 검색 (영향도 기준 개선)
        if parsed["commodity_name"]:
            news_query = """
            SELECT r.title, r.published_time, nar.sentiment_score, 
                   nar.reasoning, nar.keywords, c.name as commodity_name,
                   ABS(nar.sentiment_score - 50) as impact_score
            FROM raw_news r
            JOIN news_analysis_results nar ON r.id = nar.raw_news_id
            JOIN commodities c ON nar.commodity_id = c.id
            WHERE c.name = %s
            """
            params = [parsed["commodity_name"]]
            
            # 뉴스 조회도 동일한 로직 적용
            if parsed["dates"]:
                news_query += " AND DATE(r.published_time) = %s"
                params.append(parsed["dates"][0])
            elif parsed.get("date_range") == "recent":
                # 특정 품목의 최신 뉴스 날짜 기준으로 7일간 조회
                cursor.execute("""
                    SELECT MAX(DATE(r.published_time)) 
                    FROM raw_news r 
                    JOIN news_analysis_results nar ON r.id = nar.raw_news_id
                    JOIN commodities c ON nar.commodity_id = c.id
                    WHERE c.name = %s AND r.analysis_status = TRUE
                """, [parsed["commodity_name"]])
                latest_news_date = cursor.fetchone()[0]
                if latest_news_date:
                    news_query += " AND DATE(r.published_time) >= %s - INTERVAL '6 days' AND DATE(r.published_time) <= %s"
                    params.extend([latest_news_date, latest_news_date])
                else:
                    # 해당 품목의 뉴스가 없으면 전체에서 최근 7일
                    news_query += " AND r.published_time >= CURRENT_DATE - INTERVAL '7 days'"
            
            # 영향도 높은 뉴스 우선 (감정점수가 50에서 멀수록 영향도 높음) + 최신순
            news_query += " ORDER BY impact_score DESC, r.published_time DESC LIMIT 10"
            
            cursor.execute(news_query, tuple(params))
            news_results = cursor.fetchall()
            
            if news_results:
                # 기간 정보 계산 및 명시
                if len(news_results) > 1:
                    first_date = news_results[-1][1].date()  # 가장 오래된 뉴스 날짜
                    last_date = news_results[0][1].date()    # 가장 최신 뉴스 날짜
                    results.append(f"\n=== 관련 뉴스 분석 (분석기간: {first_date} ~ {last_date}) ===")
                else:
                    news_date = news_results[0][1].date()
                    results.append(f"\n=== 관련 뉴스 분석 (분석일자: {news_date}) ===")
                
                for row in news_results:
                    title, pub_time, score, reasoning, keywords, commodity, impact = row
                    results.append(f"제목: {title}")
                    results.append(f"발행시간: {pub_time}")
                    results.append(f"품목: {commodity}")
                    results.append(f"감정점수: {score} (영향도: {impact:.1f})")
                    results.append(f"분석 근거: {reasoning}")
                    results.append(f"키워드: {keywords}")
                    results.append("---")
        
        # 3. 가격 데이터 검색 
        # 크러시 마진 쿼리이거나 대두 관련 복수 품목 쿼리인 경우 대두, 대두유, 대두박 모든 가격 조회
        if ("크러시" in query and "마진" in query) or ("대두" in query and ("대두박" in query or "대두유" in query)):
            crush_commodities = ["Soybean", "Soybean Oil", "Soybean Meal"]
            crush_prices = []
            
            for commodity in crush_commodities:
                price_query = """
                SELECT ph.date, ph.closing_price, c.name as commodity_name
                FROM price_history ph
                JOIN commodities c ON ph.commodity_id = c.id
                WHERE c.name = %s
                """
                params = [commodity]
                
                if parsed["dates"]:
                    price_query += " AND ph.date = %s"
                    params.append(parsed["dates"][0])
                elif parsed.get("date_range") == "recent":
                    price_query += " AND ph.date >= CURRENT_DATE - INTERVAL '7 days'"
                
                price_query += " ORDER BY ph.date DESC LIMIT 1"
                
                cursor.execute(price_query, tuple(params))
                price_result = cursor.fetchone()
                if price_result:
                    crush_prices.append(price_result)
            
            if crush_prices:
                results.append("\n=== 크러시 마진 계산용 가격 정보 ===")
                for date, price, commodity in crush_prices:
                    results.append(f"날짜: {date}, 품목: {commodity}, 종가: {price}")
                    
        elif parsed["commodity_name"]:
            # 일반적인 단일 품목 가격 조회
            price_query = """
            SELECT ph.date, ph.closing_price, c.name as commodity_name
            FROM price_history ph
            JOIN commodities c ON ph.commodity_id = c.id
            WHERE c.name = %s
            """
            params = [parsed["commodity_name"]]
            
            if parsed["dates"]:
                price_query += " AND ph.date = %s"
                params.append(parsed["dates"][0])
            elif parsed.get("date_range") == "recent":
                price_query += " AND ph.date >= CURRENT_DATE - INTERVAL '7 days'"
            
            price_query += " ORDER BY ph.date DESC LIMIT 5"
            
            cursor.execute(price_query, tuple(params))
            price_results = cursor.fetchall()
            
            if price_results:
                results.append("\n=== 가격 정보 ===")
                for row in price_results:
                    date, price, commodity = row
                    results.append(f"날짜: {date}, 품목: {commodity}, 종가: {price}")
        
        conn.close()
        
        if results:
            return "\n".join(results)
        else:
            return f"'{query}'에 대한 정확한 데이터를 찾을 수 없습니다. 날짜나 품목명을 더 구체적으로 지정해주세요."
    
    except Exception as e:
        return f"데이터베이스 검색 중 오류가 발생했습니다: {str(e)}"

class CommodityCalculator:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            port=os.environ.get("DB_PORT")
        )
        self.commodity_id_map = {
            "corn": 1, "옥수수": 1,
            "wheat": 2, "밀": 2, "소맥": 2,
            "soybean": 3, "대두": 3,
            "soybean meal": 4, "대두박": 4,
            "soybean oil": 5, "대두유": 5,
            "palm oil": 6, "팜유": 6
        }
        self.conv_factors = {
            1: 0.0254,      # corn (ton/bushel)
            2: 0.0272155,   # wheat (ton/bushel)
            3: 0.0272155    # soybean (ton/bushel)
        }
        self.acre_to_hectare = 0.4046856422
        self.hectare_to_acre = 2.47105

    def get_commodity_id(self, name: str) -> int | None:
        return self.commodity_id_map.get(name.lower())

    def fetch_price(self, commodity_id: int, date: str = None) -> Tuple[float | None, str | None]:
        cursor = self.conn.cursor()
        try:
            sql = """
                SELECT closing_price, date FROM price_history
                WHERE commodity_id = %s
            """
            params = [commodity_id]
            if date:
                sql += " AND date <= %s"
                params.append(date)
            
            sql += " ORDER BY date DESC LIMIT 1"
            
            cursor.execute(sql, tuple(params))
            result = cursor.fetchone()
            return (float(result[0]), result[1].strftime('%Y-%m-%d')) if result else (None, None)
        except Exception as e:
            print(f"가격 조회 오류: {e}")
            return None, None
        finally:
            cursor.close()

    # [수정] 부셸↔톤 변환 로직: 가격이 아닌 수량을 기준으로 변환
    def bushel_to_ton(self, value: float, commodity_id: int) -> float | None:
        if commodity_id not in self.conv_factors:
            return None
        return round(value * self.conv_factors[commodity_id], 2)

    def ton_to_bushel(self, value: float, commodity_id: int) -> float | None:
        if commodity_id not in self.conv_factors:
            return None
        return round(value / self.conv_factors[commodity_id], 2)

    # [수정] USc/bu → USD/MT 변환 오류 수정 (불필요한 * 1000 제거)
    def uscbushel_to_usdmt(self, price_uscbu: float, commodity_id: int) -> float | None:
        if commodity_id not in self.conv_factors:
            return None
        # 1. 센트 -> 달러
        price_usdbu = price_uscbu / 100
        # 2. 부셸당 가격 -> 톤당 가격
        price_usd_per_ton = price_usdbu / self.conv_factors[commodity_id]
        return round(price_usd_per_ton, 2)

    # [수정] 크러시 마진 계산 시 불안정한 로직(if s1 > 10) 제거
    def board_crush_margin(self, date=None) -> Tuple | None:
        sm_price, _ = self.fetch_price(self.get_commodity_id("soybean meal"), date)
        so_price, _ = self.fetch_price(self.get_commodity_id("soybean oil"), date)
        s_price, _ = self.fetch_price(self.get_commodity_id("soybean"), date)

        if sm_price and so_price and s_price:
            # 중요: DB의 대두(soybean) 가격은 항상 센트(cents) 단위라고 가정합니다.
            # 데이터 단위의 일관성이 매우 중요합니다.
            s_price_dollar = s_price / 100
            
            # 크러시 마진 계산: (대두박 가격 * 0.022) + (대두유 가격 * 0.11) - 대두 가격($)
            margin = round((sm_price * 0.022) + (so_price * 0.11) - s_price_dollar, 2)
            return (margin, sm_price, so_price, s_price)
        return None

    def acre_to_hectare_func(self, acre: float) -> float:
        return round(acre * self.acre_to_hectare, 2)

    def hectare_to_acre_func(self, hectare: float) -> float:
        return round(hectare * self.hectare_to_acre, 2)

    def yield_bu_acre_to_ton_hectare(self, yield_bu_per_acre: float, commodity_id: int) -> float | None:
        ton_per_acre = self.bushel_to_ton(yield_bu_per_acre, commodity_id)
        if ton_per_acre is not None:
            # 톤/에이커 -> 톤/헥타르
            return round(ton_per_acre / self.acre_to_hectare, 3)
        return None

    def basis_to_flat(self, fut_price_uscbu: float, basis_cents: float, commodity_id: int) -> float | None:
        if commodity_id not in self.conv_factors:
            return None
        flat_price_uscbu = fut_price_uscbu + basis_cents
        return self.uscbushel_to_usdmt(flat_price_uscbu, commodity_id)

    # [수정] 전체 계산 로직을 새로운 통합 파서(parse_query_details) 기반으로 재구성
    def calculate(self, query: str) -> str:
        parsed = parse_query_details(query)
        q = query.lower()
        
        comm_name = parsed.get("commodity_name")
        comm_id = self.get_commodity_id(comm_name) if comm_name else None

        # 1. 크러시 마진 계산 (대두 관련 키워드 있을 시)
        if "크러시" in q and "마진" in q:
            dates = parsed.get("dates", [])
            if len(dates) >= 2: # 날짜 2개 비교
                margin1_data = self.board_crush_margin(dates[0])
                margin2_data = self.board_crush_margin(dates[1])
                if margin1_data and margin2_data:
                    margin1, margin2 = margin1_data[0], margin2_data[0]
                    diff = round(margin2 - margin1, 2)
                    pct = round(diff / abs(margin1) * 100, 2) if margin1 != 0 else 0
                    return (
                        f"[대두 크러시 마진 비교]\n"
                        f"- {dates[0]}: ${margin1} /bu\n"
                        f"- {dates[1]}: ${margin2} /bu\n"
                        f"- 변화: ${diff} /bu ({pct}%)"
                    )
                return "두 날짜의 데이터가 모두 필요하여 크러시 마진을 비교할 수 없습니다."
            else: # 날짜 1개 또는 미지정
                target_date = dates[0] if dates else None
                margin_data = self.board_crush_margin(target_date)
                if margin_data:
                    price_date, _ = self.fetch_price(self.get_commodity_id("soybean"), target_date) # 실제 가격 날짜 조회
                    return f"[{price_date or '최근'} 기준] 대두 크러시 마진: ${margin_data[0]} /bu"
                return "데이터가 부족해 크러시 마진을 계산할 수 없습니다."

        # [신규/수정] 2. 수량 단위 변환 (부셸↔톤)
        if comm_id and parsed.get("value") is not None and parsed.get("unit") in ["bushels", "bu", "부셸", "톤", "ton"]:
            value = parsed["value"]
            unit = parsed["unit"]
            
            if unit in ["bushels", "bu", "부셸"]: # 부셸 -> 톤
                converted = self.bushel_to_ton(value, comm_id)
                if converted:
                    return f"[{comm_name}] {value:,.2f} 부셸(bu) = {converted:,.4f} 톤(ton)"
                return f"{comm_name} 품목은 부셸-톤 변환을 지원하지 않습니다."
            
            elif unit in ["톤", "ton"]: # 톤 -> 부셸
                converted = self.ton_to_bushel(value, comm_id)
                if converted:
                    return f"[{comm_name}] {value:,.2f} 톤(ton) = {converted:,.4f} 부셸(bu)"
                return f"{comm_name} 품목은 톤-부셸 변환을 지원하지 않습니다."

        # 3. 가격 단위 변환 (USc/bu → USD/MT)
        if comm_id and "usd/mt" in q or "톤당 달러" in q:
            price, price_date = self.fetch_price(comm_id)
            if price and price_date:
                usdmt = self.uscbushel_to_usdmt(price, comm_id)
                return (
                    f"[{comm_name} 가격 변환 ({price_date} 기준)]\n"
                    f"- 선물가: {price} USc/bu\n"
                    f"- 변환가: ${usdmt} /MT"
                )
            return f"{comm_name} 가격 정보가 없습니다."

        # 4. 베이시스 플랫 가격 계산
        basis_match = re.search(r"(베이시스|basis)\s*([+-]?\d+)", q)
        if comm_id and basis_match:
            basis = float(basis_match.group(2))
            price, price_date = self.fetch_price(comm_id)
            if price and price_date:
                flat_price = self.basis_to_flat(price, basis, comm_id)
                return (
                    f"[{comm_name} 플랫가격 ({price_date} 기준)]\n"
                    f"- 최근 선물: {price} USc/bu\n"
                    f"- 베이시스: {basis} cents\n"
                    f"- 플랫가격: ${flat_price} /MT"
                )
            return f"{comm_name} 선물 가격 정보가 없어 플랫가격을 계산할 수 없습니다."

        # 5. 면적 변환 (에이커↔헥타르)
        if parsed.get("value") is not None and parsed.get("unit") in ["acres", "에이커", "hectare", "헥타르"]:
            value = parsed["value"]
            unit = parsed["unit"]
            if unit in ["acres", "에이커"]:
                hectare = self.acre_to_hectare_func(value)
                return f"[면적 변환] {value:,.2f} 에이커(acres) = {hectare:,.2f} 헥타르(hectare)"
            else: # hectare, 헥타르
                acre = self.hectare_to_acre_func(value)
                return f"[면적 변환] {value:,.2f} 헥타르(hectare) = {acre:,.2f} 에이커(acres)"
        
        # 6. 단수 변환 (bu/acre → ton/hectare)
        if comm_id and parsed.get("value") is not None and parsed.get("unit") in ["bushels per acre", "bu/acre"]:
            yield_bu_acre = parsed["value"]
            ton_hectare = self.yield_bu_acre_to_ton_hectare(yield_bu_acre, comm_id)
            if ton_hectare:
                return f"[{comm_name} 단수 변환] {yield_bu_acre} bu/acre = {ton_hectare} ton/hectare"

        return "지원하지 않는 계산이거나, 계산에 필요한 정보(품목, 수량 등)가 부족합니다."


def create_agent_tools(documents: List[Document], llm: ChatOpenAI):
    """
    [도구 상자 생성]
    - 역할: 뉴스 RAG용 검색 Tool 등 에이전트가 사용할 도구 리스트를 반환
    - 향후: 계산기, 날씨 등 추가 도구를 더 쉽게 확장 가능
    """
    # 1. 벡터스토어 persist 디렉토리(임베딩 데이터 저장 위치) 지정
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    print(f"--- [Chroma] 벡터 DB 경로: {persist_dir} ---")

    # 2. 벡터스토어 불러오기(또는 새로 생성)
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        print(f"--- [Chroma] 기존 벡터스토어를 불러왔습니다.")
        # 기존 임베딩된 문서 추적용 set 생성
        existing_ids = set()
        # .get() 메서드로 DB의 모든 메타데이터를 가져와 이미 저장된 문서의 ID를 확인합니다.
        for doc in vectorstore.get()["metadatas"]:
            key = None
            if doc.get("type") == "article_analysis" and doc.get("news_id"):
                # 뉴스 ID와 품목명을 조합하여 고유 키 생성 (하나의 뉴스에 여러 품목 분석이 있을 수 있으므로)
                key = f"article_{doc['news_id']}_{doc.get('commodity_name', '')}"
            elif doc.get("type") == "daily_summary" and doc.get("date") and doc.get("commodity"):
                key = f"summary_{doc['date']}_{doc['commodity']}"
            if key:
                existing_ids.add(key)
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        print(f"--- [Chroma] 새 벡터스토어를 생성했습니다.")
        existing_ids = set()

    # 3. 문서 chunk 분할 (너무 긴 문서는 쪼갬)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 4. 기존에 임베딩된 문서는 제외, 신규 문서만 추출 (증분 업데이트)
    new_splits = []
    for doc in splits:
        meta = doc.metadata
        key = None
        if meta.get("type") == "article_analysis" and meta.get("news_id"):
            key = f"article_{meta['news_id']}_{meta.get('commodity_name', '')}"
        elif meta.get("type") == "daily_summary" and meta.get("date") and meta.get("commodity"):
            key = f"summary_{meta['date']}_{meta['commodity']}"
        
        if key and key not in existing_ids:
            new_splits.append(doc)
            existing_ids.add(key)

    # 5. 신규 문서만 임베딩 추가 및 저장 (배치 단위로)
    print(f"  - 전체 {len(splits)}개 청크 중 신규 {len(new_splits)}개만 임베딩 추가합니다.")
    batch_size = 500  # OpenAI 임베딩 토큰 제한 고려 (안전하게 500 추천)
    if new_splits:
        for batch_docs in batch(new_splits, batch_size):
            vectorstore.add_documents(batch_docs)
        vectorstore.persist() # 변경사항을 디스크에 영구 저장
        print(f"  - 신규 문서 임베딩 및 저장(persist) 완료.")

    # 6. Retriever(검색기) 생성 - 다양한 관점 반영을 위해 mmr 사용
    retriever = vectorstore.as_retriever(
        search_type="mmr",      # 유사도 높은 결과 + 다양한 관점의 결과를 함께 가져오는 방식
        search_kwargs={"k": 6}  # 최종적으로 LLM에 전달할 문서 개수
    )

    # 7. RetrievalQA 체인 생성(뉴스 답변기)
    news_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # 검색된 문서들을 모두 컨텍스트에 넣어 LLM에 전달하는 가장 표준적인 방식
        retriever=retriever,
        return_source_documents=True # 답변의 근거가 된 원본 문서를 함께 반환
    )

    # 8. Tool 객체 생성을 위한 래퍼(wrapper) 함수
    def news_tool_func(input_text: str) -> str:
        """
        RetrievalQA 체인의 출력은 {'result': '답변', 'source_documents': [...] } 형태의 딕셔너리입니다.
        하지만 LangChain 에이전트의 도구는 반드시 문자열(string)을 반환해야 하므로,
        'result' 키의 값만 추출하여 반환하는 함수로 감싸줍니다.
        """
        output = news_qa_chain.invoke(input_text)
        return output["result"]

    # 9. 뉴스 Tool 객체 생성
    news_tool = Tool(
        name="Market News and Summary Search",
        func=news_tool_func,
        description=(
            "최신 원자재 시장 뉴스, 일일 시장 동향, 감성 점수, 분석 요약 정보를 찾아줍니다. "
            "예시: '오늘 옥수수 시장 어때?', '브라질 가뭄 뉴스 요약', '최근 대두 시장 상황이 어떤지 뉴스 근거와 함께 알려줘' 등."
        )
    )

    # 10. SQL 검색 도구 생성
    sql_tool = Tool(
        name="Precise Data Query",
        func=sql_query_tool,
        description=(
            "정확한 날짜와 품목 기반의 데이터 검색에 사용합니다. "
            "특정 날짜의 감정점수, 시장 동향, 가격 정보 등을 정확히 찾아줍니다. "
            "예시: '2025년 7월 10일 옥수수 감정점수', '어제 대두 시장 상황', '최근 밀 가격' 등 "
            "날짜나 품목이 명시된 구체적인 질문에 사용하세요."
        )
    )

    # 11. 계산기 도구 생성
    calculator = CommodityCalculator()
    calc_tool = Tool(
        name="Commodity Calculator",
        func=calculator.calculate,
        description=(
            "농산물 관련 단위 변환 및 계산을 수행합니다. 수량, 면적, 단수, 가격 변환, 크러시 마진, 베이시스 계산을 지원합니다.\n"
            "사용 예시:\n"
            "- 수량 변환: '옥수수 150 부셸은 몇 톤이야?', '밀 200톤을 부셸로'\n"
            "- 가격 변환: '최근 옥수수 가격 톤당 달러로 얼마야?'\n"
            "- 면적/단수: '95.2 million acres를 헥타르로', '183.1 bushels per acre를 톤/헥타르로'\n"
            "- 마진/베이시스: '어제랑 오늘 크러시 마진 비교해줘', '옥수수 베이시스 +20일때 플랫가격은?'"
        )
    )

    # 12. 생성된 도구들을 리스트에 담아 반환 (향후 다른 도구 추가 용이)
    print("--- [도구 준비 완료] (RAG + SQL 하이브리드) ---")
    return [news_tool, sql_tool, calc_tool]


