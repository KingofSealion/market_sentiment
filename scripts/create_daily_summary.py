import os
import sys
import logging
import json
from datetime import date, timedelta
from dotenv import load_dotenv
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import psycopg2
from psycopg2 import pool

# --- 1. 초기 설정: 로깅, 환경 변수, LLM, DB 커넥션 풀 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logging.error(f"필수 환경 변수가 .env 파일에 설정되지 않았습니다: {', '.join(missing_vars)}")
    sys.exit(1)

custom_http_client = httpx.Client(verify=False)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, http_client=custom_http_client, api_key=os.getenv("OPENAI_API_KEY"))

db_conn_info = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **db_conn_info)
    logging.info("데이터베이스 커넥션 풀이 성공적으로 생성되었습니다.")
except psycopg2.OperationalError as e:
    logging.error(f"데이터베이스 연결에 실패했습니다: {e}")
    sys.exit(1)

# --- 2. 일일 요약 생성 함수 ---
def generate_summary_for_day(cur, conn, target_date, target_commodity_id, target_commodity_name):
    """지정된 날짜와 품목에 대한 일일 종합 리포트를 생성하고 DB에 저장합니다."""
    
    logging.info(f"--- {target_commodity_name} ({target_date}) 일일 요약 생성 시작 ---")
    
    # 1. 해당 날짜, 품목의 모든 개별 분석 결과 가져오기
    # 월요일(weekday == 0)인 경우 토, 일, 월 3일치 데이터를 조회하도록 수정
    is_monday = target_date.weekday() == 0
    
    if is_monday:
        # 월요일일 경우, 이전 토요일부터 월요일까지의 데이터를 조회
        start_date = target_date - timedelta(days=2)
        end_date = target_date
        logging.info(f"월요일이므로 {start_date}부터 {end_date}까지 3일치 뉴스를 집계합니다.")
        fetch_query = """
        SELECT nar.sentiment_score, nar.reasoning, nar.keywords
        FROM news_analysis_results AS nar
        JOIN raw_news AS r ON nar.raw_news_id = r.id
        WHERE nar.commodity_id = %s AND r.published_time::date BETWEEN %s AND %s;
        """
        params = (target_commodity_id, start_date, end_date)
    else:
        # 월요일이 아닐 경우, 해당 날짜의 데이터만 조회
        fetch_query = """
        SELECT nar.sentiment_score, nar.reasoning, nar.keywords
        FROM news_analysis_results AS nar
        JOIN raw_news AS r ON nar.raw_news_id = r.id
        WHERE nar.commodity_id = %s AND r.published_time::date = %s;
        """
        params = (target_commodity_id, target_date)

    cur.execute(fetch_query, params)
    results = cur.fetchall()

    if not results:
        logging.warning(f"{target_commodity_name} ({target_date})에 분석된 뉴스가 없습니다.")
        return

    analyzed_news_count = len(results)
    logging.info(f"총 {analyzed_news_count}개의 분석된 뉴스를 찾았습니다.")

    # 2. 가중 평균 점수 계산
    total_weighted_score, total_weight = 0, 0
    all_reasonings, all_keywords = [], {}

    for score, reasoning, keywords in results:
        if score is None: continue
        weight = abs(score - 50) + 1
        total_weighted_score += score * weight
        total_weight += weight
        all_reasonings.append(reasoning)
        if keywords:
            for keyword in keywords:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
    
    daily_sentiment_score = total_weighted_score // total_weight if total_weight > 0 else 50.0

    # 3. GPT를 이용해 Reasoning 및 Keywords 요약 (LangChain 체인 사용)
    logging.info("OpenAI로 일일 리포트 요약 중...")
    
    summarization_prompt = ChatPromptTemplate.from_template(
        """You are a Head Market Analyst. Synthesize the following data for {commodity_name} on {target_date} into a concise daily summary report in JSON format.

        Individual Reasonings: {all_reasonings}
        All Individual Keywords (with frequency): {all_keywords}

        ---
        Please generate a final summary based ONLY on the provided data.
        - "daily_reasoning": A single, coherent English sentence (max 150 chars) summarizing the most dominant market driver of the day.
        - "daily_keywords": A list of the 5 most representative keywords for the day's market theme.

        Output only the final JSON object.
        """
    )
    
    summarization_chain = summarization_prompt | llm | JsonOutputParser()
    summary_result = summarization_chain.invoke({
        "commodity_name": target_commodity_name,
        "target_date": str(target_date),
        "all_reasonings": json.dumps(all_reasonings),
        "all_keywords": json.dumps(all_keywords)
    })
    
    # 4. daily_market_summary 테이블에 결과 저장 
    upsert_query = """
    INSERT INTO daily_market_summary (date, commodity_id, daily_sentiment_score, daily_reasoning, daily_keywords, analyzed_news_count)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (date, commodity_id) DO UPDATE SET
        daily_sentiment_score = EXCLUDED.daily_sentiment_score,
        daily_reasoning = EXCLUDED.daily_reasoning,
        daily_keywords = EXCLUDED.daily_keywords,
        analyzed_news_count = EXCLUDED.analyzed_news_count;
    """
    cur.execute(upsert_query, (
        target_date,
        target_commodity_id,
        daily_sentiment_score,
        summary_result.get('daily_reasoning'),
        json.dumps(summary_result.get('daily_keywords')),
        analyzed_news_count
    ))
    conn.commit()
    logging.info(f"성공적으로 {target_commodity_name} ({target_date}) 일일 요약을 생성/업데이트했습니다.")

# --- 3. 메인 실행 함수 ---
def main():
    """스크립트의 메인 실행 함수"""
    conn = None
    cur = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        # 요약할 작업을 찾을 때 토요일(6)과 일요일(7)을 제외하도록 수정
        # PostgreSQL의 EXTRACT(ISODOW FROM date)는 월요일=1, ..., 일요일=7을 반환합니다.
        find_jobs_query = """
        SELECT DISTINCT r.published_time::date AS date, nar.commodity_id, c.name AS commodity_name
        FROM news_analysis_results nar
        JOIN raw_news r ON nar.raw_news_id = r.id
        JOIN commodities c ON nar.commodity_id = c.id
        LEFT JOIN daily_market_summary dms ON r.published_time::date = dms.date AND nar.commodity_id = dms.commodity_id
        WHERE r.analysis_status = TRUE 
          AND dms.id IS NULL
          AND EXTRACT(ISODOW FROM r.published_time::date) NOT IN (6, 7);
        """
        cur.execute(find_jobs_query)
        jobs_to_do = cur.fetchall()

        if not jobs_to_do:
            logging.info("새로 생성할 일일 요약이 없습니다. 모든 요약이 최신 상태입니다.")
            return

        logging.info(f"총 {len(jobs_to_do)}개의 신규 일일 요약을 생성합니다.")

        # 찾아낸 각 조합에 대해 일일 요약 함수를 실행.
        for job in jobs_to_do:
            target_date, target_commodity_id, target_commodity_name = job
            if target_date and target_commodity_id:
                generate_summary_for_day(cur, conn, target_date, target_commodity_id, target_commodity_name)

    except (Exception, psycopg2.Error) as error:
        logging.error(f"일일 요약 생성 중 에러 발생: {error}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            db_pool.putconn(conn)
            logging.info("DB 커넥션을 풀에 반납했습니다.")

if __name__ == '__main__':
    main()
    if 'db_pool' in locals() and db_pool:
        db_pool.closeall()
        logging.info("모든 데이터베이스 커넥션이 종료되었습니다.")


