import os
import pandas as pd
import psycopg2
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

load_dotenv()


def get_documents_from_postgres() -> List[Document]:
    """
    [데이터 로더]
    - 역할: PostgreSQL DB에 연결하여, RAG가 사용할 'Document' 객체 리스트로 가공함.
    - 핵심 전략:
        1. 개별 뉴스(raw_news)와 그에 대한 원자재별 분석(news_analysis_results)을 JOIN한 결과의 **각 row를 하나의 독립된 문서**로 만듦.
        2. 일일 요약(daily_market_summary) 데이터도 마찬가지로 각 row를 별개의 문서로 만듦.
    """
    print("--- [데이터 로딩] PostgreSQL에서 데이터를 가공합니다... ---")
    documents = []

    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"), dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"), password=os.environ.get("DB_PASSWORD"),
            port=os.environ.get("DB_PORT")
        )

        # --- 1. 개별 뉴스 분석 데이터 가공 ---
        print("  - 개별 뉴스 및 분석 데이터를 가공 중...")
        # ↓ 아래 쿼리의 결과 row 하나가 문서 하나가 됨.
        query_articles = """
        SELECT 
            r.id, r.title, r.content, 
            nar.sentiment_score, nar.reasoning, nar.keywords, c.name as commodity_name,
            r.published_time, r.source
        FROM raw_news as r
        JOIN news_analysis_results nar ON r.id = nar.raw_news_id
        JOIN commodities c ON nar.commodity_id = c.id
        WHERE r.analysis_status = TRUE;
        """
        df_articles = pd.read_sql(query_articles, conn)

        for _, row in df_articles.iterrows():
            # [핵심 수정] 검색에 필요한 모든 텍스트 정보를 page_content에 포함시킴.
            page_content = f"""
            품목: {row['commodity_name']}
            기사 제목: {row['title']}
            발행 시간: {row['published_time']}
            본문: {row['content']}
            감성 점수: {row['sentiment_score']}
            감성 점수 근거: {row['reasoning']}
            주요 키워드: {row['keywords']}
            """
            # metadata에는 필터링이나 참조에 사용할 ID와 출처 등을 저장!
            metadata = {
                "type": "article_analysis", "news_id": row['id'], "source": row['source']
            }
            documents.append(Document(page_content=page_content.strip(), metadata=metadata))


        # --- 2. 일일 시장 요약 데이터 가공 ---
        print("  - 일일 시장 요약 데이터를 가공 중...")
        query_summary = """
        SELECT
            dms.date, dms.daily_sentiment_score, dms.daily_reasoning,
            dms.daily_keywords, c.name as commodity_name, dms.analyzed_news_count
        FROM daily_market_summary dms
        JOIN commodities c ON dms.commodity_id = c.id;
        """
        df_summary = pd.read_sql(query_summary, conn)

        for _, row in df_summary.iterrows():
            page_content = f"""
            날짜: {row['date']}
            품목: {row['commodity_name']}
            일일 시장 심리 점수: {row['daily_sentiment_score']}
            일일 시장 동향 요약: {row['daily_reasoning']}
            주요 키워드: {row['daily_keywords']}
            분석된 뉴스 수: {row['analyzed_news_count']}
            """
            metadata = {
                "type": "daily_summary", "date": str(row['date']), "commodity": row['commodity_name']
            }
            documents.append(Document(page_content=page_content.strip(), metadata=metadata))


        conn.close()

    except Exception as e:
        print(f"DB 연결 또는 쿼리 오류: {e}.")
        # DB 연결 실패 시, 최소한의 작동을 위한 빈 리스트 반환
        return []

    print(f"--- [데이터 로딩 완료] 총 {len(documents)}개의 문서(기사 분석 + 일일 요약)를 생성했습니다. ---")
    return documents

