import os
import re
import psycopg2
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import Tool
# [핵심 수정] OpenAI 대신 ChatOpenAI를 임포트합니다.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def batch(iterable, batch_size=500):
    """리스트를 batch_size씩 잘라서 반환하는 유틸 함수"""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def parse_date_and_commodity(query: str) -> Dict[str, Any]:
    """
    사용자 질의에서 날짜와 품목 정보를 추출합니다.
    """
    result = {"date": None, "commodity": None, "date_range": None}
    
    # 날짜 패턴 매칭 (다양한 형식 지원)
    date_patterns = [
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 2025년 7월 10일
        r'(\d{4})-(\d{1,2})-(\d{1,2})',            # 2025-07-10
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',          # 2025.07.10
        r'(\d{1,2})월\s*(\d{1,2})일',              # 7월 10일 (현재 년도)
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, query)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # 년월일 모두 있음
                year, month, day = groups
                result["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            elif len(groups) == 2:  # 월일만 있음 (현재 년도 사용)
                month, day = groups
                current_year = datetime.now().year
                result["date"] = f"{current_year}-{month.zfill(2)}-{day.zfill(2)}"
            break
    
    # 상대적 날짜 표현
    if "오늘" in query:
        result["date"] = datetime.now().strftime("%Y-%m-%d")
    elif "어제" in query:
        from datetime import timedelta
        yesterday = datetime.now() - timedelta(days=1)
        result["date"] = yesterday.strftime("%Y-%m-%d")
    elif "최근" in query or "최신" in query or "가장 최근" in query:
        result["date_range"] = "recent"
    
    # 품목명 추출 (한글 -> 영문 매핑)
    commodity_mapping = {
        "옥수수": "Corn",
        "대두": "Soybean", 
        "대두박": "Soybean Meal",
        "대두유": "Soybean Oil",
        "소맥": "Wheat",
        "밀": "Wheat",
        "팜오일": "Palm Oil",
        "팜유": "Palm Oil"
    }
    
    for korean_name, english_name in commodity_mapping.items():
        if korean_name in query:
            result["commodity"] = english_name
            break
    
    return result


def sql_query_tool(query: str) -> str:
    """
    SQL을 사용하여 정확한 날짜+품목 기반 데이터를 검색합니다.
    """
    try:
        # 쿼리에서 날짜와 품목 추출
        parsed = parse_date_and_commodity(query)
        
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
        
        if parsed["date"]:
            summary_query += " AND dms.date = %s"
            params.append(parsed["date"])
        elif parsed["date_range"] == "recent" or not parsed["date"]:
            # 최근 데이터 또는 날짜 미지정시 최근 7일간 데이터
            summary_query += " AND dms.date >= CURRENT_DATE - INTERVAL '7 days'"
        
        if parsed["commodity"]:
            summary_query += " AND c.name = %s"
            params.append(parsed["commodity"])
        
        summary_query += " ORDER BY dms.date DESC LIMIT 10"
        
        cursor.execute(summary_query, params)
        summary_results = cursor.fetchall()
        
        if summary_results:
            results.append("=== 일일 시장 요약 ===")
            for row in summary_results:
                date, commodity, score, reasoning, keywords, news_count = row
                results.append(f"날짜: {date}")
                results.append(f"품목: {commodity}")
                results.append(f"감정점수: {score}")
                results.append(f"시장 동향: {reasoning}")
                results.append(f"주요 키워드: {keywords}")
                results.append(f"분석된 뉴스 수: {news_count}")
                results.append("---")
        
        # 2. 개별 뉴스 분석 데이터 검색
        if parsed["commodity"]:
            news_query = """
            SELECT r.title, r.published_time, nar.sentiment_score, 
                   nar.reasoning, nar.keywords, c.name as commodity_name
            FROM raw_news r
            JOIN news_analysis_results nar ON r.id = nar.raw_news_id
            JOIN commodities c ON nar.commodity_id = c.id
            WHERE c.name = %s
            """
            params = [parsed["commodity"]]
            
            if parsed["date"]:
                news_query += " AND DATE(r.published_time) = %s"
                params.append(parsed["date"])
            elif parsed["date_range"] == "recent":
                news_query += " AND r.published_time >= CURRENT_DATE - INTERVAL '7 days'"
            
            news_query += " ORDER BY r.published_time DESC LIMIT 5"
            
            cursor.execute(news_query, params)
            news_results = cursor.fetchall()
            
            if news_results:
                results.append("\n=== 관련 뉴스 분석 ===")
                for row in news_results:
                    title, pub_time, score, reasoning, keywords, commodity = row
                    results.append(f"제목: {title}")
                    results.append(f"발행시간: {pub_time}")
                    results.append(f"품목: {commodity}")
                    results.append(f"감정점수: {score}")
                    results.append(f"분석 근거: {reasoning}")
                    results.append(f"키워드: {keywords}")
                    results.append("---")
        
        # 3. 가격 데이터 검색 (있는 경우)
        if parsed["commodity"]:
            price_query = """
            SELECT ph.date, ph.closing_price, c.name as commodity_name
            FROM price_history ph
            JOIN commodities c ON ph.commodity_id = c.id
            WHERE c.name = %s
            """
            params = [parsed["commodity"]]
            
            if parsed["date"]:
                price_query += " AND ph.date = %s"
                params.append(parsed["date"])
            elif parsed["date_range"] == "recent":
                price_query += " AND ph.date >= CURRENT_DATE - INTERVAL '7 days'"
            
            price_query += " ORDER BY ph.date DESC LIMIT 5"
            
            cursor.execute(price_query, params)
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


# [핵심 수정] 함수의 타입 힌트를 llm: OpenAI -> llm: ChatOpenAI 로 변경합니다.
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
    print(f"  - 전체 {len(splits)}개 청크 중 신규 {len(new_splits)}개만 임베딩 추가합니다.")
    batch_size = 500  # OpenAI 임베딩 토큰 제한 고려 (안전하게 500 추천)
    if new_splits:
        for batch_docs in batch(new_splits, batch_size):
            vectorstore.add_documents(batch_docs)
        vectorstore.persist() # 변경사항을 디스크에 영구 저장
        print(f"  - 신규 문서 임베딩 및 저장(persist) 완료.")

    # 6. Retriever(검색기) 생성 - 다양한 관점 반영을 위해 mmr 사용
    retriever = vectorstore.as_retriever(
        search_type="mmr",          # 유사도 높은 결과 + 다양한 관점의 결과를 함께 가져오는 방식
        search_kwargs={"k": 6}      # 최종적으로 LLM에 전달할 문서 개수
    )

    # 7. RetrievalQA 체인 생성(뉴스 답변기)
    news_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # 검색된 문서들을 모두 컨텍스트에 넣어 LLM에 전달하는 가장 표준적인 방식
        retriever=retriever,
        return_source_documents=True  # 답변의 근거가 된 원본 문서를 함께 반환
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

    # 9. 최종 Tool 객체 생성
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
            "예시: '2025년 7월 10일 옥수수 감정점수', '오늘 대두 시장 상황', '최근 밀 가격' 등 "
            "날짜나 품목이 명시된 구체적인 질문에 사용하세요."
        )
    )

    # 11. 생성된 도구들을 리스트에 담아 반환 (향후 다른 도구 추가 용이)
    print("--- [도구 준비 완료] (RAG + SQL 하이브리드) ---")
    return [news_tool, sql_tool]


