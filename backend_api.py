#웹 API 서버로 실행되는 챗봇 (FastAPI 기반)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_loader import get_documents_from_postgres
from app.tools import create_agent_tools
from app.agent_logic import create_analyst_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv  
import re
load_dotenv()

app = FastAPI(title="Agri Commodities Sentiment Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        dbname=os.environ.get("DB_NAME", "market_sentiment"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "password"),
        port=os.environ.get("DB_PORT", "5432")
    )

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class SentimentCard(BaseModel):
    commodity_name: str
    sentiment_score: float
    reasoning: str
    keywords: List[str]
    last_updated: datetime

class TimeSeriesData(BaseModel):
    date: datetime
    sentiment_score: Optional[float]
    price: Optional[float]

class NewsArticle(BaseModel):
    id: int
    title: str
    content: str
    sentiment_score: float
    reasoning: str
    keywords: List[str]
    published_time: datetime
    source: str

class TrendingKeyword(BaseModel):
    keyword: str
    frequency: int

# Initialize agent once
agent_executor = None


@app.on_event("startup")
async def startup_event():
    global agent_executor
    try:
        print("[INFO] Starting AI Agent initialization...")
        
        # Initialize OpenAI model
        if not os.environ.get("OPENAI_API_KEY"):
            print("[ERROR] OPENAI_API_KEY not found. Chat functionality will be disabled.")
            return
        
        print("[INFO] OPENAI_API_KEY found")
        
        # Initialize LLM
        print("[INFO] Initializing ChatOpenAI model...")
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
        print("[INFO] ChatOpenAI model initialized")
        
        # Load documents from PostgreSQL
        print("[INFO] Loading documents from PostgreSQL...")
        documents = get_documents_from_postgres()
        print(f"[INFO] Loaded {len(documents)} documents from database")
        
        # Create agent tools
        print("[INFO] Creating agent tools...")
        tools = create_agent_tools(documents, llm)
        if not tools:
            print("[ERROR] Failed to create agent tools - tools list is empty")
            return
        print(f"[INFO] Created {len(tools)} agent tools")
        
        # Create analyst agent
        print("[INFO] Creating analyst agent...")
        agent_executor = create_analyst_agent(tools, llm)
        if agent_executor:
            print("[INFO] AI Agent initialized successfully!")
            print(f"[INFO] Agent tools available: {[tool.name for tool in tools]}")
        else:
            print("[ERROR] Failed to create agent executor")
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize AI agent: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        agent_executor = None

@app.get("/api/dashboard/sentiment-cards", response_model=List[SentimentCard])
async def get_sentiment_cards():
    """Get current sentiment scores for all commodities"""
    try:
        conn = get_db_connection()
        query = """
        SELECT DISTINCT ON (c.name) 
            c.name as commodity_name,
            dms.daily_sentiment_score as sentiment_score,
            dms.daily_reasoning as reasoning,
            dms.daily_keywords,
            dms.date as last_updated
        FROM commodities c
        LEFT JOIN daily_market_summary dms ON c.id = dms.commodity_id
        ORDER BY c.name, dms.date DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        cards = []
        for _, row in df.iterrows():
            keywords = []
            if row['daily_keywords']:
                try:
                    keywords = json.loads(row['daily_keywords']) if isinstance(row['daily_keywords'], str) else row['daily_keywords']
                except:
                    keywords = str(row['daily_keywords']).split(',') if row['daily_keywords'] else []
            
            cards.append(SentimentCard(
                commodity_name=row['commodity_name'],
                sentiment_score=float(row['sentiment_score']) if row['sentiment_score'] is not None else 50.0,
                reasoning=row['reasoning'] if row['reasoning'] else "No analysis available",
                keywords=keywords,
                last_updated=row['last_updated'] if row['last_updated'] else datetime.now()
            ))
        
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sentiment cards: {str(e)}")

@app.get("/api/dashboard/time-series/{commodity_name}", response_model=List[TimeSeriesData])
async def get_time_series_data(commodity_name: str):
    """Get time series data for sentiment scores and prices for a specific commodity"""
    try:
        conn = get_db_connection()
        
        # First, get sentiment data with date range
        sentiment_query = """
        SELECT dms.date, dms.daily_sentiment_score
        FROM daily_market_summary dms
        JOIN commodities c ON dms.commodity_id = c.id
        WHERE c.name = %s
        ORDER BY dms.date
        """
        
        df_sentiment = pd.read_sql(sentiment_query, conn, params=[commodity_name])
        
        if df_sentiment.empty:
            conn.close()
            return []
        
        # Get date range from sentiment data
        min_date = df_sentiment['date'].min()
        max_date = df_sentiment['date'].max()
        
        # Get actual price data from price_history table ONLY
        price_query = """
        SELECT ph.date, ph.closing_price as price
        FROM price_history ph
        JOIN commodities c ON ph.commodity_id = c.id
        WHERE c.name = %s 
        AND ph.date >= %s 
        AND ph.date <= %s
        ORDER BY ph.date
        """
        
        try:
            df_price = pd.read_sql(price_query, conn, params=[commodity_name, min_date, max_date])
            print(f"[INFO] Found {len(df_price)} price records for {commodity_name}")
        except Exception as price_error:
            print(f"[ERROR] Could not fetch from price_history table: {price_error}")
            conn.close()
            raise HTTPException(
                status_code=404, 
                detail=f"Price data not available for {commodity_name}. price_history table may not exist or contain data for this commodity."
            )
        
        if df_price.empty:
            print(f"[WARNING] No price data found for {commodity_name}, but continuing with sentiment data only")
            # Create empty dataframe with same structure for consistent merging
            df_price = pd.DataFrame(columns=['date', 'price'])
        
        # Merge sentiment and price data - INCLUDE ALL sentiment data, price can be null
        time_series = []
        
        for _, row in df_sentiment.iterrows():
            date = row['date']
            sentiment_score = float(row['daily_sentiment_score']) if row['daily_sentiment_score'] else None
            
            # Get actual price for this date - allow null prices
            price_row = df_price[df_price['date'] == date]
            price = float(price_row.iloc[0]['price']) if not price_row.empty else None
            
            # Always include sentiment data, even if price is missing
            time_series.append(TimeSeriesData(
                date=date,
                sentiment_score=sentiment_score,
                price=price
            ))
        
        conn.close()
        return time_series
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch time series data: {str(e)}")

@app.get("/api/dashboard/news/{commodity_name}", response_model=List[NewsArticle])
async def get_news_articles(commodity_name: str):
    """Get all news articles for a specific commodity"""
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            r.id, r.title, r.content, r.published_time, r.source,
            nar.sentiment_score, nar.reasoning, nar.keywords
        FROM raw_news r
        JOIN news_analysis_results nar ON r.id = nar.raw_news_id
        JOIN commodities c ON nar.commodity_id = c.id
        WHERE c.name = %s AND r.analysis_status = TRUE
        ORDER BY r.published_time DESC
        """
        
        df = pd.read_sql(query, conn, params=[commodity_name])
        conn.close()
        
        articles = []
        for _, row in df.iterrows():
            keywords = []
            if row['keywords']:
                try:
                    keywords = json.loads(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
                except:
                    keywords = str(row['keywords']).split(',') if row['keywords'] else []
            
            articles.append(NewsArticle(
                id=int(row['id']),
                title=row['title'],
                content=row['content'] if row['content'] else "",
                sentiment_score=float(row['sentiment_score']),
                reasoning=row['reasoning'],
                keywords=keywords,
                published_time=row['published_time'],
                source=row['source']
            ))
        
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news articles: {str(e)}")

@app.get("/api/debug/database-info")
async def get_database_info():
    """Debug endpoint to check database tables and schema"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check all tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check if price_history exists and get its schema
        price_history_schema = []
        if 'price_history' in tables:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'price_history' 
                ORDER BY ordinal_position;
            """)
            price_history_schema = [
                {"column": row[0], "type": row[1], "nullable": row[2]} 
                for row in cursor.fetchall()
            ]
        
        conn.close()
        
        return {
            "all_tables": tables,
            "price_history_exists": 'price_history' in tables,
            "price_history_schema": price_history_schema
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database info: {str(e)}")

@app.get("/api/dashboard/trending-keywords", response_model=List[TrendingKeyword])
async def get_trending_keywords():
    """Get trending keywords from recent articles"""
    try:
        conn = get_db_connection()
        
        # First, get the most recent news date in the database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(r.published_time) 
            FROM raw_news r 
            WHERE r.analysis_status = TRUE
        """)
        max_date_result = cursor.fetchone()
        cursor.close()
        
        if not max_date_result[0]:
            conn.close()
            return []  # No analyzed news found
        
        most_recent_date = max_date_result[0]
        # Get keywords from 7 days before the most recent news date
        seven_days_before_recent = most_recent_date - timedelta(days=7)
        
        # Get keywords from recent articles (based on most recent news date)
        query = """
        SELECT nar.keywords
        FROM news_analysis_results nar
        JOIN raw_news r ON nar.raw_news_id = r.id
        WHERE r.published_time >= %s AND r.analysis_status = TRUE
        """
        
        df = pd.read_sql(query, conn, params=[seven_days_before_recent])
        conn.close()
        
        # Process keywords and count frequencies
        keyword_counts = {}
        for _, row in df.iterrows():
            if row['keywords']:
                try:
                    keywords = json.loads(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            keyword = str(keyword).strip().lower()
                            if keyword and len(keyword) > 2:  # Filter out very short keywords
                                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                except:
                    continue
        
        # Sort by frequency and return top 10
        trending = [
            TrendingKeyword(keyword=keyword, frequency=count)
            for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return trending
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending keywords: {str(e)}")

@app.get("/api/dashboard/trending-keywords/{commodity}", response_model=List[TrendingKeyword])
async def get_trending_keywords_by_commodity(commodity: str):
    """Get trending keywords for a specific commodity from recent articles"""
    try:
        conn = get_db_connection()
        
        # First, get the most recent news date in the database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(r.published_time) 
            FROM raw_news r 
            WHERE r.analysis_status = TRUE
        """)
        max_date_result = cursor.fetchone()
        cursor.close()
        
        if not max_date_result[0]:
            conn.close()
            return []  # No analyzed news found
        
        most_recent_date = max_date_result[0]
        # Get keywords from 7 days before the most recent news date
        seven_days_before_recent = most_recent_date - timedelta(days=7)
        
        # Get keywords from recent articles for specific commodity
        query = """
        SELECT nar.keywords
        FROM news_analysis_results nar
        JOIN raw_news r ON nar.raw_news_id = r.id
        JOIN commodities c ON nar.commodity_id = c.id
        WHERE r.published_time >= %s 
        AND r.analysis_status = TRUE
        AND c.name = %s
        """
        
        df = pd.read_sql(query, conn, params=[seven_days_before_recent, commodity])
        conn.close()
        
        # Process keywords and count frequencies
        keyword_counts = {}
        for _, row in df.iterrows():
            if row['keywords']:
                try:
                    keywords = json.loads(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            keyword = str(keyword).strip().lower()
                            if keyword and len(keyword) > 2:  # Filter out very short keywords
                                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                except:
                    continue
        
        # Sort by frequency and return top 10
        trending = [
            TrendingKeyword(keyword=keyword, frequency=count)
            for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return trending
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending keywords for {commodity}: {str(e)}")

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Stream chat responses using Server-Sent Events"""
    print(f"[INFO] Chat request received: {request.message[:100]}...")

    if not agent_executor:
        print("[ERROR] Agent executor not initialized")
        raise HTTPException(status_code=503, detail="AI agent not initialized.")

    async def generate_response():
        try:
            # Start 이벤트 전송
            yield f"data: {json.dumps({'type': 'start', 'message': ''})}\n\n"

            # 에이전트 호출
            result = await agent_executor.ainvoke({"input": request.message})
            response_text = result.get('output', '죄송합니다. 요청을 처리할 수 없습니다.')

            # 스트리밍: 줄 단위로 나누면서 \n 보존
            current_text = ""
            # splitlines(True) -> 줄 끝의 '\n' 도 함께 보존합니다
            for line in response_text.splitlines(True):
                current_text += line
                yield f"data: {json.dumps({'type': 'chunk', 'message': current_text})}\n\n"
                await asyncio.sleep(0.05)

            # End 이벤트 전송
            yield f"data: {json.dumps({'type': 'end', 'message': current_text})}\n\n"

        except Exception as e:
            print(f"[ERROR] Chat processing error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'죄송합니다. 오류가 발생했습니다: {e}'})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


#직접 실행 시 작동되는지 테스트용
if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", 8001))
    print(f"[INFO] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)