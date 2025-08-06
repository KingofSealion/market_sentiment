#!/usr/bin/env python3
"""
옥수수 데이터 존재 여부 확인
"""
import os
import sys
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def check_corn_data():
    """DB에 옥수수 데이터가 있는지 확인"""
    
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            port=os.environ.get("DB_PORT")
        )
        cursor = conn.cursor()
        
        print("=== 옥수수 데이터 존재 여부 확인 ===\n")
        
        # 1. 일일 시장 요약 데이터 확인
        print("1. 일일 시장 요약 데이터 (daily_market_summary)")
        cursor.execute("""
            SELECT dms.date, c.name, dms.daily_sentiment_score, dms.daily_keywords
            FROM daily_market_summary dms
            JOIN commodities c ON dms.commodity_id = c.id
            WHERE c.name = 'Corn'
            ORDER BY dms.date DESC
            LIMIT 10
        """)
        corn_summary = cursor.fetchall()
        
        if corn_summary:
            print(f"OK 옥수수 일일 요약 데이터: {len(corn_summary)}건 발견")
            for row in corn_summary:
                print(f"   - {row[0]}: 감정점수 {row[2]}, 키워드: {row[3]}")
        else:
            print("NO 옥수수 일일 요약 데이터 없음")
        
        # 2. 개별 뉴스 분석 데이터 확인
        print("\n2. 개별 뉴스 분석 데이터 (news_analysis_results)")
        cursor.execute("""
            SELECT COUNT(*) as count, MAX(r.published_time) as latest_news
            FROM raw_news r
            JOIN news_analysis_results nar ON r.id = nar.raw_news_id
            JOIN commodities c ON nar.commodity_id = c.id
            WHERE c.name = 'Corn'
        """)
        corn_news = cursor.fetchone()
        
        if corn_news and corn_news[0] > 0:
            print(f"OK 옥수수 뉴스 분석 데이터: {corn_news[0]}건, 최신: {corn_news[1]}")
        else:
            print("NO 옥수수 뉴스 분석 데이터 없음")
        
        # 3. 가격 데이터 확인
        print("\n3. 가격 데이터 (price_history)")
        cursor.execute("""
            SELECT ph.date, ph.closing_price, c.name
            FROM price_history ph
            JOIN commodities c ON ph.commodity_id = c.id
            WHERE c.name = 'Corn'
            ORDER BY ph.date DESC
            LIMIT 5
        """)
        corn_prices = cursor.fetchall()
        
        if corn_prices:
            print(f"OK 옥수수 가격 데이터: 최근 5건")
            for row in corn_prices:
                print(f"   - {row[0]}: ${row[1]} USc/bu")
        else:
            print("NO 옥수수 가격 데이터 없음")
        
        # 4. 전체 품목 확인
        print("\n4. 전체 품목 목록 (commodities)")
        cursor.execute("SELECT id, name FROM commodities ORDER BY id")
        all_commodities = cursor.fetchall()
        
        print("등록된 품목:")
        for row in all_commodities:
            print(f"   - ID {row[0]}: {row[1]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"DB 연결 오류: {e}")

if __name__ == "__main__":
    check_corn_data()