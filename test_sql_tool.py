#!/usr/bin/env python3
"""
SQL 도구로 옥수수 데이터 검색 테스트
"""
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from app.tools import sql_query_tool

def test_sql_corn_queries():
    """SQL 도구로 옥수수 관련 쿼리 테스트"""
    
    print("=== SQL 도구 옥수수 데이터 검색 테스트 ===\n")
    
    test_queries = [
        "옥수수 지난 한주간 시황 주요 키워드",
        "최근 옥수수 감정점수 시장 동향 주요 키워드",
        "옥수수 최근 감정점수",
        "corn recent sentiment score market trend",
        "최근 옥수수",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"테스트 {i}: {query}")
        print("-" * 50)
        
        try:
            result = sql_query_tool(query)
            print(f"결과:\n{result}")
        except Exception as e:
            print(f"오류: {e}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_sql_corn_queries()