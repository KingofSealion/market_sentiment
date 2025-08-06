#!/usr/bin/env python3
"""
LLM 라우터 기반 계산기 시스템 테스트
"""
import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트 디렉터리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.tools import CommodityCalculatorRouter
from langchain_openai import ChatOpenAI

def test_router_system():
    """새로운 LLM 라우터 시스템 테스트"""
    
    print("=== LLM 라우터 기반 계산기 시스템 테스트 ===\n")
    
    # LLM 인스턴스 생성
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # 계산기 라우터 인스턴스 생성
    calculator = CommodityCalculatorRouter(llm=llm)
    
    # 테스트 케이스들
    test_cases = [
        # 크러시 마진 테스트
        "대두 크러시 마진 계산해줘",
        "soybean crush margin이 얼마야?",
        "착유 마진을 알고 싶어",
        
        # 단위 변환 테스트
        "옥수수 150 부셸을 톤으로 변환해줘",
        "밀 200톤을 부셸로 바꿔줘",
        "corn 500 bushels는 몇 ton이야?",
        
        # 가격 변환 테스트
        "최근 옥수수 가격을 톤당 달러로 알려줘",
        "wheat price USD/MT로 변환해줘",
        
        # 면적 변환 테스트
        "95.2 million acres를 헥타르로 변환",
        "1000 헥타르를 에이커로 바꿔줘",
        
        # 베이시스 계산 테스트
        "옥수수 베이시스 +20일때 플랫가격은?",
        "wheat basis -15 flat price 계산해줘",
        
        # 다국어/다양한 표현 테스트
        "How much is 100 bushels of corn in metric tons?",
        "Convert 50 hectares to acres please",
        "What's the soybean crushing margin today?",
        
        # 폴백 테스트
        "hello world",
        "농업 관련 일반 질문",
    ]
    
    print("각 테스트 케이스별 라우터 동작 확인:\n")
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"테스트 {i}: {test_query}")
        print("-" * 50)
        
        try:
            # 라우터의 기능 식별 확인
            from app.tools import parse_query_details
            parsed = parse_query_details(test_query)
            function_name = calculator._route_to_function(test_query, parsed)
            
            print(f"파싱 결과: {parsed}")
            print(f"라우터 선택: {function_name}")
            
            # 실제 계산 실행 (DB 연결이 필요한 경우 스킵)
            try:
                result = calculator.calculate(test_query)
                print(f"계산 결과: {result}")
            except Exception as e:
                print(f"계산 실행 오류 (DB 연결 등): {e}")
            
        except Exception as e:
            print(f"테스트 오류: {e}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_router_system()