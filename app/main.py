# 터미널에서 직접 실행하는 콘솔 챗봇 (CLI 인터페이스)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_loader import get_documents_from_postgres
from app.tools import create_agent_tools
from app.agent_logic import create_analyst_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
import httpx

def run_bot():
    """농산물 시장 전문 AI 애널리스트 봇 실행 (main 함수)"""

    # 1. OpenAI API Key 체크
    if not os.environ.get("OPENAI_API_KEY"):
        print("[에러] OPENAI_API_KEY 환경변수를 세팅하세요!")
        return

    # 2. 모델 선택 
    custom_client = httpx.Client(verify=False) #회사에서 API 사용위해 넣어야하는 코드줄
    model_name = "gpt-4.1"
    llm = ChatOpenAI(model=model_name, temperature=0.3, http_client=custom_client)
    print(f"[INFO] OpenAI 모델: {model_name}")

    # 3. 데이터 로딩 (DB → 문서 객체)
    print("[INFO] DB에서 최신 데이터 로딩 중...")
    documents = get_documents_from_postgres()
    if not documents:
        # DB에 데이터가 없어도 봇이 실행은 되도록 경고 메시지로 변경합니다.
        print("[경고] DB에서 문서를 불러오지 못했습니다. 뉴스 검색 기능이 제한될 수 있습니다.")

    # 4. 도구 생성
    tools = create_agent_tools(documents, llm)
    if not tools:
        print("[에러] 도구 생성 실패. 실행 중단.")
        return

    # 5. 에이전트 생성 (메모리 자동 관리)
    agent_executor = create_analyst_agent(tools, llm)

    # AgentExecutor에 내장된 메모리를 활용하므로, main.py에서 chat_history를
    # 수동으로 관리할 필요가 없습니다. AgentExecutor가 알아서 대화 기록을 memory 객체에 추가하고 관리합니다.
    
    print("\n" + "="*60)
    print("🤖 원자재 애널리스트 봇이 준비되었습니다.")
    print("   📊 정확한 날짜+품목 데이터: SQL 검색")
    print("   📰 시장 동향+뉴스 맥락: RAG 검색")
    print("   💡 예시: '2025년 7월 10일 옥수수 감정점수는?'")
    print("   (종료하시려면 'exit'를 입력하세요)")
    print("="*60)

    while True:
        user_input = input("👤 사용자: ")
        if user_input.lower() == 'exit':
            print("🤖 봇을 종료합니다.")
            break
        
        try:
            # 에이전트 실행 시 'input'만 전달하면, AgentExecutor에 포함된
            # memory 객체가 자동으로 'chat_history'를 관리하고 주입해줍니다.
            result = agent_executor.invoke({
                "input": user_input,
            })
            
            print("\n" + "-"*25 + " 최종 답변 " + "-"*25)
            print("🤖 봇:", result['output'])
            print("-" * 60)
        except Exception as e:
            print(f"[오류 발생] 에이전트 실행 중 오류가 발생했습니다: {e}")
            print("-" * 60)

if __name__ == '__main__':
    run_bot()