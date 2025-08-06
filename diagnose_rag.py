#!/usr/bin/env python3
"""
RAG 벡터스토어 진단
"""
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def diagnose_rag_system():
    """RAG 벡터스토어 상태 진단"""
    
    print("=== RAG 벡터스토어 진단 ===\n")
    
    # 1. 벡터스토어 로드
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    
    if not os.path.exists(persist_dir):
        print("NO 벡터스토어 디렉터리가 존재하지 않습니다!")
        return
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir, 
            embedding_function=OpenAIEmbeddings()
        )
        print(f"OK 벡터스토어 로드 성공: {persist_dir}")
        
        # 2. 저장된 문서 수 확인
        collection = vectorstore._collection
        total_docs = collection.count()
        print(f"총 문서 수: {total_docs}")
        
        # 3. 메타데이터 분석
        all_data = vectorstore.get()
        metadatas = all_data["metadatas"]
        
        # 품목별 문서 수 계산
        commodity_counts = {}
        doc_types = {}
        
        for meta in metadatas:
            # 품목별 집계
            commodity = meta.get("commodity_name", "Unknown")
            commodity_counts[commodity] = commodity_counts.get(commodity, 0) + 1
            
            # 문서 타입별 집계  
            doc_type = meta.get("type", "Unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print("\n품목별 문서 수:")
        for commodity, count in sorted(commodity_counts.items()):
            print(f"   - {commodity}: {count}건")
        
        print("\n문서 타입별 분포:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"   - {doc_type}: {count}건")
        
        # 4. 옥수수 관련 검색 테스트
        print("\n옥수수 관련 검색 테스트:")
        
        test_queries = [
            "옥수수 가격 상승",
            "corn price increase",
            "브라질 가뭄",
            "Brazil drought",
            "옥수수 수출",
            "corn export",
        ]
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        for query in test_queries:
            print(f"\n검색어: '{query}'")
            try:
                docs = retriever.get_relevant_documents(query)
                print(f"  결과: {len(docs)}건 검색됨")
                
                for i, doc in enumerate(docs[:2], 1):  # 상위 2개만 표시
                    meta = doc.metadata
                    content_preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"    {i}. [{meta.get('type', 'Unknown')}] {content_preview}...")
                    
            except Exception as e:
                print(f"  오류: {e}")
        
        # 5. LLM + RAG 체인 테스트
        print("\nLLM + RAG 체인 테스트:")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        rag_test_queries = [
            "브라질 가뭄이 옥수수 시장에 미치는 영향은?",
            "최근 옥수수 가격 상승 요인은 무엇인가요?",
        ]
        
        for query in rag_test_queries:
            print(f"\n질문: {query}")
            try:
                result = qa_chain.invoke(query)
                answer = result["result"][:200].replace('\n', ' ')
                print(f"답변: {answer}...")
                print(f"참조 문서: {len(result['source_documents'])}건")
            except Exception as e:
                print(f"오류: {e}")
                
    except Exception as e:
        print(f"NO 벡터스토어 로드 실패: {e}")

if __name__ == "__main__":
    diagnose_rag_system()