"""
🌾 원자재 시장 심리화 대시보드 - Streamlit 버전
Real-Time Commodities Market News Sentiment Analysis & AI Insights
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import time
import json
from typing import List, Dict, Any, Optional
import asyncio

# 페이지 설정
st.set_page_config(
    page_title="🌾 원자재 시장 심리화 대시보드",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = 'http://localhost:8001'

# 세션 상태 초기화
if 'selected_commodity' not in st.session_state:
    st.session_state.selected_commodity = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# 커스텀 CSS
st.markdown("""
<style>
/* 메인 배경 그라디언트 */
.main > div {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 2rem;
}

/* 메트릭 카드 스타일링 */
.metric-card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
    text-align: center;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

/* 제목 스타일 */
.dashboard-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.dashboard-subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

/* 감정점수 컬러 */
.sentiment-positive { color: #4caf50; font-weight: bold; }
.sentiment-negative { color: #f44336; font-weight: bold; }
.sentiment-neutral { color: #ff9800; font-weight: bold; }

/* 카드 선택 효과 */
.selected-card {
    border: 3px solid #667eea !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* 채팅 메시지 스타일 */
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: right;
}

.assistant-message {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# 유틸리티 함수들
def get_sentiment_level(score: float) -> str:
    """감정 점수를 레벨로 변환"""
    if score >= 60:
        return 'Positive'
    elif score <= 40:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_color(score: float) -> str:
    """감정 점수에 따른 컬러 반환"""
    if score >= 60:
        return '#4caf50'  # Green
    elif score <= 40:
        return '#f44336'  # Red
    else:
        return '#ff9800'  # Orange

def get_commodity_icon(commodity_name: str) -> str:
    """원자재별 아이콘 반환"""
    name = commodity_name.lower()
    if 'corn' in name:
        return '🌽'
    elif 'wheat' in name:
        return '🌾'
    elif 'soybean meal' in name:
        return '🌱'
    elif 'soybean oil' in name:
        return '🛢️'
    elif 'soybean' in name:
        return '🌱'
    elif 'palm' in name:
        return '🌴'
    else:
        return '🌾'

def get_sentiment_emoji(score: float) -> str:
    """감정 점수에 따른 이모지 반환"""
    if score >= 60:
        return '🙂'
    elif score <= 40:
        return '🙁'
    else:
        return '😐'

def format_date(date_string: str) -> str:
    """날짜 포맷팅"""
    try:
        if isinstance(date_string, str):
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        else:
            dt = date_string
        return dt.strftime('%Y-%m-%d')
    except:
        return str(date_string)

# API 클라이언트 함수들
@st.cache_data(ttl=60)  # 1분 캐시
def fetch_sentiment_cards():
    """감정점수 카드 데이터 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/dashboard/sentiment-cards")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"감정점수 데이터를 불러올 수 없습니다: {str(e)}")
        return []

@st.cache_data(ttl=60)
def fetch_time_series_data(commodity: str):
    """시계열 데이터 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/dashboard/time-series/{commodity}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"시계열 데이터를 불러올 수 없습니다: {str(e)}")
        return []

@st.cache_data(ttl=60)
def fetch_news_articles(commodity: str):
    """뉴스 기사 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/dashboard/news/{commodity}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"뉴스 데이터를 불러올 수 없습니다: {str(e)}")
        return []

@st.cache_data(ttl=60)
def fetch_trending_keywords(commodity: str):
    """트렌딩 키워드 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/dashboard/trending-keywords/{commodity}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"트렌딩 키워드 데이터를 불러올 수 없습니다: {str(e)}")
        return []

def send_chat_message(message: str):
    """챗봇에 메시지 전송"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"message": message},
            stream=True
        )
        response.raise_for_status()
        return response
    except Exception as e:
        st.error(f"채팅 오류: {str(e)}")
        return None

# 메인 대시보드 함수들
def render_header():
    """헤더 렌더링"""
    st.markdown('<h1 class="dashboard-title">🌾 원자재 시장 심리화 대시보드</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Real-Time Commodities Market News Sentiment Analysis & AI Insights</p>', unsafe_allow_html=True)
    st.markdown("---")

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/667eea/FFFFFF?text=LOGO", width=150)
        
        st.markdown("### ⚙️ 설정")
        
        # 자동 새로고침 설정
        st.session_state.auto_refresh = st.toggle("🔄 자동 새로고침", value=st.session_state.auto_refresh)
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("새로고침 간격 (초)", 10, 300, 60)
            st.info(f"⏱️ {refresh_interval}초마다 자동으로 새로고침됩니다.")
        
        # 다크모드 토글 (향후 구현)
        dark_mode = st.toggle("🌙 다크모드", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 대시보드 정보")
        st.info("실시간 원자재 시장 감정 분석을 제공합니다.")
        
        # 새로고침 버튼
        if st.button("🔄 수동 새로고침"):
            st.cache_data.clear()
            st.rerun()

def render_sentiment_cards():
    """감정점수 카드들 렌더링"""
    st.subheader("📊 Market Overview")
    
    # 데이터 가져오기
    sentiment_cards = fetch_sentiment_cards()
    
    if not sentiment_cards:
        st.error("감정점수 데이터를 불러올 수 없습니다.")
        return
    
    # 원자재 순서 정의
    commodity_order = ['Corn', 'Wheat', 'Soybean', 'Soybean Meal', 'Soybean Oil', 'Palm Oil']
    
    # 카드 정렬
    sorted_cards = []
    for commodity in commodity_order:
        card = next((c for c in sentiment_cards if c['commodity_name'] == commodity), None)
        if card:
            sorted_cards.append(card)
    
    # 6개 컬럼으로 카드 표시
    cols = st.columns(6)
    
    for i, card in enumerate(sorted_cards):
        with cols[i]:
            # 메트릭 표시
            sentiment_score = round(card['sentiment_score'])
            sentiment_level = get_sentiment_level(sentiment_score)
            sentiment_color = get_sentiment_color(sentiment_score)
            commodity_icon = get_commodity_icon(card['commodity_name'])
            
            # 카드 선택 상태 확인
            is_selected = st.session_state.selected_commodity == card['commodity_name']
            
            # 카드 스타일 적용
            card_style = "selected-card" if is_selected else "metric-card"
            
            st.markdown(f"""
            <div class="{card_style}">
                <h3>{commodity_icon} {card['commodity_name']}</h3>
                <h2 style="color: {sentiment_color};">{sentiment_score}</h2>
                <p style="color: {sentiment_color}; font-weight: bold;">{sentiment_level} {get_sentiment_emoji(sentiment_score)}</p>
                <p style="font-size: 0.8em; margin-top: 1rem;">
                    📅 {format_date(card['last_updated'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 클릭 버튼 (각 카드마다 고유 키)
            if st.button(f"📊 Select", key=f"card_{i}", use_container_width=True):
                st.session_state.selected_commodity = card['commodity_name']
                st.rerun()

def render_sentiment_analysis():
    """감정분석 상세 정보 렌더링"""
    if not st.session_state.selected_commodity:
        return
    
    st.subheader(f"📊 Sentiment Analysis - {st.session_state.selected_commodity}")
    
    # 선택된 카드 데이터 가져오기
    sentiment_cards = fetch_sentiment_cards()
    selected_card = next((c for c in sentiment_cards if c['commodity_name'] == st.session_state.selected_commodity), None)
    
    if not selected_card:
        st.error("선택된 원자재 데이터를 찾을 수 없습니다.")
        return
    
    # 감정 점수 표시
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        sentiment_score = round(selected_card['sentiment_score'])
        sentiment_level = get_sentiment_level(sentiment_score)
        sentiment_emoji = get_sentiment_emoji(sentiment_score)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; 
                    background: linear-gradient(135deg, {get_sentiment_color(sentiment_score)}22 0%, {get_sentiment_color(sentiment_score)}44 100%);
                    border-radius: 15px; border: 2px solid {get_sentiment_color(sentiment_score)};">
            <h1 style="font-size: 4rem; margin: 0;">{sentiment_emoji}</h1>
            <h2 style="color: {get_sentiment_color(sentiment_score)}; margin: 0.5rem 0;">{sentiment_score}</h2>
            <h3 style="color: {get_sentiment_color(sentiment_score)}; margin: 0;">{sentiment_level}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 분석 이유
    st.markdown("### 💭 Analysis Reasoning")
    st.info(selected_card['reasoning'])
    
    # 키워드
    st.markdown("### 🏷️ Key Factors")
    
    # 키워드를 컬럼으로 표시
    if selected_card['keywords']:
        keyword_cols = st.columns(min(len(selected_card['keywords']), 4))
        for i, keyword in enumerate(selected_card['keywords'][:8]):
            col_idx = i % len(keyword_cols)
            with keyword_cols[col_idx]:
                st.markdown(f"""
                <span style="background: {get_sentiment_color(sentiment_score)}; 
                            color: white; padding: 0.2rem 0.5rem; 
                            border-radius: 15px; font-size: 0.8rem; 
                            display: inline-block; margin: 0.2rem;">
                    {keyword}
                </span>
                """, unsafe_allow_html=True)

def render_sentiment_chart():
    """감정분석 차트 렌더링"""
    if not st.session_state.selected_commodity:
        return
    
    st.subheader(f"📈 Sentiment Trends - {st.session_state.selected_commodity}")
    
    # 시계열 데이터 가져오기
    time_series_data = fetch_time_series_data(st.session_state.selected_commodity)
    
    if not time_series_data:
        st.warning("시계열 데이터가 없습니다.")
        return
    
    # DataFrame으로 변환
    df = pd.DataFrame(time_series_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Plotly 차트 생성
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sentiment Score', 'Price'),
        vertical_spacing=0.15,
        shared_xaxes=True
    )
    
    # 감정 점수 라인
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment_score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Sentiment:</b> %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 가격 라인 (null이 아닌 경우에만)
    if df['price'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['price'],
                mode='lines+markers',
                name='Price',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 레이아웃 설정
    fig.update_layout(
        height=500,
        showlegend=True,
        template="plotly_white",
        font=dict(family="Arial", size=12),
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_trending_keywords():
    """트렌딩 키워드 렌더링"""
    if not st.session_state.selected_commodity:
        return
    
    st.subheader(f"🔥 트렌딩 키워드 - {st.session_state.selected_commodity}")
    
    # 트렌딩 키워드 데이터 가져오기
    keywords_data = fetch_trending_keywords(st.session_state.selected_commodity)
    
    if not keywords_data:
        st.info("트렌딩 키워드 데이터가 없습니다.")
        return
    
    # 키워드를 컬럼으로 표시
    cols = st.columns(min(len(keywords_data), 6))
    
    for i, keyword_data in enumerate(keywords_data[:12]):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 10px; 
                        text-align: center; margin: 0.5rem 0;">
                <strong>{keyword_data['keyword']}</strong><br>
                <small>({keyword_data['frequency']})</small>
            </div>
            """, unsafe_allow_html=True)

def render_news_articles():
    """뉴스 기사 렌더링"""
    if not st.session_state.selected_commodity:
        return
    
    st.subheader(f"📰 최근 뉴스 & 분석 - {st.session_state.selected_commodity}")
    
    # 뉴스 데이터 가져오기
    news_data = fetch_news_articles(st.session_state.selected_commodity)
    
    if not news_data:
        st.info("뉴스 데이터가 없습니다.")
        return
    
    # 뉴스 기사들을 expander로 표시
    for article in news_data[:10]:  # 최대 10개 기사 표시
        with st.expander(f"📰 {article['title'][:80]}..." if len(article['title']) > 80 else f"📰 {article['title']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📅 발행일:** {format_date(article['published_time'])}")
                st.markdown(f"**📰 출처:** {article['source']}")
                st.markdown("---")
                
                st.markdown("**💭 AI 분석:**")
                st.info(article['reasoning'])
                
                st.markdown("**📰 기사 내용:**")
                st.markdown(article['content'][:500] + "..." if len(article['content']) > 500 else article['content'])
            
            with col2:
                # 감정 점수
                sentiment_score = round(article['sentiment_score'])
                sentiment_color = get_sentiment_color(sentiment_score)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; 
                            background: {sentiment_color}22; 
                            border-radius: 10px; 
                            border: 2px solid {sentiment_color};">
                    <h3 style="color: {sentiment_color}; margin: 0;">감정점수</h3>
                    <h2 style="color: {sentiment_color}; margin: 0.5rem 0;">{sentiment_score}</h2>
                    <p style="color: {sentiment_color}; margin: 0;">{get_sentiment_level(sentiment_score)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**🏷️ 키워드:**")
                for keyword in article['keywords'][:5]:
                    st.markdown(f"""
                    <span style="background: #e0e0e0; padding: 0.2rem 0.4rem; 
                                border-radius: 10px; font-size: 0.7rem; 
                                margin: 0.1rem; display: inline-block;">
                        {keyword}
                    </span>
                    """, unsafe_allow_html=True)

def render_chatbot():
    """AI 챗봇 렌더링"""
    st.subheader("🤖 AI 원자재 시장 분석가")
    
    # 새 대화 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 새 대화"):
            st.session_state.chat_messages = []
            st.rerun()
    
    # 채팅 메시지 표시
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_messages:
            st.info("원자재 시장에 대해 무엇이든 물어보세요! 📈")
        else:
            for i, message in enumerate(st.session_state.chat_messages):
                if message['isUser']:
                    with st.chat_message("user"):
                        st.write(message['message'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['message'])
    
    # 채팅 입력
    user_input = st.chat_input("원자재 시장에 대해 질문하세요...")
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_messages.append({
            'message': user_input,
            'isUser': True,
            'timestamp': datetime.now()
        })
        
        # AI 응답 받기
        with st.spinner("AI가 응답을 생성하고 있습니다..."):
            response = send_chat_message(user_input)
            if response:
                # 스트리밍 응답 처리
                full_response = ""
                response_placeholder = st.empty()
                
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if data.get('type') in ['chunk', 'end']:
                                    full_response = data.get('message', '')
                                    with response_placeholder.container():
                                        with st.chat_message("assistant"):
                                            st.write(full_response)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    st.error(f"응답 처리 중 오류가 발생했습니다: {str(e)}")
                    full_response = "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요."
                
                # 응답 메시지 추가
                st.session_state.chat_messages.append({
                    'message': full_response,
                    'isUser': False,
                    'timestamp': datetime.now()
                })
            else:
                st.error("AI 응답을 받을 수 없습니다.")
        
        st.rerun()

def main():
    """메인 함수"""
    # 헤더 렌더링
    render_header()
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 탭 생성
    dashboard_tab, chatbot_tab = st.tabs(["📊 대시보드", "🤖 AI 챗봇"])
    
    with dashboard_tab:
        render_dashboard()
    
    with chatbot_tab:
        render_chatbot()
    
    # 자동 새로고침
    if st.session_state.auto_refresh:
        time.sleep(1)
        st.rerun()

def render_dashboard():
    """대시보드 메인 컨텐츠 렌더링"""
    # 감정점수 카드들 표시
    render_sentiment_cards()
    
    st.markdown("---")
    
    # 선택된 원자재가 있을 때만 상세 정보 표시
    if st.session_state.selected_commodity:
        # 2열 레이아웃
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_sentiment_analysis()
        
        with col2:
            render_sentiment_chart()
        
        st.markdown("---")
        
        # 트렌딩 키워드
        render_trending_keywords()
        
        st.markdown("---")
        
        # 뉴스 기사
        render_news_articles()
    else:
        st.info("📍 위의 원자재 카드 중 하나를 선택해주세요!")

if __name__ == "__main__":
    main()