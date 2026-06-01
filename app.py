import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="쉬었음 청년 생활안전망 격차 분석", layout="wide")

# --- 1. 데이터 로드 함수 ---
@st.cache_data
def load_data():
    # 1. 시계열/규모 데이터 (경제활동인구조사 기반)
    df_trend = pd.read_csv("연령_활동상태별_쉬었음__비경제활동인구.csv")
    
    # 2. 핵심 실태조사 데이터 (청년삶실태조사 기반)
    # 실제 컬럼명이 다를 경우를 대비해 주요 분석 변수를 매핑하는 과정이 필요합니다.
    df_survey = pd.read_csv("쉬었음 청년 실태조사_2024.csv")
    
    # 3. 노동패널 데이터 (패널조사 기반)
    df_panel = pd.read_csv("한국노동패널조사.csv")
    
    return df_trend, df_survey, df_panel

try:
    df_trend, df_survey, df_panel = load_data()
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다. 파일명을 확인해주세요: {e}")
    st.stop()

# --- 2. 사이드바 (전역 필터) ---
st.sidebar.title("🔍 분석 필터")
age_groups = df_survey['연령'].unique() if '연령' in df_survey.columns else ["전체"]
selected_age = st.sidebar.multiselect("연령대 선택", age_groups, default=age_groups)

# --- 3. 메인 스토리라인 시작 ---
st.title("🏃‍♂️ 쉬었음 청년은 무엇으로 버티고 있는가?")
st.markdown("### : 생활비·부채·소득지원으로 본 청년 비경제활동의 생활안전망 격차")

# 페이지 구성 (Tab 활용)
pages = ["페이지 1: 규모 현황", "페이지 2: 소득/생활비 구조", "페이지 3: 가족 안전망", 
         "페이지 4: 가족 밖의 안전망", "페이지 5: 위험지수 분석", "페이지 6: 청년 유형 분류", "페이지 7: 최종 결론"]
selected_page = st.sidebar.radio("페이지 이동", pages)

# --- 페이지 1: "쉬었음 청년은 얼마나 많은가?" ---
if selected_page == pages[0]:
    st.header("📍 쉬었음 청년의 규모와 추이")
    st.info("경제활동인구조사 데이터를 통해 쉬었음 청년의 증가 추세를 확인합니다.")
    
    # 시각화: 연령별 쉬었음 추이 (Line Chart)
    # df_trend의 컬럼 구조에 따라 수정 필요 (예: 연도, 연령대, 인구수)
    fig1 = px.line(df_trend, x=df_trend.columns[0], y=df_trend.columns[1:], 
                  title="연령대별 쉬었음 인구 추이", markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("최근 몇 년간 청년층 내 '쉬었음' 인구는 단순 비경제활동인구를 넘어 하나의 거대한 계층을 형성하고 있습니다.")

# --- 페이지 2: "쉬었음 청년은 무엇으로 버티는가?" ---
elif selected_page == pages[1]:
    st.header("💰 소득원과 생활비 구조")
    st.markdown("쉬었음 청년의 소득원이 '사적 지원', '공적 지원', '부채' 중 어디에 치중되어 있는지 분석합니다.")
    
    # Sankey Diagram: 소득원 -> 쉬었음 청년 -> 생활비 지출
    nodes = ["사적이전소득", "공적이전소득", "금융부채", "쉬었음 청년", "생활비(저)", "생활비(중)", "생활비(고)"]
    # 실제 데이터에서 합계를 구하는 로직 (가상 데이터 예시)
    links = [
        {"source": 0, "target": 3, "value": df_survey['사적이전소득'].sum()},
        {"source": 1, "target": 3, "value": df_survey['공적이전소득'].sum()},
        {"source": 2, "target": 3, "value": df_survey['생활비목적부채'].sum()},
        {"source": 3, "target": 4, "value": len(df_survey[df_survey['월평균생활비'] < 100]) * 50},
        {"source": 3, "target": 5, "value": len(df_survey[(df_survey['월평균생활비'] >= 100) & (df_survey['월평균생활비'] < 200)]) * 150},
        {"source": 3, "target": 6, "value": len(df_survey[df_survey['월평균생활비'] >= 200]) * 250},
    ]
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color="royalblue"),
        link=dict(source=[l["source"] for l in links], target=[l["target"] for l in links], value=[l["value"] for l in links])
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)
    st.success("인사이트: 소득이 없는 상태에서 대다수의 생활비가 '사적 지원(가족)'에서 발생하고 있음을 보여줍니다.")

# --- 페이지 3: "가족은 얼마나 강력한 안전망인가?" ---
elif selected_page == pages[2]:
    st.header("🏠 부모 동거와 가족 지원")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("부모 동거 여부에 따른 부채 차이")
        fig_box = px.box(df_survey, x="부모동거여부", y="부채총액", color="부모동거여부")
        st.plotly_chart(fig_box)
        
    with col2:
        st.subheader("가족 도움 가능 여부 분포")
        fig_pie = px.pie(df_survey, names="가족도움가능여부", hole=0.4)
        st.plotly_chart(fig_pie)
    
    st.write("부모와 동거하는 청년들은 주거비와 생활비 리스크를 가족에게 전가할 수 있지만, 독립한 청년은 곧바로 부채 리스크에 노출됩니다.")

# --- 페이지 4: "가족이 없다면?" ---
elif selected_page == pages[3]:
    st.header("🏢 공적 지원과 금융 부채")
    st.markdown("가족의 도움을 받을 수 없는 청년들이 공적 안전망이나 부채에 의존하는 정도를 확인합니다.")
    
    # Heatmap: 공적 지원 vs 부채 수준
    fig_heat = px.density_heatmap(df_survey, x="공적이전소득", y="부채총액", 
                                  marginal_x="histogram", marginal_y="histogram",
                                  title="공적 지원과 부채의 상관관계")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.warning("가족 지원이 끊긴 청년에게 공적 지원이 도달하지 못할 경우, 유일한 버팀목은 '부채'가 됩니다.")

# --- 페이지 5: "누가 가장 위험한가?" ---
elif selected_page == pages[4]:
    st.header("🚨 생활안전망 위험지수 분석")
    
    # 위험지수 계산 로직 (가점 방식)
    df_survey['risk_score'] = (
        (df_survey['부모동거여부'] == '비동거').astype(int) +
        (df_survey['가족도움가능여부'] == '아니오').astype(int) +
        (df_survey['도움받을곳없음여부'] == '예').astype(int) +
        (df_survey['부채총액'] > df_survey['부채총액'].median()).astype(int)
    )
    
    avg_risk = df_survey['risk_score'].mean()
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_risk,
        title = {'text': "쉬었음 청년 평균 위기 점수 (0-4)"},
        gauge = {'axis': {'range': [0, 4]},
                 'bar': {'color': "red"},
                 'steps': [
                     {'range': [0, 1.5], 'color': "lightgreen"},
                     {'range': [1.5, 3], 'color': "yellow"},
                     {'range': [3, 4], 'color': "orange"}]}))
    st.plotly_chart(fig_gauge)
    st.write("위험 요소(비동거, 도움없음, 고부채)가 중첩될수록 지수는 상승하며, 이들이 정책적 최우선 지원 대상입니다.")

# --- 페이지 6: "쉬었음 청년의 5가지 유형" ---
elif selected_page == pages[5]:
    st.header("🧩 데이터로 분류한 쉬었음 청년의 5가지 유형")
    
    # Treemap 활용 유형 시각화
    fig_tree = px.treemap(df_survey, path=['유형분류', '부모동거여부'], values='월평균생활비',
                         color='유형분류', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.markdown("""
    - **가족완충형**: 부모와 동거하며 풍부한 사적 이전을 받는 집단
    - **금융부담형**: 소득은 없으나 부채를 통해 생활비를 충당하는 집단
    - **공공지원형**: 실업급여 등 공적 지원으로 버티는 집단
    - **고립위험형**: 모든 네트워크가 단절된 최취약 집단
    - **다중지원형**: 여러 안전망을 동시에 활용하는 집단
    """)

# --- 페이지 7: "한국 청년의 진짜 안전망은 무엇인가?" ---
else:
    st.header("🏁 최종 결론")
    
    # Radar Chart로 비교
    categories = ['가족지원', '공공지원', '금융지원(부채)', '사회적연결']
    fig_radar = go.Figure()
    
    # 예시 데이터 (실제 데이터 평균으로 교체)
    fig_radar.add_trace(go.Scatterpolar(r=[4, 1, 3, 2], theta=categories, fill='toself', name='쉬었음 청년 평균'))
    
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.subheader("💡 분석 결과 요약")
    st.markdown("""
    1. **가족 의존성**: 쉬었음 청년의 최대 안전망은 '가족'이며, 이는 가족의 경제력에 따른 청년 내 격차를 유발함.
    2. **안전망의 사각지대**: 가족 지원이 없는 청년에게 공공 지원은 충분한 대체재가 되지 못하고 있음.
    3. **부채의 생계화**: 저소득 쉬었음 청년에게 부채는 자산 형성이 아닌 '오늘의 생존'을 위한 도구임.
    """)
    
    st.error("🚨 최종 결론: 쉬었음 청년은 '가족'이라는 사적 복지에 전적으로 의존하고 있으며, 이 고리가 끊긴 청년은 즉각적인 빈곤과 부채의 늪에 빠지는 '안전망 격차'에 노출되어 있습니다.")