import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os

st.set_page_config(
    page_title="쉬었음 청년 생활안전망 격차 분석",
    layout="wide"
)

# =========================
# 1. DB 연결
# =========================

DB_PATH = "쉬었음 청년 분석.db"

@st.cache_data
def load_table(table_name):
    if not os.path.exists(DB_PATH):
        st.error(f"DB 파일을 찾을 수 없습니다: {DB_PATH}")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM '{table_name}'", conn)
    conn.close()
    return df

@st.cache_data
def get_table_names():
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )
    conn.close()
    return tables["name"].tolist()

# =========================
# 2. 데이터 로드
# =========================

try:
    df_survey = load_table("EDA용 2024 청년삶실태조사")
    df_trend = load_table("연령_활동상태별_쉬었음__비경제활동인구")
    df_panel = load_table("한국노동패널조사")
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.write("DB 안 테이블 목록을 확인하세요.")
    if os.path.exists(DB_PATH):
        st.write(get_table_names())
    st.stop()

# =========================
# 3. 컬럼명 설정
# =========================

COL_PARENT = "부모 동거 여부"
COL_FAMILY = "가족"
COL_FRIEND = "지인"
COL_PUBLIC = "공공기관"
COL_PRIVATE = "민간기관"
COL_NONE = "없음"
COL_COST = "월 평균 총생활비"
COL_PRIVATE_INCOME = "사적 이전소득"
COL_PUBLIC_INCOME = "공적 이전소득"
COL_TOTAL_INCOME = "청년 연간소득 - 총 소득"
COL_DEBT = "청년 기준 부채 총액"
COL_LIVING_DEBT = "생활비 부채"
COL_INTEREST = "월평균 이자"

needed_cols = [
    COL_PARENT, COL_FAMILY, COL_FRIEND, COL_PUBLIC, COL_PRIVATE, COL_NONE,
    COL_COST, COL_PRIVATE_INCOME, COL_PUBLIC_INCOME, COL_TOTAL_INCOME,
    COL_DEBT, COL_LIVING_DEBT, COL_INTEREST
]

missing = [c for c in needed_cols if c not in df_survey.columns]

if missing:
    st.error("필수 컬럼이 없습니다.")
    st.write(missing)
    st.write("현재 컬럼 목록")
    st.write(df_survey.columns.tolist())
    st.stop()

# 숫자형 변환
for col in needed_cols:
    df_survey[col] = pd.to_numeric(df_survey[col], errors="coerce").fillna(0)

# =========================
# 4. 파생변수 생성
# =========================

df_survey["부모동거"] = df_survey[COL_PARENT].map({1: "부모 동거", 0: "부모 비동거"})
df_survey["가족지원"] = df_survey[COL_FAMILY].map({1: "가족 도움 가능", 0: "가족 도움 없음"})
df_survey["공공지원"] = df_survey[COL_PUBLIC].map({1: "공공지원 가능", 0: "공공지원 없음"})
df_survey["고립여부"] = df_survey[COL_NONE].map({1: "도움 받을 곳 없음", 0: "도움망 있음"})

df_survey["부채보유"] = (df_survey[COL_DEBT] > 0).astype(int)
df_survey["이자부담"] = (df_survey[COL_INTEREST] > 0).astype(int)
df_survey["생활비부채보유"] = (df_survey[COL_LIVING_DEBT] > 0).astype(int)

df_survey["생활비구간"] = pd.cut(
    df_survey[COL_COST],
    bins=[-1, 100, 200, 300, 999999],
    labels=["100만 원 미만", "100~200만 원", "200~300만 원", "300만 원 이상"]
)

df_survey["위험점수"] = (
    (df_survey[COL_PARENT] == 0).astype(int) +
    (df_survey[COL_FAMILY] == 0).astype(int) +
    (df_survey[COL_NONE] == 1).astype(int) +
    (df_survey[COL_DEBT] > 0).astype(int) +
    (df_survey[COL_INTEREST] > 0).astype(int)
)

def classify_type(row):
    if row[COL_NONE] == 1:
        return "고립위험형"
    elif row[COL_FAMILY] == 1 and row[COL_DEBT] == 0:
        return "가족완충형"
    elif row[COL_DEBT] > 0 or row[COL_INTEREST] > 0:
        return "금융부담형"
    elif row[COL_PUBLIC] == 1:
        return "공공지원형"
    else:
        return "취약잠재형"

df_survey["유형분류"] = df_survey.apply(classify_type, axis=1)

# =========================
# 5. 사이드바
# =========================

st.sidebar.title("🔍 분석 필터")

selected_parent = st.sidebar.multiselect(
    "부모 동거 여부",
    df_survey["부모동거"].unique(),
    default=df_survey["부모동거"].unique()
)

selected_family = st.sidebar.multiselect(
    "가족지원 여부",
    df_survey["가족지원"].unique(),
    default=df_survey["가족지원"].unique()
)

selected_isolation = st.sidebar.multiselect(
    "도움망 여부",
    df_survey["고립여부"].unique(),
    default=df_survey["고립여부"].unique()
)

filtered = df_survey[
    (df_survey["부모동거"].isin(selected_parent)) &
    (df_survey["가족지원"].isin(selected_family)) &
    (df_survey["고립여부"].isin(selected_isolation))
]

pages = [
    "1. 규모 현황",
    "2. 무엇으로 버티는가",
    "3. 가족 안전망",
    "4. 가족 밖의 안전망",
    "5. 위험지수",
    "6. 청년 유형 분류",
    "7. 최종 결론"
]

selected_page = st.sidebar.radio("페이지 이동", pages)

# =========================
# 6. 메인 제목
# =========================

st.title("쉬었음 청년은 무엇으로 버티고 있는가?")
st.markdown("### 생활비·부채·소득지원으로 본 청년 비경제활동의 생활안전망 격차")

# =========================
# PAGE 1
# =========================

if selected_page == pages[0]:
    st.header("1. 쉬었음 청년은 얼마나 많은가?")

    st.info("경제활동인구조사 기반 데이터를 활용해 쉬었음 청년 규모를 배경 통계로 확인합니다.")

    year_cols = [c for c in df_trend.columns if str(c).isdigit()]

    if len(year_cols) > 0:
        id_cols = [c for c in df_trend.columns if c not in year_cols]

        trend_long = df_trend.melt(
            id_vars=id_cols,
            value_vars=year_cols,
            var_name="연도",
            value_name="인구"
        )

        trend_long["인구"] = pd.to_numeric(trend_long["인구"], errors="coerce")

        category_col = id_cols[0] if len(id_cols) > 0 else None

        if category_col:
            fig = px.line(
                trend_long,
                x="연도",
                y="인구",
                color=category_col,
                markers=True,
                title="연도별 쉬었음·비경제활동인구 추이"
            )
        else:
            fig = px.line(
                trend_long,
                x="연도",
                y="인구",
                markers=True,
                title="연도별 쉬었음 인구 추이"
            )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("연도 컬럼을 찾지 못했습니다.")
        st.dataframe(df_trend.head())

    st.markdown("""
    **핵심 해석**  
    이 페이지는 쉬었음 청년이 개인의 일시적 문제가 아니라 청년 비경제활동 구조 안에서 확인되는 사회적 현상임을 보여주는 출발점이다.
    """)

# =========================
# PAGE 2
# =========================

elif selected_page == pages[1]:
    st.header("2. 쉬었음 청년은 무엇으로 버티는가?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균 생활비", f"{filtered[COL_COST].mean():,.0f}")
    col2.metric("평균 사적 이전소득", f"{filtered[COL_PRIVATE_INCOME].mean():,.0f}")
    col3.metric("평균 공적 이전소득", f"{filtered[COL_PUBLIC_INCOME].mean():,.0f}")
    col4.metric("평균 부채", f"{filtered[COL_DEBT].mean():,.0f}")

    st.subheader("Sankey Diagram: 생활비를 버티는 경로")

    total_private = filtered[COL_PRIVATE_INCOME].sum()
    total_public = filtered[COL_PUBLIC_INCOME].sum()
    total_debt = filtered[COL_LIVING_DEBT].sum()
    total_income = filtered[COL_TOTAL_INCOME].sum()

    nodes = [
        "사적 이전소득",
        "공적 이전소득",
        "청년 본인 소득",
        "생활비 목적 부채",
        "쉬었음 청년",
        "생활비 지출"
    ]

    sources = [0, 1, 2, 3, 4]
    targets = [4, 4, 4, 4, 5]
    values = [
        total_private,
        total_public,
        total_income,
        total_debt,
        filtered[COL_COST].sum()
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=18,
            line=dict(color="gray", width=0.5),
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(title_text="쉬었음 청년의 생활비 충당 구조", font_size=13)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **핵심 인사이트**  
    이 시각화는 쉬었음 청년이 단순히 소득이 없는 집단이 아니라, 사적 지원·공적 지원·부채를 조합해 생활비를 유지하는 집단임을 보여준다.
    """)

# =========================
# PAGE 3
# =========================

elif selected_page == pages[2]:
    st.header("3. 가족은 얼마나 강력한 안전망인가?")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            filtered,
            x="부모동거",
            y=COL_DEBT,
            color="부모동거",
            title="부모 동거 여부에 따른 부채 총액 차이"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            x="가족지원",
            y=COL_COST,
            color="가족지원",
            title="가족지원 여부에 따른 생활비 수준"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sunburst Chart: 부모동거 → 가족지원 → 부채보유")

    fig = px.sunburst(
        filtered,
        path=["부모동거", "가족지원", "부채보유"],
        values=COL_COST,
        title="가족 안전망 구조"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **핵심 인사이트**  
    부모와 함께 산다는 사실 자체보다 중요한 것은 가족으로부터 실제 도움을 받을 수 있는지 여부다.  
    즉, 쉬었음 청년의 안정성은 개인의 노동상태만이 아니라 가족지원망의 존재 여부에 따라 달라진다.
    """)

# =========================
# PAGE 4
# =========================

elif selected_page == pages[3]:
    st.header("4. 가족이 없다면 무엇으로 버티는가?")

    no_family = filtered[filtered[COL_FAMILY] == 0]

    st.metric("가족 도움 없음 청년 수", f"{len(no_family):,}명")

    col1, col2 = st.columns(2)

    with col1:
        support_sum = pd.DataFrame({
            "지원망": ["지인", "공공기관", "민간기관", "도움 없음"],
            "인원수": [
                no_family[COL_FRIEND].sum(),
                no_family[COL_PUBLIC].sum(),
                no_family[COL_PRIVATE].sum(),
                no_family[COL_NONE].sum()
            ]
        })

        fig = px.bar(
            support_sum,
            x="지원망",
            y="인원수",
            text="인원수",
            title="가족지원이 없는 청년의 대체 지원망"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.density_heatmap(
            no_family,
            x=COL_PUBLIC_INCOME,
            y=COL_DEBT,
            marginal_x="histogram",
            marginal_y="histogram",
            title="가족지원 부재 집단의 공적지원과 부채 관계"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **핵심 인사이트**  
    가족지원이 없는 청년에게 공공지원이 충분히 도달하지 못하면, 생활비를 버티는 수단은 부채 또는 고립으로 이동할 가능성이 크다.
    """)

# =========================
# PAGE 5
# =========================

elif selected_page == pages[4]:
    st.header("5. 누가 가장 위험한 쉬었음 청년인가?")

    avg_risk = filtered["위험점수"].mean()

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        title={"text": "평균 생활안전망 위험점수"},
        gauge={
            "axis": {"range": [0, 5]},
            "steps": [
                {"range": [0, 1.5], "color": "#d8f3dc"},
                {"range": [1.5, 3], "color": "#fff3b0"},
                {"range": [3, 5], "color": "#ffccd5"}
            ],
            "bar": {"color": "#d00000"}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x="위험점수",
            nbins=6,
            title="위험점수 분포"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            x="위험점수",
            y=COL_DEBT,
            color="위험점수",
            title="위험점수별 부채 총액"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **위험점수 구성**  
    - 부모 비동거  
    - 가족지원 없음  
    - 도움 받을 곳 없음  
    - 부채 있음  
    - 이자 부담 있음  

    **핵심 인사이트**  
    쉬었음 청년 내부에서도 위험요소가 중첩되는 집단이 존재한다.  
    이들은 단순 미취업자가 아니라 생활안전망이 동시에 약한 정책 우선 대상이다.
    """)

# =========================
# PAGE 6
# =========================

elif selected_page == pages[5]:
    st.header("6. 쉬었음 청년의 유형 분류")

    type_count = filtered["유형분류"].value_counts().reset_index()
    type_count.columns = ["유형분류", "인원수"]

    fig = px.treemap(
        type_count,
        path=["유형분류"],
        values="인원수",
        title="쉬었음 청년 유형별 규모"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered,
        x=COL_COST,
        y=COL_TOTAL_INCOME,
        size=COL_DEBT,
        color="유형분류",
        hover_data=[
            COL_PRIVATE_INCOME,
            COL_PUBLIC_INCOME,
            COL_DEBT,
            COL_INTEREST,
            "위험점수"
        ],
        title="생활비·소득·부채로 본 쉬었음 청년 유형"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **유형 정의**  
    - 가족완충형: 가족지원이 있고 부채가 없는 집단  
    - 금융부담형: 부채 또는 이자 부담이 있는 집단  
    - 공공지원형: 공공기관 도움 가능성이 있는 집단  
    - 고립위험형: 도움 받을 곳이 없는 집단  
    - 취약잠재형: 뚜렷한 지원망과 부채가 모두 약한 집단  
    """)

# =========================
# PAGE 7
# =========================

else:
    st.header("7. 한국 청년의 진짜 안전망은 무엇인가?")

    safety = pd.DataFrame({
        "안전망": ["가족", "지인", "공공기관", "민간기관", "도움 없음"],
        "비율": [
            filtered[COL_FAMILY].mean(),
            filtered[COL_FRIEND].mean(),
            filtered[COL_PUBLIC].mean(),
            filtered[COL_PRIVATE].mean(),
            filtered[COL_NONE].mean()
        ]
    })

    fig = px.bar(
        safety,
        x="안전망",
        y="비율",
        text=safety["비율"].apply(lambda x: f"{x:.1%}"),
        title="쉬었음 청년의 생활안전망 구성"
    )
    st.plotly_chart(fig, use_container_width=True)

    categories = ["가족지원", "지인지원", "공공지원", "민간지원", "비고립"]
    values = [
        filtered[COL_FAMILY].mean(),
        filtered[COL_FRIEND].mean(),
        filtered[COL_PUBLIC].mean(),
        filtered[COL_PRIVATE].mean(),
        1 - filtered[COL_NONE].mean()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="생활안전망 수준"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="쉬었음 청년의 안전망 레이더 차트"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.error("""
    최종 결론: 쉬었음 청년은 단순히 일을 하지 않는 집단이 아니라,
    가족지원·공공지원·부채·도움망의 차이에 따라 전혀 다른 방식으로 버티는 집단이다.
    특히 가족지원이 약하고 부채와 고립이 중첩된 청년은 생활안전망 격차의 최전선에 있다.
    """)
