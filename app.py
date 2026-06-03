import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


# =========================================================
# 0. 기본 설정
# =========================================================

st.set_page_config(
    page_title="쉬었음 청년 생활안전망 격차 분석",
    layout="wide"
)

DB_PATH = "쉬었음 청년 분석.db"

SURVEY_TABLE = "EDA용 2024 청년삶실태조사"
TREND_TABLE = "연령_활동상태별_쉬었음__비경제활동인구"
PANEL_TABLE = "한국노동패널조사"


# =========================================================
# 1. 스타일
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .insight-box {
        background-color: #f7f5ff;
        border-left: 6px solid #6c63ff;
        padding: 1.05rem 1.2rem;
        border-radius: 0.65rem;
        margin-top: 1rem;
        line-height: 1.75;
        font-size: 0.98rem;
    }
    .policy-box {
        background-color: #f0f8ff;
        border-left: 6px solid #2f80ed;
        padding: 1.05rem 1.2rem;
        border-radius: 0.65rem;
        margin-top: 1rem;
        line-height: 1.75;
        font-size: 0.98rem;
    }
    .conclusion-box {
        background-color: #fff8e6;
        border-left: 7px solid #f2a900;
        padding: 1.2rem 1.3rem;
        border-radius: 0.7rem;
        margin-top: 1.2rem;
        line-height: 1.8;
        font-size: 1rem;
    }
    .small-note {
        color: #666;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2. 데이터 로드
# =========================================================

@st.cache_data
def get_table_names():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )
    conn.close()
    return tables["name"].tolist()


@st.cache_data
def load_table(table_name):
    if not os.path.exists(DB_PATH):
        st.error(f"DB 파일을 찾을 수 없습니다: {DB_PATH}")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM '{table_name}'", conn)
    conn.close()
    return df


try:
    df_survey = load_table(SURVEY_TABLE)
    df_trend = load_table(TREND_TABLE)
    df_panel = load_table(PANEL_TABLE)
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.write("현재 DB 안 테이블 목록")
    st.write(get_table_names())
    st.stop()


# =========================================================
# 3. 컬럼 설정
# =========================================================

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

missing = [col for col in needed_cols if col not in df_survey.columns]

if missing:
    st.error("필수 컬럼이 없습니다. DB 컬럼명을 확인해주세요.")
    st.write("없는 컬럼")
    st.write(missing)
    st.write("현재 컬럼 목록")
    st.write(df_survey.columns.tolist())
    st.stop()


# =========================================================
# 4. 전처리와 파생변수
# =========================================================

for col in needed_cols:
    df_survey[col] = pd.to_numeric(df_survey[col], errors="coerce").fillna(0)

df_survey["부모동거"] = np.where(df_survey[COL_PARENT] == 1, "부모 동거", "부모 비동거")
df_survey["가족지원"] = np.where(df_survey[COL_FAMILY] == 1, "가족 도움 가능", "가족 도움 없음")
df_survey["공공지원"] = np.where(df_survey[COL_PUBLIC] == 1, "공공기관 도움 가능", "공공기관 도움 없음")
df_survey["고립여부"] = np.where(df_survey[COL_NONE] == 1, "도움 받을 곳 없음", "도움망 있음")

df_survey["부채여부"] = np.where(df_survey[COL_DEBT] > 0, "부채 있음", "부채 없음")
df_survey["이자부담여부"] = np.where(df_survey[COL_INTEREST] > 0, "이자 부담 있음", "이자 부담 없음")
df_survey["생활비부채여부"] = np.where(df_survey[COL_LIVING_DEBT] > 0, "생활비 부채 있음", "생활비 부채 없음")

df_survey["생활비구간"] = pd.cut(
    df_survey[COL_COST],
    bins=[-1, 100, 200, 300, 999999],
    labels=["100만 원 미만", "100~200만 원", "200~300만 원", "300만 원 이상"]
)

df_survey["위험점수"] = (
    (df_survey[COL_PARENT] != 1).astype(int) +
    (df_survey[COL_FAMILY] == 0).astype(int) +
    (df_survey[COL_NONE] == 1).astype(int) +
    (df_survey[COL_DEBT] > 0).astype(int) +
    (df_survey[COL_INTEREST] > 0).astype(int)
)

df_survey["위험수준"] = pd.cut(
    df_survey["위험점수"],
    bins=[-1, 1, 3, 5],
    labels=["저위험", "중위험", "고위험"]
)

def classify_type(row):
    if row[COL_NONE] == 1:
        return "고립위험형"
    if row[COL_DEBT] > 0 or row[COL_INTEREST] > 0 or row[COL_LIVING_DEBT] > 0:
        return "금융부담형"
    if row[COL_FAMILY] == 1 and row[COL_PARENT] == 1:
        return "가족완충형"
    if row[COL_PUBLIC] == 1:
        return "공공지원형"
    if row[COL_FRIEND] == 1 or row[COL_PRIVATE] == 1:
        return "대체지원형"
    return "취약잠재형"

df_survey["생활안전망유형"] = df_survey.apply(classify_type, axis=1)


# =========================================================
# 5. 공통 함수
# =========================================================

def safe_mean(series):
    if len(series) == 0:
        return 0
    value = pd.to_numeric(series, errors="coerce").mean()
    if pd.isna(value):
        return 0
    return value


def safe_sum(series):
    if len(series) == 0:
        return 0
    value = pd.to_numeric(series, errors="coerce").sum()
    if pd.isna(value):
        return 0
    return value


def fmt_num(x):
    if pd.isna(x):
        return "0"
    return f"{x:,.0f}"


def holding_rate(series):
    s = pd.to_numeric(series, errors="coerce")
    if len(s) == 0:
        return 0.0
    return float((s > 0).mean())


def holder_median(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[s > 0]
    if len(s) == 0:
        return 0.0
    return float(s.median())


def fmt_p(p):
    if pd.isna(p):
        return "계산 불가"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def mann_whitney(data, group_col, label_a, label_b, value_col, min_n=10):
    ga = pd.to_numeric(data.loc[data[group_col] == label_a, value_col], errors="coerce").dropna()
    gb = pd.to_numeric(data.loc[data[group_col] == label_b, value_col], errors="coerce").dropna()
    if len(ga) < min_n or len(gb) < min_n:
        return None
    try:
        _, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
    except ValueError:
        return None
    return {
        "n_a": int(len(ga)), "n_b": int(len(gb)),
        "med_a": float(ga.median()), "med_b": float(gb.median()),
        "p": float(p)
    }


def chi2_holding(data, group_col, value_col, threshold=0, min_n=10):
    sub = data[[group_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    sub["__has"] = sub[value_col] > threshold
    ct = pd.crosstab(sub[group_col], sub["__has"])
    if ct.shape[0] < 2 or ct.shape[1] < 2 or int(ct.values.sum()) < min_n:
        return None
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    method = "카이제곱 검정"
    if ct.shape == (2, 2) and (expected < 5).any():
        _, p = stats.fisher_exact(ct.values)
        method = "Fisher 정확검정"
    rates = sub.groupby(group_col)["__has"].mean().to_dict()
    return {"method": method, "p": float(p), "rates": rates}


def insight_box(title, body):
    st.markdown(
        f"""
        <div class="insight-box">
        <b>{title}</b><br>
        {body}
        </div>
        """,
        unsafe_allow_html=True
    )


def policy_box(title, body):
    st.markdown(
        f"""
        <div class="policy-box">
        <b>{title}</b><br>
        {body}
        </div>
        """,
        unsafe_allow_html=True
    )


def conclusion_box(body):
    st.markdown(
        f"""
        <div class="conclusion-box">
        {body}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_filter_summary(data):
    st.caption(
        f"현재 분석 대상: {len(data):,}명 / 전체 {len(df_survey):,}명 "
        f"({len(data) / len(df_survey) * 100:.1f}%)"
    )


# =========================================================
# 6. 사이드바 필터
# =========================================================

st.sidebar.title("🔍 분석 필터")

st.sidebar.markdown(
    """
    <div class="small-note">
    기본값은 <b>전체 데이터</b>입니다.  
    특정 유형이나 조건을 선택하면 해당 집단의 생활안전망 구조를 세부적으로 볼 수 있습니다.
    </div>
    """,
    unsafe_allow_html=True
)

analysis_type = st.sidebar.selectbox(
    "생활안전망 유형",
    ["전체"] + sorted(df_survey["생활안전망유형"].dropna().unique().tolist())
)

risk_level = st.sidebar.selectbox(
    "위험수준",
    ["전체", "저위험", "중위험", "고위험"]
)

parent_filter = st.sidebar.selectbox(
    "부모 동거 여부",
    ["전체", "부모 동거", "부모 비동거"]
)

family_filter = st.sidebar.selectbox(
    "가족지원 여부",
    ["전체", "가족 도움 가능", "가족 도움 없음"]
)

public_filter = st.sidebar.selectbox(
    "공공지원 접근성",
    ["전체", "공공기관 도움 가능", "공공기관 도움 없음"]
)

debt_filter = st.sidebar.selectbox(
    "부채 여부",
    ["전체", "부채 있음", "부채 없음"]
)

interest_filter = st.sidebar.selectbox(
    "이자 부담 여부",
    ["전체", "이자 부담 있음", "이자 부담 없음"]
)

isolation_filter = st.sidebar.selectbox(
    "도움망 여부",
    ["전체", "도움망 있음", "도움 받을 곳 없음"]
)

cost_filter = st.sidebar.selectbox(
    "생활비 구간",
    ["전체", "100만 원 미만", "100~200만 원", "200~300만 원", "300만 원 이상"]
)

filtered = df_survey.copy()

if analysis_type != "전체":
    filtered = filtered[filtered["생활안전망유형"] == analysis_type]

if parent_filter != "전체":
    filtered = filtered[filtered["부모동거"] == parent_filter]

if family_filter != "전체":
    filtered = filtered[filtered["가족지원"] == family_filter]

if public_filter != "전체":
    filtered = filtered[filtered["공공지원"] == public_filter]

if debt_filter != "전체":
    filtered = filtered[filtered["부채여부"] == debt_filter]

if interest_filter != "전체":
    filtered = filtered[filtered["이자부담여부"] == interest_filter]

if isolation_filter != "전체":
    filtered = filtered[filtered["고립여부"] == isolation_filter]

if cost_filter != "전체":
    filtered = filtered[filtered["생활비구간"].astype(str) == cost_filter]

# 위험수준 필터는 위험점수로 직접 정의되므로, 위험점수 기반 지표(위험군 비율 게이지)에서는
# 순환참조를 피하기 위해 적용 직전의 집단을 따로 보관한다.
filtered_excl_risk = filtered

if risk_level != "전체":
    filtered = filtered[filtered["위험수준"].astype(str) == risk_level]

if len(filtered) == 0:
    st.warning("현재 필터 조건에 해당하는 데이터가 없습니다. 필터를 완화해주세요.")
    st.stop()

pages = [
    "1. 전체 규모와 문제 제기",
    "2. 무엇으로 버티는가",
    "3. 가족 안전망",
    "4. 가족 밖의 안전망",
    "5. 위험지수 분석",
    "6. 생활안전망 유형",
    "7. 최종 결론"
]

selected_page = st.sidebar.radio("페이지 이동", pages)


# =========================================================
# 7. 메인 제목
# =========================================================

st.markdown('<div class="main-title">쉬었음 청년은 무엇으로 버티고 있는가?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">생활비·부채·소득지원으로 본 청년 비경제활동의 생활안전망 격차</div>',
    unsafe_allow_html=True
)

show_filter_summary(filtered)


# =========================================================
# PAGE 1
# =========================================================

if selected_page == pages[0]:
    st.header("1. 전체 규모와 문제 제기")

    st.markdown(
        """
        이 페이지는 ‘쉬었음 청년’이 개인적 선택이나 일시적 휴식으로만 설명되기 어렵다는 점을 보여주는 출발점이다.  
        경제활동인구조사 기반 자료를 통해 청년층 비경제활동과 ‘쉬었음’ 상태의 규모를 먼저 확인하고, 이후 생활비·소득지원·부채·지원망 분석으로 연결한다.
        """
    )

    year_cols = [c for c in df_trend.columns if str(c).isdigit()]

    if len(year_cols) > 0:
        id_cols = [c for c in df_trend.columns if c not in year_cols]

        trend_long = df_trend.melt(
            id_vars=id_cols,
            value_vars=year_cols,
            var_name="연도",
            value_name="인구"
        )

        trend_long["인구"] = pd.to_numeric(
            trend_long["인구"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        category_col = id_cols[0] if len(id_cols) > 0 else None

        if category_col:
            age_bands = ["15 - 19세", "20 - 29세", "30 - 39세", "40 - 49세", "50 - 59세", "60세이상"]
            trend_long[category_col] = trend_long[category_col].astype(str).str.strip()
            age_long = trend_long[trend_long[category_col].isin(age_bands)].copy()

            if len(age_long) == 0:
                st.warning("연령대 행을 찾지 못해 전체 추이를 표시합니다.")
                age_long = trend_long
            else:
                age_long[category_col] = pd.Categorical(
                    age_long[category_col], categories=age_bands, ordered=True
                )
                age_long = age_long.sort_values([category_col, "연도"])

            fig = px.line(
                age_long,
                x="연도",
                y="인구",
                color=category_col,
                markers=True,
                title="연도별 연령대별 쉬었음 인구 추이 (천 명)"
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "※ ‘전체’·‘15–64세’·‘15–29세’·‘15–24세’처럼 다른 구간과 겹치는 합계 구간은 중복 집계라 제외하고, "
                "서로 겹치지 않는 단일 연령대만 표시했습니다. 청년(15–19세·20–29세)을 다른 연령대와 비교해 보세요. 단위: 천 명."
            )
        else:
            fig = px.line(
                trend_long,
                x="연도",
                y="인구",
                markers=True,
                title="연도별 쉬었음 인구 추이"
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("연도형 컬럼을 찾지 못했습니다. 원자료 일부를 표시합니다.")
        st.dataframe(df_trend.head(20), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 대상 수", f"{len(filtered):,}명")
    col2.metric("평균 생활비", fmt_num(safe_mean(filtered[COL_COST])))
    col3.metric("부채 보유율", f"{holding_rate(filtered[COL_DEBT]):.1%}")
    col4.metric("평균 위험점수", f"{safe_mean(filtered['위험점수']):.2f} / 5")

    st.caption(
        f"※ 부채는 응답자의 약 {1 - holding_rate(filtered[COL_DEBT]):.0%}가 0이라 평균이 소수 고액 부채자에 좌우됩니다. "
        f"부채 보유자({holding_rate(filtered[COL_DEBT]):.1%})의 부채 중앙값은 {fmt_num(holder_median(filtered[COL_DEBT]))}만 원입니다. "
        "(평균 부채 표기 대신 보유율·보유자 중앙값으로 표시)"
    )

    insight_box(
        "해석",
        """
        이 프로젝트의 핵심은 ‘쉬었음 청년이 많다’는 규모 확인에서 멈추지 않는 것이다.
        규모 통계는 문제의 사회적 배경을 보여주지만, 정작 중요한 질문은 이들이 노동시장 밖에 있는 동안 어떤 자원으로 생활을 유지하는가이다.
        위 추이에서는 청년(15–19·20–29세)을 다른 연령대와 비교해 쉬었음의 흐름을 가늠할 수 있고, 부채처럼 응답자의 대다수가 0인 변수는 평균 대신 ‘보유율’로 보아 소수의 고액 사례에 평균이 휘둘리지 않도록 했다.
        같은 쉬었음 상태라도 가족의 도움을 받을 수 있는지, 부채·이자 부담을 지는지에 따라 생활 조건은 크게 달라지므로, 이후 분석은 이 내부 격차를 추적한다.
        """
    )

    policy_box(
        "분석 방향",
        """
        전체 규모는 정책적 관심의 필요성을 보여주고, 세부 분석은 정책 우선순위를 정하는 근거가 된다.
        단순히 ‘쉬었음 청년 전체’를 지원 대상으로 설정하면 실제 위험이 높은 집단이 흐려질 수 있다.
        본 대시보드는 가족지원망, 공적지원 접근성, 부채·이자 부담, 도움 없음 여부를 기준으로 누가 더 취약한지를 구분하는 데 초점을 둔다.
        """
    )


# =========================================================
# PAGE 2
# =========================================================

elif selected_page == pages[1]:
    st.header("2. 쉬었음 청년은 무엇으로 버티는가?")

    st.caption(
        "※ 단위 주의: 소득·이전소득은 연 단위, 생활비·이자는 월 단위로 조사된 값입니다. "
        "비교를 위해 생활비는 연 환산(월×12)했으며, 모든 금액은 1인 평균·만원 기준입니다. "
        "또한 '청년 연간소득(총소득)'에는 사적·공적 이전소득이 이미 포함되어 있어(전체 응답자 100% 확인), "
        "중복 합산을 피하려고 총소득을 '자력소득(근로·사업 등) / 사적 이전 / 공적 이전'으로 분해했습니다."
    )

    mean_total = safe_mean(filtered[COL_TOTAL_INCOME])
    mean_private = safe_mean(filtered[COL_PRIVATE_INCOME])
    mean_public = safe_mean(filtered[COL_PUBLIC_INCOME])
    mean_self = max(mean_total - mean_private - mean_public, 0)
    mean_living_debt = safe_mean(filtered[COL_LIVING_DEBT])
    mean_cost_year = safe_mean(filtered[COL_COST]) * 12

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균 연 생활비(월×12)", fmt_num(mean_cost_year))
    col2.metric("평균 자력소득(연)", fmt_num(mean_self))
    col3.metric("평균 사적 이전(연)", fmt_num(mean_private))
    col4.metric("평균 공적 이전(연)", fmt_num(mean_public))

    st.subheader("소득은 무엇으로 구성되는가: 소득 구성 Sankey")

    nodes = [
        "자력소득(근로·사업 등)",
        "사적 이전소득",
        "공적 이전소득",
        "청년 연간 총소득"
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=22,
            thickness=20,
            line=dict(color="rgba(80,80,80,0.4)", width=0.6),
            label=nodes,
            color=["#51cf66", "#7b68ee", "#4dabf7", "#495057"]
        ),
        link=dict(
            source=[0, 1, 2],
            target=[3, 3, 3],
            value=[mean_self, mean_private, mean_public],
            color=[
                "rgba(81,207,102,0.35)",
                "rgba(123,104,238,0.35)",
                "rgba(77,171,247,0.35)"
            ]
        )
    )])
    fig.update_layout(
        title_text="쉬었음 청년의 연간 소득 구성 (1인 평균, 만원)",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "세 구성요소(자력소득 + 사적 이전 + 공적 이전)의 합이 청년 연간 총소득과 정확히 일치하도록 "
        "분해해, 흐름이 보존되고 이전소득이 중복 계산되지 않습니다."
    )

    st.subheader("연간 생활비와 가용 자원 비교 (1인 평균, 만원)")

    compare_df = pd.DataFrame({
        "항목": ["연 환산 생활비", "자력소득", "사적 이전소득", "공적 이전소득", "생활비 목적 부채(잔액)"],
        "금액": [mean_cost_year, mean_self, mean_private, mean_public, mean_living_debt],
        "구분": ["지출", "소득", "소득", "소득", "부채"]
    })

    fig_bar = px.bar(
        compare_df,
        x="항목",
        y="금액",
        color="구분",
        text="금액",
        title="생활비 대비 소득·부채 자원 (1인 평균, 만원)"
    )
    fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(height=460)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "생활비는 가구 기준, 소득은 청년 개인 기준일 수 있어 두 값의 직접적인 차감 해석은 주의가 필요합니다. "
        "여기서는 수준 비교용으로만 제시합니다."
    )

    insight_box(
        "핵심 인사이트",
        """
        이 분석의 초점은 쉬었음 청년에게 소득이 없다는 사실 자체가 아니라, 소득이 어떤 출처로 구성되는지에 있다.
        총소득을 자력소득(근로·사업 등)·사적 이전·공적 이전으로 분해해 보면, 정부의 공적 이전이 차지하는 몫은 매우 작고 가족·주변에서 오는 사적 이전이 그보다 크게 나타난다.
        즉 노동시장 밖에 있는 동안의 소득 공백을 공적 안전망보다 가족 자원이 메우는 구조에 가깝다.
        또한 연 환산 생활비가 청년 개인의 연소득을 웃도는 경우가 많은데, 이는 생활비가 가구 단위로 충당되고 있을 가능성, 곧 부모·가족의 비화폐적 지원에 기대고 있을 가능성을 시사한다.
        """
    )

    policy_box(
        "정책적 시사점",
        """
        지원정책은 쉬었음 여부만이 아니라 소득이 무엇으로 구성되는지에 따라 달라져야 한다.
        공적 이전의 몫이 작다는 것은 현재의 공공 안전망이 가족 자원의 공백을 충분히 대체하지 못하고 있음을 뜻한다.
        가족 자원에 의존하는 구조는 가족의 경제력 차이에 따라 청년의 생활 안정성을 갈리게 만들고, 부족분을 부채로 메우는 청년에게는 금융 취약성이 시간이 지날수록 누적될 수 있다.
        따라서 공적 지원의 사각지대와 가족 의존의 격차를 함께 살펴야 한다.
        """
    )


# =========================================================
# PAGE 3
# =========================================================

elif selected_page == pages[2]:
    st.header("3. 가족은 얼마나 강력한 안전망인가?")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            filtered,
            x="부모동거",
            y=COL_DEBT,
            color="부모동거",
            title="부모 동거 여부에 따른 부채 총액"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            x="가족지원",
            y=COL_COST,
            color="가족지원",
            title="가족지원 여부에 따른 월평균 생활비"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("통계 검정: 가족 안전망 효과가 유의한가?")

    chi_debt = chi2_holding(filtered, "부모동거", COL_DEBT)
    mw_debt = mann_whitney(filtered, "부모동거", "부모 동거", "부모 비동거", COL_DEBT)
    mw_cost = mann_whitney(filtered, "가족지원", "가족 도움 가능", "가족 도움 없음", COL_COST)

    lines = []
    if chi_debt:
        r = chi_debt["rates"]
        lines.append(
            f"- **부모 동거 ↔ 부채 보유**({chi_debt['method']}): p = {fmt_p(chi_debt['p'])} · "
            f"보유율 동거 {r.get('부모 동거', float('nan')):.1%} vs 비동거 {r.get('부모 비동거', float('nan')):.1%}"
        )
    if mw_debt:
        lines.append(
            f"- **부모 동거 ↔ 부채 총액**(Mann–Whitney U): p = {fmt_p(mw_debt['p'])} · "
            f"중앙값 동거 {fmt_num(mw_debt['med_a'])} vs 비동거 {fmt_num(mw_debt['med_b'])}만 원 "
            f"(n = {mw_debt['n_a']} / {mw_debt['n_b']})"
        )
    if mw_cost:
        lines.append(
            f"- **가족지원 ↔ 월 생활비**(Mann–Whitney U): p = {fmt_p(mw_cost['p'])} · "
            f"중앙값 도움가능 {fmt_num(mw_cost['med_a'])} vs 도움없음 {fmt_num(mw_cost['med_b'])}만 원 "
            f"(n = {mw_cost['n_a']} / {mw_cost['n_b']})"
        )

    if lines:
        st.markdown("\n".join(lines))
    else:
        st.info("현재 필터에서는 두 비교 집단의 표본이 부족해(각 n<10) 통계 검정을 생략했습니다. 필터를 완화하면 결과가 표시됩니다.")

    st.caption(
        "검정 방법: 부채·생활비는 분포가 0 쪽으로 크게 치우쳐 정규성을 가정하는 t-검정 대신 "
        "비모수 검정인 Mann–Whitney U와 카이제곱(기대빈도가 5 미만이면 Fisher 정확검정)을 사용했습니다. "
        "각 집단 n<10이면 검정을 생략합니다. 위험점수·생활안전망유형은 부채·가족 변수로 정의된 파생지표여서 "
        "순환논리를 피하려고 검정 대상에서 제외했습니다."
    )

    st.subheader("부모동거 → 가족지원 → 부채여부 구조")

    sunburst_df = filtered.copy()
    fig = px.sunburst(
        sunburst_df,
        path=["부모동거", "가족지원", "부채여부"],
        values=COL_COST,
        title="가족 안전망의 단계별 구조"
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "핵심 인사이트",
        """
        부모와 함께 산다는 사실은 쉬었음 청년에게 주거비와 생활비 부담을 완화하는 중요한 조건일 수 있다.
        위의 검정 결과를 보면, 부모와 동거하는 청년은 비동거 청년보다 부채 보유율과 부채 규모가 낮은 쪽으로 통계적으로 의미 있는 차이를 보인다.
        다만 가족지원이 가능한 청년의 월 생활비가 오히려 더 높게 나타나는데, 이는 가족 자원이 지출을 무조건 줄인다기보다 일정 수준의 생활을 유지하게 하는 완충 역할을 한다는 해석과 맞닿는다.
        결국 생활안전망은 ‘동거 여부’와 ‘가족에게 도움을 요청할 수 있는지’가 결합될 때 더 선명해지며, 둘 다 약한 청년은 생활비 압박과 부채 위험에 더 쉽게 노출된다.
        따라서 가족은 단순한 배경 변수가 아니라 쉬었음 청년의 생활비 지속 가능성을 좌우하는 핵심 안전망이다.
        """
    )

    policy_box(
        "정책적 시사점",
        """
        청년지원 정책은 가구 단위의 자원 차이를 충분히 반영해야 한다.
        같은 비경제활동 청년이라도 부모와 동거하며 지원을 받는 청년과 독립 상태에서 도움을 받지 못하는 청년은 위험 수준이 다르다.
        특히 가족지원을 전제로 한 정책 설계는 가족자원이 약한 청년을 사각지대에 남길 수 있으므로, 가족지원 가능성 자체를 정책 선별 기준 중 하나로 고려할 필요가 있다.
        """
    )


# =========================================================
# PAGE 4
# =========================================================

elif selected_page == pages[3]:
    st.header("4. 가족이 없다면 무엇으로 버티는가?")

    no_family = filtered[filtered[COL_FAMILY] == 0]

    if len(no_family) == 0:
        st.warning("현재 필터 조건에서 가족 도움 없음 집단이 없습니다. 필터를 완화하면 확인할 수 있습니다.")
    else:
        st.metric("가족 도움 없음 집단", f"{len(no_family):,}명")

        support_sum = pd.DataFrame({
            "지원망": ["지인", "공공기관", "민간기관", "도움 받을 곳 없음"],
            "인원수": [
                safe_sum(no_family[COL_FRIEND]),
                safe_sum(no_family[COL_PUBLIC]),
                safe_sum(no_family[COL_PRIVATE]),
                safe_sum(no_family[COL_NONE])
            ]
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                support_sum,
                x="지원망",
                y="인원수",
                text="인원수",
                title="가족지원이 없는 청년의 대체 지원망"
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            exposure = pd.DataFrame({
                "항목": ["공적지원 보유", "부채 보유", "이자 부담", "도움 받을 곳 없음"],
                "비율": [
                    safe_mean(no_family[COL_PUBLIC]),
                    (no_family[COL_DEBT] > 0).mean(),
                    (no_family[COL_INTEREST] > 0).mean(),
                    safe_mean(no_family[COL_NONE])
                ]
            })
            fig = px.bar(
                exposure,
                x="항목",
                y="비율",
                text=exposure["비율"].apply(lambda v: f"{v:.1%}"),
                title="가족지원 부재 집단의 공적지원·부채·고립 노출 비율"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=480, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            rho = no_family[COL_PUBLIC_INCOME].corr(no_family[COL_DEBT], method="spearman")
            rho_txt = "계산 불가" if pd.isna(rho) else f"{rho:.3f}"
            st.caption(
                f"※ 공적 이전소득과 부채 총액의 순위상관(Spearman)은 {rho_txt}로 사실상 관계가 없고, "
                "두 변수 모두 값이 0인 경우가 대부분이라 분포 패턴이 드러나지 않습니다. "
                "그래서 '관계'를 그리는 대신 공적지원·부채·고립 노출 비율 비교로 대체했습니다."
            )

        st.subheader("가족지원 부재 집단의 생활비·소득·부채 분포")

        fig = px.scatter(
            no_family,
            x=COL_COST,
            y=COL_TOTAL_INCOME,
            size=COL_DEBT,
            color="고립여부",
            hover_data=[COL_PUBLIC_INCOME, COL_PRIVATE_INCOME, COL_LIVING_DEBT, COL_INTEREST, "위험점수"],
            title="가족지원이 없는 청년은 공공지원·부채·고립 중 어디에 가까운가?"
        )
        fig.update_layout(height=580)
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "핵심 인사이트",
        """
        가족지원이 없는 청년을 따로 보는 이유는 이들이 쉬었음 청년 내부에서 가장 중요한 정책적 분기점이 될 수 있기 때문이다.
        가족지원이 작동하지 않을 때 청년은 지인·공공기관·민간기관·금융자원, 또는 아무 도움도 없는 상태 중 하나로 이동한다.
        위 비율 비교를 보면 이 집단에서는 ‘도움 받을 곳 없음’의 비중이 공적지원 보유 비중보다 크게 나타나, 공공 안전망이 가족지원의 공백을 충분히 대체하지 못하고 있음을 보여준다.
        또한 공적 이전소득과 부채 사이에는 뚜렷한 상관이 없어, 공적지원이 부채 위험을 체계적으로 줄여 주는 구조라고 보기도 어렵다.
        특히 가족지원도 없고 공적지원도 약하며 부채·이자 부담이 동반되는 청년은 단기적 생활비 문제를 장기적 금융 취약성으로 전환시킬 위험이 있다.
        """
    )

    policy_box(
        "정책적 시사점",
        """
        가족지원 부재 집단은 단순히 ‘취약한 청년’이 아니라 사적 안전망이 끊어진 집단이다.
        이들에게 필요한 정책은 일괄적 취업지원만이 아니라 생활비 긴급지원, 채무 부담 완화, 공적지원 접근성 개선이 결합된 형태여야 한다.
        본 분석은 청년 비경제활동 문제를 노동시장 복귀만의 문제가 아니라 생활 유지 기반의 문제로 확장해 볼 필요가 있음을 보여준다.
        """
    )


# =========================================================
# PAGE 5
# =========================================================

elif selected_page == pages[4]:
    st.header("5. 누가 가장 위험한 쉬었음 청년인가?")

    # 위험군 비율은 위험점수로 정의되므로, 같은 변수를 거르는 '위험수준' 필터는 제외하고 계산한다
    # (위험수준 필터를 적용하면 저위험=0% / 중·고위험=100%로 고정돼 의미가 사라짐).
    gauge_base = filtered_excl_risk
    risk_share = round((gauge_base["위험점수"] >= 2).mean() * 100, 1)
    overall_share = round((df_survey["위험점수"] >= 2).mean() * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_share,
        number={"suffix": "%", "valueformat": ".1f"},
        delta={
            "reference": overall_share,
            "valueformat": ".1f",
            "suffix": "%p",
            "increasing": {"color": "#e03131"},
            "decreasing": {"color": "#2f9e44"}
        },
        title={"text": "위험군 비율 (위험요인 2개 이상 중첩)<br><span style='font-size:0.8em;color:gray'>(검은 선 = 전체 비율, ▲/▼ = 전체 대비 %p)</span>"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "steps": [
                {"range": [0, 15], "color": "#d8f3dc"},
                {"range": [15, 30], "color": "#fff3b0"},
                {"range": [30, 100], "color": "#ffccd5"}
            ],
            "bar": {"color": "#6c63ff"},
            "threshold": {
                "line": {"color": "#343a40", "width": 3},
                "thickness": 0.85,
                "value": overall_share
            }
        }
    ))
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"보라색 막대는 현재 분석 대상 중 위험요인을 2개 이상(중·고위험) 가진 청년의 비율, "
        f"검은 기준선은 전체 비율({overall_share:.1f}%)입니다. ▲/▼는 전체 대비 차이(%p)입니다. "
        "평균 위험점수는 0점대가 많아 낮게 보이므로, 위험이 중첩된 집단의 규모로 심각성을 표시했습니다. "
        "(색 구간 0–15·15–30·30%+는 시각적 참고용 임의 구분입니다.)"
    )
    if risk_level != "전체":
        st.caption(
            f"※ 게이지는 위험점수로 정의되는 지표라 '위험수준({risk_level})' 필터는 적용하지 않았습니다. "
            "아래 분포·표·박스플롯에는 위험수준 필터가 반영됩니다."
        )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x="위험점수",
            nbins=6,
            color="위험수준",
            title="생활안전망 위험점수 분포"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            x="위험수준",
            y=COL_DEBT,
            color="위험수준",
            title="위험수준별 부채 총액"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    risk_table = filtered.groupby("위험수준", observed=True).agg(
        인원수=("위험점수", "count"),
        평균생활비=(COL_COST, "mean"),
        부채보유율=(COL_DEBT, holding_rate),
        부채중앙값_보유자=(COL_DEBT, holder_median),
        이자보유율=(COL_INTEREST, holding_rate),
        가족지원비율=(COL_FAMILY, "mean"),
        공공지원비율=(COL_PUBLIC, "mean"),
        도움없음비율=(COL_NONE, "mean")
    ).reset_index()

    st.subheader("위험수준별 생활안전망 특성")
    st.dataframe(risk_table, use_container_width=True)
    st.caption(
        "※ 부채·이자는 0인 응답자가 대부분이라 평균 대신 '보유율'과 '보유자 한정 중앙값(만원)'으로 표시했습니다. "
        "비율 컬럼은 0–1(=0–100%) 값입니다."
    )

    insight_box(
        "핵심 인사이트",
        """
        위험점수는 취약성을 하나의 값으로 단순화하려는 것이 아니라, 위험요인이 얼마나 중첩되는지를 보여주는 지표다.
        부모 비동거, 가족지원 없음, 도움 받을 곳 없음, 부채 보유, 이자 부담의 다섯 요인을 합산하며(모두 충족 시 5점), 점수가 높을수록 여러 취약 요인이 동시에 작동함을 뜻한다.
        분포를 보면 대다수는 저위험에 몰려 있고 고위험은 소수에 그치지만, 이 소수는 단지 소득이 낮은 집단이 아니라 생활비를 완충할 가족자원·공적지원·사회적 도움망이 동시에 약한 다중취약 집단이다.
        이 지점에서 쉬었음 청년 문제는 단순한 미취업 상태가 아니라 위험요소의 누적과 안전망 접근성의 격차 문제로 해석된다.
        """
    )

    policy_box(
        "정책적 시사점",
        """
        정책 대상 선정에서 중요한 것은 쉬었음 여부 하나가 아니라 위험요인의 중첩 정도다.
        위험점수가 높은 청년은 취업지원 이전에 생활비, 부채, 이자, 공적지원 접근성 문제를 동시에 겪고 있을 가능성이 크다.
        따라서 고위험 집단에는 고용서비스만 제공하기보다 생계지원, 금융상담, 공공지원 연결, 사회적 관계망 회복을 묶은 통합형 지원이 필요하다.
        """
    )


# =========================================================
# PAGE 6
# =========================================================

elif selected_page == pages[5]:
    st.header("6. 쉬었음 청년의 생활안전망 유형")

    type_count = filtered["생활안전망유형"].value_counts().reset_index()
    type_count.columns = ["생활안전망유형", "인원수"]

    fig = px.treemap(
        type_count,
        path=["생활안전망유형"],
        values="인원수",
        title="쉬었음 청년 유형별 규모"
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered,
        x=COL_COST,
        y=COL_TOTAL_INCOME,
        size=COL_DEBT,
        color="생활안전망유형",
        hover_data=[
            COL_PRIVATE_INCOME,
            COL_PUBLIC_INCOME,
            COL_DEBT,
            COL_LIVING_DEBT,
            COL_INTEREST,
            "위험점수"
        ],
        title="생활비·소득·부채로 본 생활안전망 유형"
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    type_profile = filtered.groupby("생활안전망유형").agg(
        인원수=("생활안전망유형", "count"),
        평균생활비=(COL_COST, "mean"),
        평균연간소득=(COL_TOTAL_INCOME, "mean"),
        사적이전_보유율=(COL_PRIVATE_INCOME, holding_rate),
        공적이전_보유율=(COL_PUBLIC_INCOME, holding_rate),
        부채_보유율=(COL_DEBT, holding_rate),
        이자_보유율=(COL_INTEREST, holding_rate),
        평균위험점수=("위험점수", "mean")
    ).reset_index()

    st.subheader("유형별 프로파일")
    st.dataframe(type_profile, use_container_width=True)
    st.caption(
        "※ 사적·공적 이전소득, 부채, 이자는 0인 응답자가 대부분이라 평균이 왜곡되기 쉬워 '보유율(0–1)'로 표시했습니다. "
        "생활비·연간소득은 평균(만원)입니다."
    )

    insight_box(
        "핵심 인사이트",
        """
        유형 분류는 쉬었음 청년을 하나의 취약 집단으로 묶어 설명하는 방식의 한계를 보완한다.
        가족완충형은 가족지원과 부모동거를 통해 생활비 위험을 일정 부분 흡수하는 집단이며, 금융부담형은 부채나 이자 부담을 통해 현재의 생활비를 미래의 부담으로 이전하는 집단이다.
        공공지원형은 공적 안전망과 연결되어 있다는 점에서 상대적으로 정책 접점이 존재하지만, 지원 규모가 충분한지는 별도 검토가 필요하다.
        고립위험형은 도움 받을 곳이 없다는 점에서 가장 심각한 사각지대이며, 단순히 소득이나 부채 규모만으로는 포착되지 않는 사회적 취약성을 보여준다.
        """
    )

    policy_box(
        "정책적 시사점",
        """
        쉬었음 청년 정책은 단일 처방으로 설계되기 어렵다.
        가족완충형에는 장기적 노동시장 복귀와 자립 지원이 중요할 수 있고, 금융부담형에는 채무·이자 부담 완화와 생활비 지원이 우선될 수 있다.
        고립위험형에는 소득지원뿐 아니라 공공기관과의 연결, 상담, 긴급지원, 지역사회 기반 지원망 회복이 함께 필요하다.
        따라서 유형별 접근은 제한된 정책 자원을 더 정밀하게 배분하기 위한 근거가 된다.
        """
    )


# =========================================================
# PAGE 7
# =========================================================

else:
    st.header("7. 최종 결론: 한국 청년의 진짜 안전망은 무엇인가?")

    safety = pd.DataFrame({
        "안전망": ["가족", "지인", "공공기관", "민간기관", "도움 없음"],
        "비율": [
            safe_mean(filtered[COL_FAMILY]),
            safe_mean(filtered[COL_FRIEND]),
            safe_mean(filtered[COL_PUBLIC]),
            safe_mean(filtered[COL_PRIVATE]),
            safe_mean(filtered[COL_NONE])
        ]
    })

    fig = px.bar(
        safety,
        x="안전망",
        y="비율",
        text=safety["비율"].apply(lambda x: f"{x:.1%}"),
        title="쉬었음 청년의 생활안전망 구성"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%", height=460)
    st.plotly_chart(fig, use_container_width=True)

    categories = ["가족지원", "지인지원", "공공지원", "민간지원", "비고립"]
    values = [
        safe_mean(filtered[COL_FAMILY]),
        safe_mean(filtered[COL_FRIEND]),
        safe_mean(filtered[COL_PUBLIC]),
        safe_mean(filtered[COL_PRIVATE]),
        1 - safe_mean(filtered[COL_NONE])
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
        title="쉬었음 청년의 안전망 레이더 차트",
        height=560
    )
    st.plotly_chart(fig, use_container_width=True)

    conclusion_box(
        """
        <b>최종 결론</b><br><br>
        본 분석은 ‘쉬었음 청년’을 단순히 일하지 않는 청년, 혹은 동일한 취약 집단으로 바라보는 시각에 한계가 있음을 보여준다.
        쉬었음 상태에 있는 청년들은 모두 노동시장 밖에 있다는 공통점을 가지지만, 실제 생활을 버티는 방식은 크게 다르다.
        일부는 부모와 가족의 지원을 통해 생활비 위험을 완충하고 있었고, 일부는 공적 지원이나 지인·민간기관의 도움과 연결되어 있었다.
        반면 또 다른 일부는 생활비를 부채로 충당하거나 이자 부담을 지고 있었으며, 가장 취약한 집단은 도움을 받을 수 있는 곳 자체가 없는 상태에 놓여 있었다.
        <br><br>
        따라서 이 프로젝트의 핵심 메시지는 ‘쉬었음 청년이 얼마나 힘든가’가 아니라, <b>같은 쉬었음 청년 내부에서도 가족지원망, 공적지원 접근성, 부채·이자 부담, 도움 없음 여부에 따라 생활안전망 격차가 뚜렷하게 갈린다</b>는 것이다.
        특히 가족지원이 약하고 공적지원도 충분히 연결되지 않으며 부채와 고립이 중첩된 청년은 생활안전망 격차의 최전선에 있다.
        결국 쉬었음 청년 문제는 단순한 노동시장 이탈 문제가 아니라, 청년이 노동시장 밖에 머무는 동안 무엇으로 생계를 유지하는지, 그리고 그 안전망이 누구에게는 있고 누구에게는 없는지를 묻는 사회적 문제로 이해해야 한다.
        """
    )

    policy_box(
        "정책 제안 방향",
        """
        쉬었음 청년 지원은 취업 독려 중심의 단일 정책만으로는 충분하지 않다.
        가족완충형, 금융부담형, 공공지원형, 고립위험형처럼 생활안전망 유형을 구분하고 각 집단이 실제로 필요로 하는 지원을 다르게 설계해야 한다.
        특히 고립위험형과 금융부담형에는 생계지원, 부채·이자 부담 완화, 공공지원 연결, 사회적 관계망 회복을 통합한 지원이 필요하다.
        """
    )

    st.markdown(
        """
        <div class="small-note">
        ※ NaN은 Not a Number의 약자로 결측값 또는 계산 불가능한 값을 의미합니다.
        이 앱에서는 결측값을 0으로 처리하고, 필터 결과가 없는 경우에는 시각화 대신 안내문이 뜨도록 수정했습니다.
        </div>
        """,
        unsafe_allow_html=True
    )
