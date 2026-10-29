import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 데이터 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 뇌졸중 데이터 탐색 및 시각화")
st.markdown("뇌졸중 예측 데이터를 다양한 조건과 변수별로 탐색하고 시각화하는 페이지입니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 수치형 데이터 안전 변환
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'avg_glucose_level' in df.columns:
        df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 나이와 평균 혈당 분포 (히스토그램 두 개 나란히)
# -----------------------------------------------------------------------------
st.header("📊 1. 나이와 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x='age', nbins=30,
        title="<b>나이(age) 분포</b>",
        labels={'age': '나이', 'count': '사람 수(명)'},
        color_discrete_sequence=['#3366CC']
    )
    fig_age.update_layout(template="plotly_white")
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x='avg_glucose_level', nbins=30,
        title="<b>평균 혈당(avg_glucose_level) 분포</b>",
        labels={'avg_glucose_level': '평균 혈당', 'count': '사람 수(명)'},
        color_discrete_sequence=['#FF9900']
    )
    fig_glucose.update_layout(template="plotly_white")
    st.plotly_chart(fig_glucose, use_container_width=True)

st.info("💡 **이 그래프로 알 수 있는 것:** 데이터셋에 포함된 사람들의 연령대 분포와 평균 혈당 수치의 전반적인 쏠림 현상을 파악할 수 있습니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 4. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교 (박스플롯 + 평균값 표)
# -----------------------------------------------------------------------------
st.header("📦 2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")

# stroke 값을 보기 쉽게 문자열로 변환 (0 -> 정상, 1 -> 뇌졸중)
df['stroke_label'] = df['stroke'].apply(lambda x: '뇌졸중' if x == 1 else '정상')

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df, x='stroke_label', y='age', color='stroke_label',
        title="<b>뇌졸중 여부별 나이 비교 (박스플롯)</b>",
        labels={'stroke_label': '뇌졸중 여부', 'age': '나이'},
        color_discrete_map={'정상': '#1f77b4', '뇌졸중': '#d62728'}
    )
    fig_box_age.update_layout(showlegend=False, template="plotly_white")
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df, x='stroke_label', y='avg_glucose_level', color='stroke_label',
        title="<b>뇌졸중 여부별 평균 혈당 비교 (박스플롯)</b>",
        labels={'stroke_label': '뇌졸중 여부', 'avg_glucose_level': '평균 혈당'},
        color_discrete_map={'정상': '#1f77b4', '뇌졸중': '#d62728'}
    )
    fig_box_glucose.update_layout(showlegend=False, template="plotly_white")
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 그룹별 평균값 표 계산
mean_summary = df.groupby('stroke_label')[['age', 'avg_glucose_level']].mean().reset_index()
mean_summary.columns = ['뇌졸중 여부', '평균 나이 (세)', '평균 혈당']
mean_summary['평균 나이 (세)'] = mean_summary['평균 나이 (세)'].round(1)
mean_summary['평균 혈당'] = mean_summary['평균 혈당'].round(2)

st.subheader("📋 뇌졸중 여부에 따른 그룹별 평균값 비교")
st.dataframe(mean_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 고혈압 / 심장병 여부에 따른 뇌졸중 비율 (막대그래프)
# -----------------------------------------------------------------------------
st.header("📈 3. 질환(고혈압, 심장병)에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    # 고혈압(hypertension)별 뇌졸중 비율 계산
    hyp_df = df.groupby('hypertension')['stroke'].mean().reset_index()
    hyp_df['stroke_ratio'] = hyp_df['stroke'] * 100
    hyp_df['hypertension_label'] = hyp_df['hypertension'].apply(lambda x: '고혈압 있음 (1)' if x == 1 else '고혈압 없음 (0)')
    
    fig_hyp = px.bar(
        hyp_df, x='hypertension_label', y='stroke_ratio', color='hypertension_label',
        title="<b>고혈압 여부에 따른 뇌졸중 비율</b>",
        labels={'hypertension_label': '고혈압 여부', 'stroke_ratio': '뇌졸중 비율 (%)'},
        text_auto='.2f'
    )
    fig_hyp.update_layout(showlegend=False, yaxis_title="뇌졸중 비율 (%)", template="plotly_white")
    st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
    # 심장병(heart_disease)별 뇌졸중 비율 계산
    heart_df = df.groupby('heart_disease')['stroke'].mean().reset_index()
    heart_df['stroke_ratio'] = heart_df['stroke'] * 100
    heart_df['heart_label'] = heart_df['heart_disease'].apply(lambda x: '심장병 있음 (1)' if x == 1 else '심장병 없음 (0)')
    
    fig_heart = px.bar(
        heart_df, x='heart_label', y='stroke_ratio', color='heart_label',
        title="<b>심장병 여부에 따른 뇌졸중 비율</b>",
        labels={'heart_label': '심장병 여부', 'stroke_ratio': '뇌졸중 비율 (%)'},
        text_auto='.2f'
    )
    fig_heart.update_layout(showlegend=False, yaxis_title="뇌졸중 비율 (%)", template="plotly_white")
    st.plotly_chart(fig_heart, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. BMI 결측치 인원의 뇌졸중 비율 vs 전체 뇌졸중 비율 (표 비교)
# -----------------------------------------------------------------------------
st.header("📉 4. BMI 비어 있는 사람들의 뇌졸중 비율 비교")

total_count = len(df)
total_stroke_ratio = (df['stroke'].sum() / total_count) * 100 if total_count > 0 else 0

# BMI가 비어 있는(NaN) 데이터 필터링
bmi_null_df = df[df['bmi'].isnull()]
null_count = len(bmi_null_df)
null_stroke_ratio = (bmi_null_df['stroke'].sum() / null_count) * 100 if null_count > 0 else 0

comparison_data = [
    {
        "구분": "전체 데이터",
        "인원수 (명)": total_count,
        "뇌졸중 환자수 (명)": int(df['stroke'].sum()),
        "뇌졸중 비율 (%)": round(total_stroke_ratio, 2)
    },
    {
        "구분": "BMI 비어 있는 사람들",
        "인원수 (명)": null_count,
        "뇌졸중 환자수 (명)": int(bmi_null_df['stroke'].sum()),
        "뇌졸중 비율 (%)": round(null_stroke_ratio, 2)
    }
]

comparison_table = pd.DataFrame(comparison_data)
st.dataframe(comparison_table, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. 흡연 상태(smoking_status)별 사람 수 (표)
# -----------------------------------------------------------------------------
st.header("🚬 5. 흡연 상태별 사람 수")

# 흡연 상태 값별 인원수 및 뇌졸중 환자수 집계
smoking_summary = df.groupby('smoking_status').agg(
    인원수=('id', 'count'),
    뇌졸중환자수=('stroke', 'sum')
).reset_index()

smoking_summary['비율(%)'] = (smoking_summary['인원수'] / total_count) * 100
smoking_summary['비율(%)'] = smoking_summary['비율(%)'].round(2)
smoking_summary.columns = ['흡연 상태 (smoking_status)', '인원수 (명)', '뇌졸중 환자수 (명)', '전체 대비 비율 (%)']

st.dataframe(smoking_summary, use_container_width=True, hide_index=True)
