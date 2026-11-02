import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 브라우저 탭 제목 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.markdown("뇌졸중 데이터셋의 구성과 주요 특성을 탐색하고 분석할 수 있는 실습 공간입니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 내 기록 (JSON) 저장 및 불러오기 관리
# -----------------------------------------------------------------------------
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "my_notes.json")

def load_user_records():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"data_source": "", "column_meanings": {}}

def save_user_records(records):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

# 저장된 사용자 데이터 불러오기
user_records = load_user_records()

# -----------------------------------------------------------------------------
# 3. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. 주요 수치 카드 (Metrics)
# -----------------------------------------------------------------------------
total_people = len(df)
total_cols = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / total_people) * 100 if total_people > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="👥 전체 사람 수", value=f"{total_people:,} 명")
with m2:
    st.metric(label="📋 열 개수", value=f"{total_cols} 개")
with m3:
    st.metric(label="⚠️ 뇌졸중 환자 수 (stroke=1)", value=f"{stroke_count:,} 명")
with m4:
    st.metric(label="📊 뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 열 설명 테이블 및 우리말 뜻 입력창
# -----------------------------------------------------------------------------
st.subheader("📋 데이터 열(Column) 상세 메타데이터")
st.markdown("각 열의 이름, 값의 종류(데이터 타입), 결측치 개수를 확인하고 **우리말 뜻**을 직접 교재를 참고해 채워 넣어 보세요.")

# 열 정보 수집
columns_info = []
saved_meanings = user_records.get("column_meanings", {})

for col in df.columns:
    null_count = int(df[col].isnull().sum())
    dtype_str = str(df[col].dtype)
    columns_info.append({
        "열 이름": col,
        "우리말 뜻": saved_meanings.get(col, ""),
        "값의 종류": dtype_str,
        "빈 값 개수": null_count
    })

info_df = pd.DataFrame(columns_info)

# 데이터 에디터로 사용자가 직접 '우리말 뜻' 편집 가능
edited_df = st.data_editor(
    info_df,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", disabled=True),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (직접 입력 가능)", help="교재를 보고 직접 채워 넣으세요."),
        "값의 종류": st.column_config.TextColumn("값의 종류", disabled=True),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", disabled=True)
    },
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. 데이터 미리보기 (상위 5줄)
# -----------------------------------------------------------------------------
st.subheader("🔍 데이터 미리보기 (상위 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. 데이터 출처 작성 구역 및 기록 저장 버튼
# -----------------------------------------------------------------------------
st.subheader("📝 데이터 출처 및 메모 작성")

data_source_text = st.text_area(
    "데이터 출처를 입력하세요 (교재 참고):",
    value=user_records.get("data_source", ""),
    height=100,
    placeholder="예: Kaggle Stroke Prediction Dataset..."
)

if st.button("💾 입력한 정보 저장하기", type="primary"):
    # 수정된 우리말 뜻 추출
    updated_meanings = {}
    for idx, row in edited_df.iterrows():
        updated_meanings[row["열 이름"]] = row["우리말 뜻"] if row["우리말 뜻"] is not None else ""
    
    # JSON 파일 업데이트
    new_records = {
        "data_source": data_source_text,
        "column_meanings": updated_meanings
    }
    
    save_user_records(new_records)
    st.success("✅ '우리말 뜻'과 '데이터 출처'가 [내_기록/my_notes.json]에 성공적으로 저장되었습니다! 새로고침해도 그대로 유지됩니다.")
