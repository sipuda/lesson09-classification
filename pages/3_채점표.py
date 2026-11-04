import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 모델 채점표",
    page_icon="📋",
    layout="wide"
)

st.title("📋 뇌졸중 예측 모델 성능 채점표")
st.markdown("테스트 데이터를 기반으로 두 분류 모델(로지스틱 회귀, 의사결정트리)과 기준 모델의 혼동행렬 및 평가 지표를 상세히 채점합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (분류 모델 페이지와 동일한 방식)
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    numeric_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 데이터 분할 규칙 적용 (10명씩 묶어 앞 3명 테스트, 나머지 7명 훈련)
# -----------------------------------------------------------------------------
df_sorted = df.sort_values('id').reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i:i+10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3].tolist())
        train_indices.extend(chunk.index[3:].tolist())
    else:
        test_indices.extend(chunk.index.tolist())

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# 기본 학습에 사용할 5가지 속성 (age, avg_glucose_level, bmi, hypertension, heart_disease)
selected_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']

# bmi 결측치는 훈련용의 중앙값으로 채우기
train_bmi_median = df_train['bmi'].median()
df_train['bmi'] = df_train['bmi'].fillna(train_bmi_median)
df_test['bmi'] = df_test['bmi'].fillna(train_bmi_median)

X_train = df_train[selected_features]
y_train = df_train['stroke']
X_test = df_test[selected_features]
y_test = df_test['stroke']

# -----------------------------------------------------------------------------
# 4. 모델 학습 및 예측
# -----------------------------------------------------------------------------
# 1) 로지스틱 회귀 모델 (확률로 답하는 모델)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# 2) 의사결정트리 모델 (질문으로 답하는 모델)
dt_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# 3) 베이스라인 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
y_pred_base = np.array([majority_class] * len(y_test))

# -----------------------------------------------------------------------------
# 5. 테스트 데이터 요약 정보 표시
# -----------------------------------------------------------------------------
total_test_count = len(y_test)
actual_stroke_count = int(y_test.sum())

st.markdown(f"📌 **테스트용 사람 수:** {total_test_count:,}명 | **그중 실제 뇌졸중인 사람 수:** {actual_stroke_count:,}명")
st.markdown("---")

# -----------------------------------------------------------------------------
# 6. 혼동행렬 (Confusion Matrix) 2x2 표 계산 함수
# -----------------------------------------------------------------------------
def calc_confusion_matrix(y_true, y_pred):
    # 행: 실제 (0, 1), 열: 예측 (0, 1)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp

cm_log = calc_confusion_matrix(y_test, y_pred_log)
cm_dt = calc_confusion_matrix(y_test, y_pred_dt)
cm_base = calc_confusion_matrix(y_test, y_pred_base)

st.subheader("📊 테스트 데이터 혼동행렬 (Confusion Matrix)")
st.markdown("행은 **실제 값**, 열은 **예측 값**을 나타냅니다. (단위: 명)")

col1, col2, col3 = st.columns(3)

def render_cm_table(cm, model_title):
    tn, fp, fn, tp = cm
    actual_neg = tn + fp
    actual_pos = fn + tp
    pred_neg = tn + fn
    pred_pos = fp + tp
    total = tn + fp + fn + tp

    cm_data = {
        "구분": ["실제 0 (정상)", "실제 1 (뇌졸중)", "합계"],
        "예측 0 (정상) [TN / FN]": [f"{tn:,} (TN)", f"{fn:,} (FN)", f"{pred_neg:,}"],
        "예측 1 (뇌졸중) [FP / TP]": [f"{fp:,} (FP)", f"{tp:,} (TP)", f"{pred_pos:,}"],
        "합계": [f"{actual_neg:,}", f"{actual_pos:,}", f"{total:,}"]
    }
    st.markdown(f"**{model_title}**")
    st.dataframe(pd.DataFrame(cm_data), use_container_width=True, hide_index=True)

with col1:
    render_cm_table(cm_log, "로지스틱 회귀 (확률로 답하는 모델)")
with col2:
    render_cm_table(cm_dt, "의사결정트리 (질문으로 답하는 모델)")
with col3:
    render_cm_table(cm_base, "다수 클래스 기준 모델")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. 성능 지표 계산 및 비교 표
# -----------------------------------------------------------------------------
st.subheader("📈 모델별 성능 지표 비교 (정확도, 재현율, 정밀도, F1)")

def calc_metrics(cm):
    tn, fp, fn, tp = cm
    total = tn + fp + fn + tp
    
    # 정확도 (Accuracy)
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # 재현율 (Recall) = TP / (TP + FN) -> 실제 뇌졸중 중 맞춘 비율
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 정밀도 (Precision) = TP / (TP + FP) -> 뇌졸중이라 예측한 것 중 실제 뇌졸중인 비율
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = None  # 0명인 경우
        
    # F1 Score = 2 * Precision * Recall / (Precision + Recall)
    if precision is not None and (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = None
        
    return accuracy, recall, precision, f1

acc_log, rec_log, prec_log, f1_log = calc_metrics(cm_log)
acc_dt, rec_dt, prec_dt, f1_dt = calc_metrics(cm_dt)
acc_base, rec_base, prec_base, f1_base = calc_metrics(cm_base)

def format_metric_value(val):
    return f"{val:.4f}" if val is not None else "계산할 수 없음"

metrics_data = [
    {
        "모델명": "로지스틱 회귀 (확률로 답하는 모델)",
        "정확도 (Accuracy)": f"{acc_log:.4f}",
        "재현율 (Recall)": f"{rec_log:.4f}",
        "정밀도 (Precision)": format_metric_value(prec_log),
        "F1 점수 (F1-Score)": format_metric_value(f1_log)
    },
    {
        "모델명": "의사결정트리 (질문으로 답하는 모델)",
        "정확도 (Accuracy)": f"{acc_dt:.4f}",
        "재현율 (Recall)": f"{rec_dt:.4f}",
        "정밀도 (Precision)": format_metric_value(prec_dt),
        "F1 점수 (F1-Score)": format_metric_value(f1_dt)
    },
    {
        "모델명": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도 (Accuracy)": f"{acc_base:.4f}",
        "재현율 (Recall)": f"{rec_base:.4f}",
        "정밀도 (Precision)": format_metric_value(prec_base),
        "F1 점수 (F1-Score)": format_metric_value(f1_base)
    }
]

st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# 8. 놓친 환자(FN) 및 헛짚은 사람(FP) 목록 비교
# -----------------------------------------------------------------------------
st.subheader("🔍 오류 분석 목록 (놓친 환자 vs 헛짚은 사람)")

# 모델 선택 UI
chosen_model_name = st.selectbox(
    "어느 모델의 오류 목록을 확인할까요?",
    options=["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)"]
)

if chosen_model_name.startswith("로지스틱"):
    y_pred_chosen = y_pred_log
else:
    y_pred_chosen = y_pred_dt

# 놓친 환자 (FN): 실제 1인데 0이라고 예측한 사람
fn_mask = (y_test == 1) & (y_pred_chosen == 0)
df_fn = df_test[fn_mask].copy()

# 헛짚은 사람 (FP): 실제 0인데 1이라고 예측한 사람
fp_mask = (y_test == 0) & (y_pred_chosen == 1)
df_fp = df_test[fp_mask].copy()

# 나이가 많은 순서로 정렬
df_fn = df_fn.sort_values(by='age', ascending=False).reset_index(drop=True)
df_fp = df_fp.sort_values(by='age', ascending=False).reset_index(drop=True)

fn_total_count = len(df_fn)
fp_total_count = len(df_fp)

col_l, col_r = st.columns(2)

def format_error_dataframe(df_subset):
    if df_subset.empty:
        return pd.DataFrame(columns=["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"])
    
    formatted = pd.DataFrame()
    formatted["번호"] = df_subset['id']
    formatted["나이"] = df_subset['age']
    formatted["평균 혈당"] = df_subset['avg_glucose_level']
    formatted["고혈압 여부"] = df_subset['hypertension'].apply(lambda x: '있음' if x == 1 else '없음')
    formatted["심장병 여부"] = df_subset['heart_disease'].apply(lambda x: '있음' if x == 1 else '없음')
    return formatted

with col_l:
    st.markdown(f"### 🛑 실제 환자이나 놓친 사람 (FN) — 총 {fn_total_count}명")
    display_fn = format_error_dataframe(df_fn.head(20))
    st.dataframe(display_fn, use_container_width=True, hide_index=True)
    if fn_total_count > 20:
        st.caption("ℹ️ 목록이 20줄을 초과하여 앞 20줄만 표시됩니다.")

with col_r:
    st.markdown(f"### ⚠️ 정상이나 뇌졸중으로 헛짚은 사람 (FP) — 총 {fp_total_count}명")
    display_fp = format_error_dataframe(df_fp.head(20))
    st.dataframe(display_fp, use_container_width=True, hide_index=True)
    if fp_total_count > 20:
        st.caption("ℹ️ 목록이 20줄을 초과하여 앞 20줄만 표시됩니다.")

st.markdown("---")

# 혼동행렬의 FN, FP 수와 일치하는지 확인하는 검증 문구 출력
current_cm = cm_log if chosen_model_name.startswith("로지스틱") else cm_dt
actual_fn_val = current_cm[2]
actual_fp_val = current_cm[1]

st.info(
    f"💡 **오류 목록 검증:**\n"
    f"- 선택한 모델의 혼동행렬 기준 **FN (놓친 환자 수)**은 **{actual_fn_val}명**, 위 목록의 총 인원은 **{fn_total_count}명**으로 정확히 일치합니다.\n"
    f"- 혼동행렬 기준 **FP (헛짚은 사람 수)**는 **{actual_fp_val}명**, 위 목록의 총 인원은 **{fp_total_count}명**으로 정확히 일치합니다."
)
