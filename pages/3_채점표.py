import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 모델 성능 높이기",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ 뇌졸중 예측 모델 성능 높이기 (가중치 조절)")
st.markdown("클래스 불균형 문제를 해결하기 위해 모델에 가중치(class_weight='balanced')를 적용하기 전후의 성능과 결정 경계를 비교합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
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
# 3. 속성 선택 UI 설정 ('분류 모델' 페이지와 동일)
# -----------------------------------------------------------------------------
st.subheader("⚙️ 모델 입력 속성 선택")

feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}

default_features = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease']
all_labels = list(feature_map.values())

selected_labels = st.multiselect(
    "학습에 사용할 속성을 두 개 이상 고르세요:",
    options=all_labels,
    default=[feature_map[f] for f in default_features]
)

label_to_feature = {v: k for k, v in feature_map.items()}
selected_features = [label_to_feature[lbl] for lbl in selected_labels]

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 **두 개 이상** 선택해 주세요. (시각화를 위해 최소 2개가 필요합니다.)")
    st.stop()

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. 데이터 분할 및 결측치 처리 (채점표 / 분류모델과 동일)
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

# 체질량지수(bmi) 결측치는 훈련용의 중앙값으로 채우기
if 'bmi' in selected_features:
    train_bmi_median = df_train['bmi'].median()
    df_train['bmi'] = df_train['bmi'].fillna(train_bmi_median)
    df_test['bmi'] = df_test['bmi'].fillna(train_bmi_median)

X_train = df_train[selected_features]
y_train = df_train['stroke']
X_test = df_test[selected_features]
y_test = df_test['stroke']

# -----------------------------------------------------------------------------
# 5. 모델 학습 (가중치 적용 전후)
# -----------------------------------------------------------------------------
# 1) 로지스틱 회귀 (기본 vs 가중치 균등)
log_reg_before = LogisticRegression(random_state=42)
log_reg_before.fit(X_train, y_train)

log_reg_after = LogisticRegression(class_weight='balanced', random_state=42)
log_reg_after.fit(X_train, y_train)

# 2) 의사결정트리 (기본 vs 가중치 균등)
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42)
dt_after.fit(X_train, y_train)

# 예측 결과
y_pred_log_b = log_reg_before.predict(X_test)
y_pred_log_a = log_reg_after.predict(X_test)
y_pred_dt_b = dt_before.predict(X_test)
y_pred_dt_a = dt_after.predict(X_test)

# -----------------------------------------------------------------------------
# 6. 혼동행렬 및 지표 계산 헬퍼 함수
# -----------------------------------------------------------------------------
def get_metrics_and_cm(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    f1 = (2 * precision * recall) / (precision + recall) if precision is not None and (precision + recall) > 0 else None
    
    return (tn, fp, fn, tp), (accuracy, recall, precision, f1)

cm_lb, m_lb = get_metrics_and_cm(y_test, y_pred_log_b)
cm_la, m_la = get_metrics_and_cm(y_test, y_pred_log_a)
cm_db, m_db = get_metrics_and_cm(y_test, y_pred_dt_b)
cm_da, m_da = get_metrics_and_cm(y_test, y_pred_dt_a)

# -----------------------------------------------------------------------------
# 7. 화면 비교 출력 (전·후 나란히)
# -----------------------------------------------------------------------------
st.subheader("📊 모델별 가중치 적용 전·후 성능 비교")

def render_comparison_section(model_title, cm_b, m_b, cm_a, m_a, found_b, found_a, fn_b, fn_a, fp_b, fp_a):
    st.markdown(f"### {model_title}")
    col1, col2 = st.columns(2)
    
    def display_sub_card(title_sub, cm, m):
        tn, fp, fn, tp = cm
        acc, rec, prec, f1 = m
        prec_str = f"{prec:.4f}" if prec is not None else "계산할 수 없음"
        f1_str = f"{f1:.4f}" if f1 is not None else "계산할 수 없음"
        
        st.markdown(f"**[{title_sub}]**")
        cm_df = pd.DataFrame({
            "구분": ["실제 0", "실제 1", "합계"],
            "예측 0 [TN/FN]": [f"{tn} (TN)", f"{fn} (FN)", f"{tn+fn}"],
            "예측 1 [FP/TP]": [f"{fp} (FP)", f"{tp} (TP)", f"{fp+tp}"],
            "합계": [f"{tn+fp}", f"{fn+tp}", f"{tn+fp+fn+tp}"]
        })
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
        st.markdown(f"- **정확도:** {acc:.4f} | **재현율:** {rec:.4f} | **정밀도:** {prec_str} | **F1:** {f1_str}")

    with col1:
        display_sub_card("가중치 맞추기 전", cm_b, m_b)
    with col2:
        display_sub_card("가중치 맞춘 후 (balanced)", cm_a, m_a)
        
    st.markdown(
        f"📝 **변화 요약:** 찾아낸 환자 수(TP)는 **{found_b}명**에서 **{found_a}명**으로, "
        f"놓친 환자 수(FN)는 **{fn_b}명**에서 **{fn_a}명**으로, "
        f"헛짚은 사람 수(FP)는 **{fp_b}명**에서 **{fp_a}명**으로 바뀌었습니다."
    )
    st.markdown("---")

render_comparison_section(
    "로지스틱 회귀 (확률로 답하는 모델)",
    cm_lb, m_lb, cm_la, m_la, cm_lb[3], cm_la[3], cm_lb[2], cm_la[2], cm_lb[1], cm_la[1]
)

render_comparison_section(
    "의사결정트리 (질문으로 답하는 모델)",
    cm_db, m_db, cm_da, m_da, cm_db[3], cm_da[3], cm_db[2], cm_da[2], cm_db[1], cm_da[1]
)

# -----------------------------------------------------------------------------
# 8. 결정 경계 시각화 (로지스틱 회귀 기준 전후 0.5 경계선 겹치기)
# -----------------------------------------------------------------------------
st.subheader("📈 로지스틱 회귀 모델 가중치 적용 전·후 0.5 경계선 비교")

if len(selected_features) >= 2:
    x_feat = selected_features[0]
    y_feat = selected_features[1]
    x_label = feature_map[x_feat]
    y_label = feature_map[y_feat]
    
    df_test_plot = df_test.copy()
    df_test_plot['stroke_str'] = df_test_plot['stroke'].apply(lambda x: '뇌졸중' if x == 1 else '정상')

    other_features = [f for f in selected_features if f not in [x_feat, y_feat]]
    medians = {f: df_train[f].median() for f in other_features}

    if len(selected_features) == 2:
        # 2차원 산점도 및 0.5 경계선 두 개 겹치기
        fig = px.scatter(
            df_test_plot, x=x_feat, y=y_feat, color='stroke_str',
            title="<b>로지스틱 회귀 0.5 결정 경계선 (전 vs 후)</b>",
            labels={x_feat: x_label, y_feat: y_label, 'stroke_str': '실제 뇌졸중 여부'},
            color_discrete_map={'정상': '#1f77b4', '뇌졸중': '#d62728'}
        )
        
        x_min, x_max = df_test[x_feat].min() - 1, df_test[x_feat].max() + 1
        y_min, y_max = df_test[y_feat].min() - 1, df_test[y_feat].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        grid_df = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
        grid_df[x_feat] = xx.ravel()
        grid_df[y_feat] = yy.ravel()
        for f in other_features:
            grid_df[f] = medians[f]
        grid_df = grid_df[selected_features]
        
        probs_b = log_reg_before.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        probs_a = log_reg_after.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        
        # 가중치 전 경계선 (파란색 계열 점선)
        fig.add_trace(go.Contour(
            z=probs_b, x=np.linspace(x_min, x_max, 200), y=np.linspace(y_min, y_max, 200),
            showscale=False, contours=dict(coloring='lines', showlabels=True, values=[0.5]),
            line=dict(color='blue', width=2, dash='dash'), name='가중치 전 (0.5)'
        ))
        
        # 가중치 후 경계선 (주황색 계열 실선)
        fig.add_trace(go.Contour(
            z=probs_a, x=np.linspace(x_min, x_max, 200), y=np.linspace(y_min, y_max, 200),
            showscale=False, contours=dict(coloring='lines', showlabels=True, values=[0.5]),
            line=dict(color='orange', width=2), name='가중치 후 (0.5)'
        ))
        
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ 파란색 점선은 가중치 적용 전의 0.5 경계선이며, 주황색 실선은 가중치(balanced) 적용 후의 0.5 경계선입니다.")

    else:
        # 속성이 셋 이상인 경우 첫 세 개를 축(X, Y, Z)으로 하는 3차원 산점도 및 0.5 평면 겹치기
        z_feat = selected_features[2]
        z_label = feature_map[z_feat]
        
        st.markdown(f"📌 속성이 3개 이상 선택되어 **{x_label}, {y_label}, {z_label}** 3개 축을 기준으로 3차원 산점도 및 결정 평면을 표시합니다. (마우스로 회전 가능)")
        
        fig = px.scatter_3d(
            df_test_plot, x=x_feat, y=y_feat, z=z_feat, color='stroke_str',
            title="<b>3차원 로지스틱 회귀 0.5 결정 평면 비교</b>",
            labels={x_feat: x_label, y_feat: y_label, z_feat: z_label, 'stroke_str': '실제 뇌졸중 여부'},
            color_discrete_map={'정상': '#1f77b4', '뇌졸중': '#d62728'}
        )
        
        # 3차원 격자 생성
        x_min, x_max = df_test[x_feat].min(), df_test[x_feat].max()
        y_min, y_max = df_test[y_feat].min(), df_test[y_feat].max()
        z_min, z_max = df_test[z_feat].min(), df_test[z_feat].max()
        
        gx, gy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
        
        # 나머지 속성 중앙값 처리용 3차원 평면 격자 계산
        remaining_feats = [f for f in selected_features if f not in [x_feat, y_feat, z_feat]]
        rem_medians = {f: df_train[f].median() for f in remaining_feats}
        
        # 로지스틱 회귀 공식 z = -(coef_x * x + coef_y * y + intercept) / coef_z 기반 평면 표면 추가 시도
        # 계수 추출을 통한 3차원 평면 Z값 산출
        coefs = log_reg_before.coef_[0]
        intercept = log_reg_before.intercept_[0]
        
        # 매핑 딕셔너리 인덱스 확인
        feat_idx = {f: idx for idx, f in enumerate(selected_features)}
        cx, cy, cz = coefs[feat_idx[x_feat]], coefs[feat_idx[y_feat]], coefs[feat_idx[z_feat]]
        
        if cz != 0:
            # 가중치 전 평면 Z값 계산
            other_sum_b = sum(coefs[feat_idx[f]] * rem_medians[f] for f in remaining_feats) if remaining_feats else 0
            gz_b = -(cx * gx + gy * cy + other_sum_b + intercept) / cz
            
            fig.add_trace(go.Surface(
                x=gx, y=gy, z=gz_b, colorscale=[[0, 'blue'], [1, 'blue']],
                opacity=0.3, showscale=False, name='가중치 전 0.5 평면'
            ))
            
        coefs_a = log_reg_after.coef_[0]
        intercept_a = log_reg_after.intercept_[0]
        cxa, cya, cza = coefs_a[feat_idx[x_feat]], coefs_a[feat_idx[y_feat]], coefs_a[feat_idx[z_feat]]
        
        if cza != 0:
            other_sum_a = sum(coefs_a[feat_idx[f]] * rem_medians[f] for f in remaining_feats) if remaining_feats else 0
            gz_a = -(cxa * gx + gy * cya + other_sum_a + intercept_a) / cza
            
            fig.add_trace(go.Surface(
                x=gx, y=gy, z=gz_a, colorscale=[[0, 'orange'], [1, 'orange']],
                opacity=0.3, showscale=False, name='가중치 후 0.5 평면'
            ))
            
        fig.update_layout(template="plotly_white", scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label))
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ 3차원 공간 상에 파란색 평면(가중치 전)과 주황색 평면(가중치 후)이 겹쳐서 표시됩니다. 마우스로 드래그하여 돌려볼 수 있습니다.")
