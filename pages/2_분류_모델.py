import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.tree import _tree

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 분류 모델 실습",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 뇌졸중 분류 모델 학습 및 평가 실습실")
st.markdown("선택한 입력 속성을 바탕으로 로지스틱 회귀와 의사결정트리 모델을 훈련하고 성능을 비교합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 수치형 변환
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
# 3. 속성 선택 UI 설정
# -----------------------------------------------------------------------------
st.subheader("⚙️ 모델 입력 속성 선택")

feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}

# 기본 선택: 체질량지수(bmi)를 뺀 넷 (age, avg_glucose_level, hypertension, heart_disease)
default_features = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease']
all_features = list(feature_map.keys())
all_labels = list(feature_map.values())

selected_labels = st.multiselect(
    "학습에 사용할 속성을 두 개 이상 고르세요:",
    options=all_labels,
    default=[feature_map[f] for f in default_features]
)

# 우리말 라벨을 원래 열 이름으로 매핑
label_to_feature = {v: k for k, v in feature_map.items()}
selected_features = [label_to_feature[lbl] for lbl in selected_labels]

# 두 개보다 적게 고르면 안내문 표시 후 중단
if len(selected_features) < 2:
    st.warning("⚠️ 속성을 **두 개 이상** 선택해 주세요. (가로축과 세로축 시각화를 위해 최소 2개가 필요합니다.)")
    st.stop()

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. 데이터 분할 규칙 적용 (10명씩 묶어 앞 3명 테스트, 나머지 7명 훈련)
# -----------------------------------------------------------------------------
# 번호 순으로 정렬 (id 기준 정렬)
df_sorted = df.sort_values('id').reset_index(drop=True)

test_indices = []
train_indices = []

# 10명씩 묶기
for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i:i+10]
    # 각 묶음의 앞 세 명은 테스트용, 나머지(최대 7명)는 훈련용
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3].tolist())
        train_indices.extend(chunk.index[3:].tolist())
    else:
        test_indices.extend(chunk.index.tolist())

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리는 '고른 경우에만' 훈련용의 중앙값으로 채우기
if 'bmi' in selected_features:
    train_bmi_median = df_train['bmi'].median()
    df_train['bmi'] = df_train['bmi'].fillna(train_bmi_median)
    df_test['bmi'] = df_test['bmi'].fillna(train_bmi_median)

# 훈련용 및 테스트용 X, y 세팅
X_train = df_train[selected_features]
y_train = df_train['stroke']
X_test = df_test[selected_features]
y_test = df_test['stroke']

# -----------------------------------------------------------------------------
# 5. 모델 학습 및 평가
# -----------------------------------------------------------------------------
# 1) 로지스틱 회귀 모델
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
train_acc_log = log_reg.score(X_train, y_train)
test_acc_log = log_reg.score(X_test, y_test)

# 2) 의사결정트리 모델 (질문 3번까지, 리프노드 최소 5명, 난수 고정)
dt_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_clf.fit(X_train, y_train)
train_acc_dt = dt_clf.score(X_train, y_train)
test_acc_dt = dt_clf.score(X_test, y_test)

# 3) 베이스라인 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_train_preds = [majority_class] * len(y_train)
baseline_test_preds = [majority_class] * len(y_test)
train_acc_base = (baseline_train_preds == y_train).mean()
test_acc_base = (baseline_test_preds == y_test).mean()

# -----------------------------------------------------------------------------
# 6. 정확도 카드 표시
# -----------------------------------------------------------------------------
st.subheader("📊 모델별 정확도 비교")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 로지스틱 회귀\n(확률로 답하는 모델)")
    st.metric(label="테스트 정확도", value=f"{test_acc_log * 100:.2f}%")
    st.caption(f"• 훈련 데이터 정확도: {train_acc_log * 100:.2f}%\n• 테스트 데이터 정확도: {test_acc_log * 100:.2f}%")

with c2:
    st.markdown("### 의사결정트리\n(질문으로 답하는 모델)")
    st.metric(label="테스트 정확도", value=f"{test_acc_dt * 100:.2f}%")
    st.caption(f"• 훈련 데이터 정확도: {train_acc_dt * 100:.2f}%\n• 테스트 데이터 정확도: {test_acc_dt * 100:.2f}%")

with c3:
    st.markdown("### 다수 클래스 기준\n(입력을 보지 않는 모델)")
    st.metric(label="테스트 정확도", value=f"{test_acc_base * 100:.2f}%")
    st.caption(f"• 훈련 데이터 정확도: {train_acc_base * 100:.2f}%\n• 테스트 데이터 정확도: {test_acc_base * 100:.2f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. 산점도 및 결정 경계(로지스틱 회귀) 시각화
# -----------------------------------------------------------------------------
st.subheader("📈 로지스틱 회귀 모델 결정 경계 산점도")

if len(selected_features) >= 2:
    x_feat = selected_features[0]
    y_feat = selected_features[1]
    x_label = feature_map[x_feat]
    y_label = feature_map[y_feat]

    df_test_plot = df_test.copy()
    df_test_plot['stroke_str'] = df_test_plot['stroke'].apply(lambda x: '뇌졸중' if x == 1 else '정상')

    fig = px.scatter(
        df_test_plot, x=x_feat, y=y_feat, color='stroke_str',
        title=f"<b>테스트 데이터 산점도 및 로지스틱 회귀 경계선</b>",
        labels={x_feat: x_label, y_feat: y_label, 'stroke_str': '실제 뇌졸중 여부'},
        color_discrete_map={'정상': '#1f77b4', '뇌졸중': '#d62728'}
    )

    other_features = [f for f in selected_features if f not in [x_feat, y_feat]]
    medians = {f: df_train[f].median() for f in other_features}
    
    median_desc_list = [f"**{feature_map[f]}** = {medians[f]:.2f}" for f in other_features]
    median_desc_str = ", ".join(median_desc_list) if median_desc_list else "없음"

    x_min, x_max = df_test[x_feat].min() - 1, df_test[x_feat].max() + 1
    y_min, y_max = df_test[y_feat].min() - 1, df_test[y_feat].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    grid_df = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
    grid_df[x_feat] = xx.ravel()
    grid_df[y_feat] = yy.ravel()
    for f in other_features:
        grid_df[f] = medians[f]
    
    grid_df = grid_df[selected_features]
    
    probs = log_reg.predict_proba(grid_df)[:, 1].reshape(xx.shape)
    
    fig.add_trace(go.Contour(
        z=probs,
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        showscale=False,
        contours=dict(coloring='lines', showlabels=True, labelformat='%.1f'),
        line=dict(color='black', width=2, dash='dash'),
        name='확률 0.5 경계선'
    ))

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"ℹ️ **산점도 설명 안내:**\n"
        f"- 가로축: **{x_label}**, 세로축: **{y_label}**\n"
        f"- 두 축으로 선택되지 않은 나머지 속성({median_desc_str})은 테스트 데이터의 중앙값으로 고정하여 경계선을 계산했습니다.\n"
        f"- 검은색 점선은 로지스틱 회귀 모델이 예측 확률 **0.5**로 양성과 음성을 가르는 경계선입니다."
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 8. 의사결정트리 규칙 구조 시각화 (Graphviz DOT 활용)
# -----------------------------------------------------------------------------
st.subheader("🌳 의사결정트리 질문 구조 시각화")

feature_names_ko = [feature_map[f] for f in selected_features]
dot_data = export_graphviz(
    dt_clf,
    out_file=None,
    feature_names=feature_names_ko,
    class_names=['정상', '뇌졸중'],
    rounded=True,
    filled=True,
    special_characters=True,
    precision=2
)

st.graphviz_chart(dot_data)

def get_tree_summary(tree, feature_names):
    tree_ = tree.tree_
    leaf_count = 0
    negative_leaf_count = 0
    used_features = set()

    def recurse(node):
        nonlocal leaf_count, negative_leaf_count
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            used_features.add(name)
            recurse(tree_.children_left[node])
            recurse(tree_.children_right[node])
        else:
            leaf_count += 1
            values = tree_.value[node][0]
            if values[0] > values[1]:
                negative_leaf_count += 1

    recurse(0)
    return leaf_count, negative_leaf_count, list(used_features)

total_leaves, neg_leaves, utilized_features = get_tree_summary(dt_clf, feature_names_ko)

st.markdown(f"""
- 📌 **답을 내는 마디(리프 노드)의 총 개수:** 총 {total_leaves}개
- 📌 **그중 '아님(정상)'이라고 답하는 마디의 개수:** {neg_leaves}개
- 📌 **이 나무가 실제로 질문에 활용한 속성:** {', '.join(utilized_features) if utilized_features else '없음'}
""")
