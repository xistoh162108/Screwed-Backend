import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# ===================================================================
# 0. 데이터 준비 (기존과 동일)
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")
file_path = 'Screwed-Backend/data/MERGED_file.csv' # 사용하시는 파일 경로로 수정하세요.
df = pd.read_csv(file_path)

columns_to_drop = ['CODE', 'YEAR', 'LAT', 'LON', 'MAIZE', 'WHEAT', 'RICE']
df = df.drop(columns=columns_to_drop, errors='ignore')

if 'SOYBEAN' in df.columns:
    df = df.rename(columns={'SOYBEAN': 'yield'})
df['yield'] = df['yield'].fillna(0)
df.fillna(df.mean(), inplace=True)

X = df.drop(['yield', 'MONTH'], axis=1)
y = (df['yield'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("--- 0. 데이터 준비 완료 ---\n")

# ===================================================================
# 1. 하이퍼파라미터 튜닝으로 최적의 모델 찾기
# ===================================================================
print("--- 1. 하이퍼파라미터 튜닝 시작 ---")

# 데이터 불균형 처리를 위한 가중치 계산
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 테스트해볼 파라미터 후보들을 정의
param_grid = {
    'max_depth': [3, 5, 7],             # 트리의 최대 깊이
    'learning_rate': [0.01, 0.1],     # 학습률
    'n_estimators': [100, 200],         # 만들 트리의 개수
    'subsample': [0.8, 1.0]             # 각 트리를 훈련할 때 사용할 데이터 샘플 비율
}

# 기본 XGBoost 모델
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)

# GridSearchCV 설정
# cv=5: 데이터를 5개로 나눠 교차 검증
# scoring='f1': 불균형 데이터에서는 'accuracy'보다 'f1-score'가 더 신뢰성 있는 평가지표
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=5, verbose=1)

# 최적의 파라미터를 찾기 위해 학습 시작
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 그때의 성능 출력
print(f"\n✅ 최적의 파라미터: {grid_search.best_params_}")
print(f"✅ 최적 파라미터 사용 시 F1 점수: {grid_search.best_score_:.4f}")

# 가장 성능이 좋았던 모델을 최종 모델로 선택
best_model = grid_search.best_estimator_
print("--- 1. 하이퍼파라미터 튜닝 완료 ---\n")

# ===================================================================
# 2. 최적화된 모델로 성능 평가
# ===================================================================
print("--- 2. 최적화된 모델로 성능 평가 시작 ---")
y_pred = best_model.predict(X_test)
print("--- 분류 모델 정밀 진단 리포트 ---")
print(classification_report(y_test, y_pred, target_names=['No_Harvest (0)', 'Harvest (1)']))
print("--- 2. 모델 성능 평가 완료 ---\n")

# ===================================================================
# 3. 특성 중요도 분석
# ===================================================================
print("--- 3. 특성 중요도 분석 시작 ---")
# 특성 중요도를 DataFrame으로 만들어 보기 쉽게 정렬
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("--- 상위 10개 중요 특성 ---")
print(feature_importances.head(10))

# 특성 중요도 시각화 (그래프로 그리기)
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['feature'][:10], feature_importances['importance'][:10])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Important Features')
plt.gca().invert_yaxis() # 중요도가 높은 것이 맨 위에 오도록
plt.tight_layout()
# 'feature_importance.png' 라는 이름으로 그래프 이미지 파일 저장
plt.savefig('feature_importance.png')
print("\n✅ 'feature_importance.png' 파일로 중요도 그래프가 저장되었습니다.")
print("--- 3. 특성 중요도 분석 완료 ---")