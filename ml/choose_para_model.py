import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score

# ===================================================================
# 0. 데이터 준비
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")
# 사용하시는 파일 경로로 수정하세요.
file_path = 'Screwed-Backend/data/MERGED_file.csv' 
df = pd.read_csv(file_path)

columns_to_drop = ['CODE', 'YEAR', 'LAT', 'LON', 'MAIZE', 'WHEAT', 'RICE']
df = df.drop(columns=columns_to_drop, errors='ignore')

if 'SOYBEAN' in df.columns:
    df = df.rename(columns={'SOYBEAN': 'yield'})
df['yield'] = df['yield'].fillna(0)
df.fillna(df.mean(), inplace=True)

num_features_lstm = df.shape[1] 

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 모델 1: 수확 여부 예측 (XGBoost 분류 모델)
# ===================================================================
print("--- 1. XGBoost 분류 모델 학습 시작 ---")
X_clf = df.drop(['yield', 'MONTH'], axis=1)
y_clf = (df['yield'] > 0).astype(int)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# --- 1-1. 특성 중요도 분석 및 선택 ---
print("--- 1-1. 특성 중요도 분석 및 상위 10개 선택 ---")
# 가중치 계산
scale_pos_weight = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()

# 초기 모델을 학습하여 특성 중요도 계산
initial_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, eval_metric='logloss')
initial_model.fit(X_clf_train, y_clf_train)

# 중요도를 DataFrame으로 만들어 정렬
feature_importances = pd.DataFrame({
    'feature': X_clf_train.columns,
    'importance': initial_model.feature_importances_
}).sort_values('importance', ascending=False)

# 상위 10개 특성 이름 추출
top_10_features = feature_importances.head(10)['feature'].tolist()
print("✅ 선택된 상위 10개 특성:")
print(top_10_features)

# 상위 10개 특성만 가진 새로운 데이터셋 생성
X_train_top10 = X_clf_train[top_10_features]
X_test_top10 = X_clf_test[top_10_features]


# --- 1-2. 하이퍼파라미터 튜닝 (선택된 특성 사용) ---
print("\n--- 1-2. 하이퍼파라미터 튜닝 시작 ---")
param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.1], 'n_estimators': [100, 200]}
grid_search = GridSearchCV(estimator=initial_model, param_grid=param_grid, scoring='f1', cv=3, verbose=0)
# 상위 10개 특성으로 튜닝 진행
grid_search.fit(X_train_top10, y_clf_train)

best_classifier = grid_search.best_estimator_
print(f"✅ 최적의 파라미터: {grid_search.best_params_}")


# --- 1-3. 최종 분류 모델 성능 평가 ---
y_pred_clf = best_classifier.predict(X_test_top10)
print("\n--- [상위 10개 특성 사용 시] 분류 모델 정밀 진단 리포트 ---")
print(classification_report(y_clf_test, y_pred_clf, target_names=['No_Harvest (0)', 'Harvest (1)']))
print("--- 1. XGBoost 분류 모델 학습 완료 ---\n")


# ===================================================================
# 2. 모델 2: 생산량 예측 (회귀 LSTM 모델)
# ===================================================================
print("--- 2. 회귀 LSTM 모델 학습 시작 ---")
harvest_only_df = scaled_df[df['yield'].values > 0]
X_reg, y_reg = [], []
for i in range(len(harvest_only_df) - 1):
    X_reg.append(harvest_only_df.iloc[i].values)
    y_reg.append(harvest_only_df.iloc[i + 1].values)

if len(X_reg) > 0:
    X_reg = np.array(X_reg).reshape(-1, 1, num_features_lstm)
    y_reg = np.array(y_reg)
    X_reg_tensor = torch.from_numpy(X_reg).float()
    y_reg_tensor = torch.from_numpy(y_reg).float()

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    regressor = LSTMModel(input_size=num_features_lstm, hidden_size=50, output_size=num_features_lstm)
    criterion_reg = nn.MSELoss()
    optimizer_reg = torch.optim.Adam(regressor.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        outputs = regressor(X_reg_tensor)
        loss = criterion_reg(outputs, y_reg_tensor)
        optimizer_reg.zero_grad()
        loss.backward()
        optimizer_reg.step()
    print(f"LSTM 최종 Loss: {loss.item():.4f}")
else:
    regressor = None
    print("경고: LSTM 모델을 학습할 수확 데이터가 부족합니다.")
print("--- 2. 회귀 LSTM 모델 학습 완료 ---\n")


# ===================================================================
# 3. 통합 예측 테스트 (강화 버전)
# ===================================================================
print("--- 3. 통합 예측 테스트 시작 ---")
results_list = []

if regressor is not None:
    regressor.eval()

with torch.no_grad():
    for i in range(len(X_clf_test)):
        # 분류 예측에는 상위 10개 특성만 사용
        input_clf_df_top10 = X_test_top10.iloc[[i]]
        
        predicted_harvest = best_classifier.predict(input_clf_df_top10)[0]
        harvest_probability = best_classifier.predict_proba(input_clf_df_top10)[0][1]

        predicted_yield = 0.0
        if predicted_harvest == 1 and regressor is not None:
            original_data_index = input_clf_df_top10.index[0]
            input_reg_np = scaled_df.iloc[original_data_index].values
            input_reg = torch.from_numpy(input_reg_np).float()
            next_state_scaled = regressor(input_reg.reshape(1, 1, num_features_lstm))
            next_state_original = scaler.inverse_transform(next_state_scaled.numpy())
            yield_col_index = df.columns.get_loc('yield')
            predicted_yield = next_state_original[0, yield_col_index]

        original_data_index = input_clf_df_top10.index[0]
        original_row = df.iloc[original_data_index]
        month = original_row['MONTH']
        actual_yield = original_row['yield']
        actual_harvest = y_clf_test.iloc[i]
        
        results_list.append({
            'Month': int(month),
            'Actual_Harvest': actual_harvest,
            'Predicted_Harvest': predicted_harvest,
            'Harvest_Probability(%)': harvest_probability * 100,
            'Actual_Yield': actual_yield,
            'Predicted_Yield': predicted_yield,
        })

results_df = pd.DataFrame(results_list)
print("\n--- 📝 최종 예측 결과 상세표 (상위 20개) ---")
pd.options.display.float_format = '{:.2f}'.format
print(results_df.head(20).to_string())

print("\n\n" + "="*50)
print("--- 📊 최종 성능 요약 ---")
print("="*50)
print("\n[분류 모델 성능 요약]")
print(classification_report(results_df['Actual_Harvest'], results_df['Predicted_Harvest'], target_names=['No_Harvest (0)', 'Harvest (1)']))
harvest_samples_df = results_df[results_df['Actual_Harvest'] == 1]
if not harvest_samples_df.empty:
    mae = mean_absolute_error(harvest_samples_df['Actual_Yield'], harvest_samples_df['Predicted_Yield'])
    r2 = r2_score(harvest_samples_df['Actual_Yield'], harvest_samples_df['Predicted_Yield'])
    print("\n[회귀 모델 성능 요약 (실제 수확이 있었던 경우)]")
    print(f"평균 절대 오차 (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
else:
    print("\n[회귀 모델 성능 요약]")
    print("평가할 수확 샘플이 테스트 세트에 없습니다.")
print("\n--- 3. 통합 예측 테스트 완료 ---")