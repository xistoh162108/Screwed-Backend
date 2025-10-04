import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score

# ===================================================================
# 0. 데이터 준비
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

# LSTM 모델의 input_size를 위해 MONTH를 제외하기 전의 num_features를 저장
num_features_lstm = df.shape[1] 

# 정규화된 데이터 생성 (LSTM 학습 및 예측에 사용)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 모델 1: 수확 여부 예측 (XGBoost 분류 모델)
# ===================================================================
print("--- 1. XGBoost 분류 모델 학습 시작 ---")
# 입력(X): 'yield'와 'MONTH'를 제외한 모든 변수
X_clf = df.drop(['yield', 'MONTH'], axis=1)
# 정답(y): 'yield'가 0보다 크면 1, 아니면 0
y_clf = (df['yield'] > 0).astype(int)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# 데이터 불균형 처리를 위한 가중치 계산
scale_pos_weight = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()

# XGBoost 모델 및 GridSearchCV 설정
param_grid = {'max_depth': [3, 5], 'learning_rate': [0.1], 'n_estimators': [100]}
xgb_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=3, verbose=0)
grid_search.fit(X_clf_train, y_clf_train)

best_classifier = grid_search.best_estimator_
print(f"✅ 최적의 파라미터: {grid_search.best_params_}")

# ===================================================================
# ✅ 1-1. 최적화된 분류 모델 성능 평가 (추가된 부분)
# ===================================================================
# 테스트 데이터로 예측 수행
y_pred = best_classifier.predict(X_clf_test)

print("\n--- 분류 모델 정밀 진단 리포트 ---")
# y_true -> y_clf_test, y_pred -> y_pred 로 변수명을 맞춰줍니다.
print(classification_report(y_clf_test, y_pred, target_names=['No_Harvest (0)', 'Harvest (1)']))

print("--- 혼동 행렬 (Confusion Matrix) ---")
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_clf_test, y_pred)
print(cm)
print("해석: Harvest(1) 클래스의 recall 점수가 모델이 실제 수확하는 달을 얼마나 잘 맞추는지를 나타냅니다.")
# ===================================================================

print("--- 1. XGBoost 분류 모델 학습 완료 ---\n")

# ===================================================================
# 2. 모델 2: 생산량 예측 (회귀 LSTM 모델)
# ===================================================================
print("--- 2. 회귀 LSTM 모델 학습 시작 ---")
# 생산량이 0보다 큰 데이터만 필터링 (정규화된 데이터 사용)
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

new_threshold = 0.65
print(f"✅ 수확 판단 기준(Threshold)이 {new_threshold*100:.0f}%로 설정되었습니다.\n")

if regressor is not None:
    regressor.eval()

# torch.no_grad() 블록은 PyTorch 모델을 사용할 때만 필요
with torch.no_grad():
    # 전체 테스트 세트(X_clf_test)에 대해 반복
    for i in range(len(X_clf_test)):
        
        input_clf_df = X_clf_test.iloc[[i]]
        
        # --- 1단계: XGBoost로 수확 여부 예측 ---
        # predict_proba()는 [수확 안할 확률, 수확할 확률]을 반환
        harvest_probability = best_classifier.predict_proba(input_clf_df)[0][1]
        predicted_harvest = 1 if harvest_probability >= new_threshold else 0

        # --- 2단계: '수확함(1)'으로 예측될 때만 LSTM으로 생산량 예측 ---
        predicted_yield = 0.0
        if predicted_harvest == 1 and regressor is not None:
            original_data_index = input_clf_df.index[0]
            
            input_reg_np = scaled_df.iloc[original_data_index].values
            input_reg = torch.from_numpy(input_reg_np).float()

            next_state_scaled = regressor(input_reg.reshape(1, 1, num_features_lstm))
            next_state_original = scaler.inverse_transform(next_state_scaled.numpy())
            
            yield_col_index = df.columns.get_loc('yield')
            predicted_yield = next_state_original[0, yield_col_index]

        # --- 결과 기록 ---
        original_data_index = input_clf_df.index[0]
        original_row = df.iloc[original_data_index]
        month = original_row['MONTH']
        actual_yield = original_row['yield']
        # 실제 수확 여부 (0 또는 1)
        actual_harvest = y_clf_test.iloc[i]
        
        results_list.append({
            'Month': int(month),
            'Actual_Harvest': actual_harvest,
            'Predicted_Harvest': predicted_harvest,
            'Harvest_Probability(%)': harvest_probability * 100,
            'Actual_Yield': actual_yield,
            'Predicted_Yield': predicted_yield,
        })

# --- 최종 결과 분석 ---
results_df = pd.DataFrame(results_list)

print("\n--- 📝 최종 예측 결과 상세표 (상위 20개) ---")
pd.options.display.float_format = '{:.2f}'.format
print(results_df.head(20).to_string())

print("\n\n" + "="*50)
print("--- 📊 최종 성능 요약 ---")
print("="*50)

# 1. 분류 모델의 최종 성능 (테스트 세트 전체 대상)
print("\n[분류 모델 성능 요약]")
print(classification_report(results_df['Actual_Harvest'], results_df['Predicted_Harvest'], target_names=['No_Harvest (0)', 'Harvest (1)']))

# 2. 회귀 모델의 최종 성능 (실제로 수확이 있었던 샘플 대상)
# 실제 수확이 있었던 데이터만 필터링
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