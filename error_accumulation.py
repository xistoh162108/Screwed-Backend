import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ===================================================================
# 0. 데이터 준비 (공통)
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")
file_path = 'Screwed-Backend/data/SPN_Climate.csv'
df = pd.read_csv(file_path)

if 'YIELD' in df.columns:
    df = df.rename(columns={'YIELD': 'yield'})

df['yield'].fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)

# 월(MONTH) 정보는 나중에 사용하기 위해 따로 저장
month_data = df['MONTH'].values
df_features = df.drop('MONTH', axis=1)

# 데이터 정규화 (MinMaxScaler)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_features)
scaled_df = pd.DataFrame(scaled_features, columns=df_features.columns)

print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 모델 A: 다음 달 기후 예측 LSTM 모델 학습
# ===================================================================
print("--- 1. 기후 예측 LSTM 모델 학습 시작 ---")

# --- 1-1. LSTM용 데이터셋 생성 ---
X_lstm, y_lstm = [], []
for i in range(len(scaled_df) - 1):
    X_lstm.append(scaled_df.iloc[i].values)
    y_lstm.append(scaled_df.iloc[i+1].values)

X_lstm = np.array(X_lstm).reshape(-1, 1, scaled_df.shape[1])
y_lstm = np.array(y_lstm)

# LSTM 모델은 전체 데이터로 학습 (test 분리 안 함)
X_lstm_tensor = torch.from_numpy(X_lstm).float()
y_lstm_tensor = torch.from_numpy(y_lstm).float()

# --- 1-2. LSTM 모델 정의 및 학습 ---
class ClimateForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(ClimateForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

climate_model = ClimateForecaster(input_size=scaled_df.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(climate_model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    outputs = climate_model(X_lstm_tensor)
    loss = criterion(outputs, y_lstm_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'LSTM Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
print("--- 1. 기후 예측 LSTM 모델 학습 완료 ---\n")


# ===================================================================
# 2. 모델 B: 9월 생산량 예측 XGBoost 모델 학습
# ===================================================================
print("--- 2. 생산량 예측 XGBoost 모델 학습 시작 ---")
df_harvest = df[df['yield'] > 0].copy()
if df_harvest.empty:
    print("오류: 생산량 예측 모델을 학습할 데이터가 없습니다.")
else:
    X_xgb = df_harvest.drop(['yield', 'MONTH'], axis=1)
    y_xgb = df_harvest['yield']
    
    yield_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror')
    yield_model.fit(X_xgb, y_xgb)
    print("--- 2. 생산량 예측 XGBoost 모델 학습 완료 ---\n")

# ===================================================================
# 3. 연쇄 예측 시뮬레이션 (수정된 로직)
# ===================================================================
print("--- 3. 연쇄 예측 시뮬레이션 시작 ---")

possible_start_indices = df.index[:-12] # 최소 1년치 데이터가 남도록 보정
num_simulations = 10
selected_indices = np.random.choice(possible_start_indices, min(num_simulations, len(possible_start_indices)), replace=False)

for i, start_month_index in enumerate(selected_indices):
    
    start_month_info = df.iloc[start_month_index]
    year = start_month_index // 12 + 1
    start_month = int(start_month_info['MONTH'])
    
    # ✅ 로직 수정: 다음 9월까지 남은 개월 수 계산
    if start_month < 9:
        steps_to_september = 9 - start_month
    else: # 9월~12월에 시작하면, 다음 해 9월까지 계산
        steps_to_september = (12 - start_month) + 9
        
    print("\n" + "="*50)
    print(f"🔄 시뮬레이션 {i+1}/{num_simulations}:")
    print(f"▶ 시작점: 약 {year}년 {start_month}월")
    print(f"▶ 목표: {steps_to_september}개월 후인 다음 9월의 생산량 예측")
    print("="*50)

    prediction_log = []
    current_month_features = torch.from_numpy(scaled_df.iloc[start_month_index].values).float().reshape(1, 1, -1)

    prediction_log.append({
        'Month': start_month,
        'Predicted_Yield': 0.0,
        'Note': f'실제 {start_month}월 데이터'
    })

    climate_model.eval()
    with torch.no_grad():
        # ✅ 로직 수정: 계산된 개월 수만큼만 반복
        for step in range(1, steps_to_september + 1):
            predicted_features_tensor = climate_model(current_month_features).reshape(1, 1, -1)
            
            predicted_yield = 0.0
            note = '기후 예측'
            
            # 마지막 단계(9월)가 되면, 생산량 예측
            if step == steps_to_september:
                predicted_final_features_scaled = predicted_features_tensor.squeeze().numpy()
                predicted_final_features = scaler.inverse_transform(predicted_final_features_scaled.reshape(1, -1))
                
                predicted_final_df = pd.DataFrame(predicted_final_features, columns=df_features.columns)
                predicted_final_df_for_xgb = predicted_final_df.drop('yield', axis=1)
                
                predicted_yield = yield_model.predict(predicted_final_df_for_xgb)[0]
                note = 'LSTM 기후 예측 -> XGBoost 생산량 예측'

            current_month_features = predicted_features_tensor
            
            current_month = (start_month + step - 1) % 12 + 1
            prediction_log.append({
                'Month': current_month,
                'Predicted_Yield': predicted_yield,
                'Note': note
            })

    # --- 시뮬레이션 결과 출력 ---
    results_df = pd.DataFrame(prediction_log)
    print("\n--- 연쇄 예측 과정 ---")
    pd.options.display.float_format = '{:.2f}'.format
    print(results_df.to_string(index=False))

    # 비교를 위한 실제 생산량
    actual_yield_index = start_month_index + steps_to_september
    actual_yield = df.iloc[actual_yield_index]['yield']
    actual_month = int(df.iloc[actual_yield_index]['MONTH'])
    print("-" * 30)
    print(f"ℹ️ 참고: 실제 {actual_month}월 생산량: {actual_yield:.2f}")