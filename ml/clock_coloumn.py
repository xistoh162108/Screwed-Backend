import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ===================================================================
# 0. 데이터 준비: 시계열 데이터 생성
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")
# 실제 파일 경로로 수정해주세요.
file_path = 'Screwed-Backend/data/ML_src.csv' 
try:
    df = pd.read_csv(file_path)
    print(f"✅ 파일 로딩 성공. 원본 데이터 형태: {df.shape}")
except FileNotFoundError:
    print(f"⚠️ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
    exit()

df.drop(columns=['MAIZE', 'WHEAT', 'RICE', 'SOYBEAN'])

climate_features = [
    'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
    'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
    'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
    'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
    'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
]
df[climate_features] = df[climate_features].fillna(0)

SEQUENCE_LENGTH = 6
X_sequences, y_targets = [], []
grouped = df.groupby(['CODE', 'LAT', 'LON'])

for _, group_df in grouped:
    group_df = group_df.sort_values(by=['YEAR', 'MONTH']).reset_index(drop=True)
    if len(group_df) >= SEQUENCE_LENGTH + 1:
        for i in range(len(group_df) - SEQUENCE_LENGTH):
            sequence = group_df.loc[i : i + SEQUENCE_LENGTH - 1, climate_features]
            target = group_df.loc[i + SEQUENCE_LENGTH, climate_features]
            X_sequences.append(sequence.values)
            y_targets.append(target.values)

X_final = np.array(X_sequences)
y_final = np.array(y_targets)

if X_final.shape[0] == 0:
    print("⚠️ 오류: 생성된 시계열 데이터가 없습니다.")
    exit()

print(f"✅ 생성된 시계열 데이터 형태 (X): {X_final.shape}")
print(f"✅ 생성된 정답 데이터 형태 (y): {y_final.shape}")
print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 훈련/테스트 데이터 분리 및 정규화
# ===================================================================
print("--- 1. 훈련/테스트 데이터 분리 및 정규화 시작 ---")

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# LSTM은 데이터 스케일에 민감하므로 정규화(Normalization) 필수
# 3D 데이터를 2D로 펼쳐서 scaler 학습 후, 다시 3D로 복원
scaler_X = MinMaxScaler()
X_train_2d = X_train.reshape(-1, X_train.shape[2])
scaler_X.fit(X_train_2d)
X_train_scaled_2d = scaler_X.transform(X_train_2d)
X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)

X_test_scaled_2d = scaler_X.transform(X_test.reshape(-1, X_test.shape[2]))
X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# PyTorch 텐서로 변환
X_train_tensor = torch.from_numpy(X_train_scaled).float()
y_train_tensor = torch.from_numpy(y_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_test_tensor = torch.from_numpy(y_test_scaled).float()
print("✅ 데이터 정규화 및 텐서 변환 완료")
print("--- 1. 훈련/테스트 데이터 분리 및 정규화 완료 ---\n")


# ===================================================================
# 2. LSTM 모델 정의
# ===================================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, # (batch, seq, feature) 형태로 데이터 받음
            dropout=0.2       # 과적합 방지를 위한 드롭아웃
        )
        self.fc = nn.Linear(hidden_size, input_size) # 출력은 다시 28개 기후 변수

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 시계열의 마지막 스텝의 출력만 사용
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

model = LSTMRegressor(input_size=X_final.shape[2])
criterion = nn.MSELoss() # 손실 함수: 평균 제곱 오차
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 옵티마이저: Adam
print("✅ LSTM 모델 정의 완료")


# ===================================================================
# 3. 모델 학습
# ===================================================================
print("\n--- 3. 모델 학습 시작 ---")
epochs = 50
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        model.train()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
print("--- 3. 모델 학습 완료 ---\n")


# ===================================================================
# 4. 모델 평가
# ===================================================================
print("--- 4. 모델 평가 시작 ---")
model.eval() # 모델을 평가 모드로 전환
with torch.no_grad(): # 평가 시에는 그래디언트 계산 비활성화
    
    # 테스트 데이터로 예측 수행
    y_pred_scaled = model(X_test_tensor)
    
    # 정규화된 예측값과 실제값을 원래 스케일로 되돌림
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_test_original = scaler_y.inverse_transform(y_test_scaled)
    
    # 전체 변수에 대한 평균 성능 지표 계산
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"✅ 전체 변수에 대한 평균 절대 오차 (MAE): {mae:.4f}")
    print(f"✅ 전체 변수에 대한 R-squared (R²): {r2:.4f}")

    # --- 특정 변수('T2M')에 대한 결과 분석 및 시각화 ---
    # 'T2M' 컬럼의 인덱스 찾기
    t2m_index = climate_features.index('T2M')
    
    # 'T2M'의 실제값과 예측값만 추출
    actual_t2m = y_test_original[:, t2m_index]
    predicted_t2m = y_pred[:, t2m_index]
    
    t2m_mae = mean_absolute_error(actual_t2m, predicted_t2m)
    t2m_r2 = r2_score(actual_t2m, predicted_t2m)
    
    print(f"\n--- 'T2M' (기온) 변수 예측 성능 ---")
    print(f"✅ 'T2M' 평균 절대 오차 (MAE): {t2m_mae:.4f}")
    print(f"✅ 'T2M' R-squared (R²): {t2m_r2:.4f}")

    # 시각화: 실제값 vs 예측값 산점도
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_t2m, predicted_t2m, alpha=0.5, label='Predictions')
    # 완벽한 예측을 나타내는 y=x 직선 추가
    plt.plot([actual_t2m.min(), actual_t2m.max()], [actual_t2m.min(), actual_t2m.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title("'T2M' (기온) 실제값 vs 예측값")
    plt.xlabel("실제값 (Actual Values)")
    plt.ylabel("예측값 (Predicted Values)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lstm_evaluation_scatter.png')
    print("✅ 'lstm_evaluation_scatter.png' 파일로 평가 그래프가 저장되었습니다.")
    
print("--- 4. 모델 평가 완료 ---")