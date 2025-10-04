import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ===================================================================
# 0. 데이터 준비
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")

file_path = 'Screwed-Backend/data/SPN_Climate.csv'
df = pd.read_csv(file_path)

if 'YIELD' in df.columns:
    df = df.rename(columns={'YIELD': 'yield'})

# 'yield' 컬럼의 NaN은 0으로, 나머지는 평균으로 채우기
df['yield'] = df['yield'].fillna(0)
df.fillna(df.mean(), inplace=True)

# 결과 비교를 위해 MONTH 컬럼은 잠시 보관
month_data = df['MONTH'].values
# 학습에는 불필요하므로 데이터프레임에서 MONTH 컬럼 제외
if 'MONTH' in df.columns:
    df = df.drop('MONTH', axis=1)

num_features = df.shape[1]
print(f"학습에 사용될 변수 개수: {num_features}")
print(f"데이터 형태: {df.shape}")

# 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 시계열 데이터를 LSTM 입력 형태로 변환 (현재 턴 -> 다음 턴)
X, y = [], []
for i in range(len(scaled_data) - 1):
    X.append(scaled_data[i])
    y.append(scaled_data[i + 1])

X = np.array(X).reshape(-1, 1, num_features)
y = np.array(y)

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 가중 손실 함수 (Weighted Loss Function) 정의
# ===================================================================
def weighted_mse_loss(prediction, target, weight=10.0):
    """
    생산량(yield)이 0보다 큰 샘플에 대해 더 큰 가중치를 부여하는 손실 함수
    """
    loss = torch.mean((prediction - target)**2, dim=1) # 각 샘플별로 MSE 계산
    
    # 'yield' 컬럼의 인덱스를 찾음
    yield_col_index = df.columns.get_loc('yield')
    
    # 실제 정답(target)의 yield 값이 0보다 클 경우 가중치를 적용
    weights = torch.where(target[:, yield_col_index] > 0, weight, 1.0)
    
    # 가중치가 적용된 손실의 평균을 반환
    return torch.mean(loss * weights)


# ===================================================================
# 2. LSTM 모델 정의
# ===================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

model = LSTMModel(input_size=num_features, hidden_size=50, output_size=num_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ===================================================================
# 3. 모델 학습
# ===================================================================
print("--- 3. 모델 학습 시작 ---")
epochs = 100 # 학습 횟수를 늘려주는 것이 좋음
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        # ✅ 일반 nn.MSELoss() 대신 직접 만든 가중 손실 함수 사용
        loss = weighted_mse_loss(outputs, y_batch, weight=15.0) # weight 값은 조절 가능

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
print("--- 3. 모델 학습 완료 ---\n")


# ===================================================================
# 4. 최종 예측 결과 비교
# ===================================================================
print("--- 4. 최종 예측 결과 비교 시작 ---")
num_samples_to_test = 24
results_list = []

model.eval()
with torch.no_grad():
    for i in range(min(num_samples_to_test, len(X_test_tensor))):
        
        input_tensor = X_test_tensor[i].unsqueeze(0)
        
        # 모델로 다음 달 전체 상태 예측
        next_state_scaled = model(input_tensor)
        
        # 원래 스케일로 복원
        next_state_original = scaler.inverse_transform(next_state_scaled.numpy())
        predicted_yield = next_state_original[0, df.columns.get_loc('yield')]
        
        # 실제 정답 데이터도 원래 스케일로 복원
        actual_state_original = scaler.inverse_transform(y_test_tensor[i].unsqueeze(0).numpy())
        actual_yield = actual_state_original[0, df.columns.get_loc('yield')]
        
        # 월(Month) 정보 가져오기
        original_data_index = X_train.shape[0] + i
        month = month_data[original_data_index]

        results_list.append({
            'Month': int(month),
            'Actual_Yield': actual_yield,
            'Predicted_Yield': predicted_yield
        })

results_df = pd.DataFrame(results_list)
print("\n--- 최종 예측 결과 ---")
print(results_df.to_string())

print("\n--- 4. 최종 예측 결과 비교 완료 ---")