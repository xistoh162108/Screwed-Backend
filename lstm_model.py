import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ===================================================================
# 0. 데이터 준비
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")
file_path = 'Screwed-Backend/data/SPN_Climate.csv'
df = pd.read_csv(file_path)

if 'YIELD' in df.columns:
    df = df.rename(columns={'YIELD': 'yield'})

df['yield'].fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)

# 모델 학습에는 월(MONTH) 정보가 직접적으로 필요 없으므로 제외
df_features = df.drop('MONTH', axis=1)
num_features = df_features.shape[1]

# 데이터 정규화
scaler_features = MinMaxScaler()
scaler_yield = MinMaxScaler()

scaled_features = scaler_features.fit_transform(df_features)
# 생산량(yield)은 별도로 정규화하여 나중에 원래 값으로 복원하기 쉽게 함
scaled_yield = scaler_yield.fit_transform(df[['yield']])

print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 시계열 데이터셋 생성
# ===================================================================
print("--- 1. 시계열 데이터셋 생성 시작 ---")

def create_sequences(features, yield_data, month_data, sequence_length=8):
    """
    (1~8월 기후 -> 9월 생산량) 같은 시계열 데이터 쌍을 생성하는 함수
    """
    X, y = [], []
    for i in range(len(features) - sequence_length):
        # 9월(harvest_month)의 생산량을 예측하는 데이터만 생성
        harvest_month_index = i + sequence_length
        if month_data[harvest_month_index] == 9:
            # 1월부터 8월까지의 기후 데이터를 X로 사용
            X.append(features[i:i+sequence_length])
            # 9월의 생산량을 y로 사용
            y.append(yield_data[harvest_month_index])
            
    return np.array(X), np.array(y)

# 8개월치 데이터를 보고 다음 9번째 달의 생산량을 예측하는 데이터셋 생성
SEQUENCE_LENGTH = 8
X_seq, y_seq = create_sequences(scaled_features, scaled_yield, df['MONTH'].values, SEQUENCE_LENGTH)

if len(X_seq) == 0:
    print("오류: 생성된 시계열 데이터가 없습니다. 데이터나 sequence_length를 확인하세요.")
else:
    print(f"생성된 시계열 데이터셋 형태 (X): {X_seq.shape}") # (샘플 수, 8개월, 변수 개수)
    print(f"생성된 시계열 데이터셋 형태 (y): {y_seq.shape}") # (샘플 수, 1)

    # 훈련/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=True, random_state=42)

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    print("--- 1. 시계열 데이터셋 생성 완료 ---\n")


    # ===================================================================
    # 2. 어텐션 LSTM 모델 정의
    # ===================================================================
    print("--- 2. 어텐션 LSTM 모델 정의 시작 ---")
    
    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super(Attention, self).__init__()
            self.attn = nn.Linear(hidden_size, 1)

        def forward(self, lstm_output):
            attention_weights = self.attn(lstm_output)
            attention_scores = F.softmax(attention_weights, dim=1)
            weighted_output = lstm_output * attention_scores
            context_vector = torch.sum(weighted_output, dim=1)
            return context_vector, attention_scores.squeeze(-1)

    class LSTMAttentionForecaster(nn.Module):
        def __init__(self, input_size, hidden_size=50):
            super(LSTMAttentionForecaster, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            context_vector, attn_scores = self.attention(lstm_out)
            out = self.fc(context_vector)
            return out, attn_scores

    model = LSTMAttentionForecaster(input_size=num_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("--- 2. 어텐션 LSTM 모델 정의 완료 ---\n")

    # ===================================================================
    # 3. 모델 학습
    # ===================================================================
    print("--- 3. 어텐션 LSTM 모델 학습 시작 ---")
    epochs = 200
    for epoch in range(epochs):
        model.train()
        outputs, _ = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    print("--- 3. 모델 학습 완료 ---\n")

    # ===================================================================
    # 4. 모델 평가 및 결과 비교
    # ===================================================================
    print("--- 4. 모델 평가 및 결과 비교 시작 ---")
    model.eval()
    with torch.no_grad():
        y_pred_scaled, attention_scores = model(X_test_tensor)
        
        # 예측값과 실제값을 원래 스케일로 복원
        y_pred = scaler_yield.inverse_transform(y_pred_scaled.numpy())
        y_test_original = scaler_yield.inverse_transform(y_test_tensor.numpy())
        
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        print(f"✅ 평균 절대 오차 (MAE): {mae:.2f}")
        print(f"✅ R-squared (R²): {r2:.4f}")
        
        results_df = pd.DataFrame({
            'Actual_Yield': y_test_original.flatten(),
            'Predicted_Yield': y_pred.flatten()
        })
        
        pd.options.display.float_format = '{:.2f}'.format
        print("\n--- 실제 생산량 vs 예측 생산량 비교 ---")
        print(results_df.to_string())
        
        # 어텐션 스코어 확인 (첫 번째 테스트 샘플에 대해)
        print("\n--- 첫 번째 테스트 샘플의 월별 어텐션 가중치 ---")
        first_sample_attention = attention_scores[0].numpy()
        for month_idx, score in enumerate(first_sample_attention):
            print(f"입력 {month_idx+1}개월차 데이터의 가중치: {score:.4f}")

    print("--- 4. 모델 평가 및 결과 비교 완료 ---")