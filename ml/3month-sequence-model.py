import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score

# ===================================================================
# 0. 데이터 준비: 로딩, 병합 및 전처리
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")

# --- 파일 로딩 ---
merged_df = pd.read_csv('Screwed-Backend/data/MERGED_file.csv')

# --- 전처리 ---
# 예측 대상 작물 설정 (SOYBEAN, MAIZE, WHEAT, RICE 중 선택)
TARGET_CROP = 'SOYBEAN'
# 나머지 작물들은 불필요하므로 삭제
other_crops = ['MAIZE', 'WHEAT', 'RICE', 'SOYBEAN']
other_crops.remove(TARGET_CROP)

# ✅ 수정: CODE, LAT, LON을 바로 삭제하지 않고, 위치 식별자로 사용
columns_to_drop = other_crops
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')
merged_df = merged_df.rename(columns={TARGET_CROP: 'yield'})
merged_df.fillna(0, inplace=True)
print(f"전처리 후 데이터 형태: {merged_df.shape}")
print("--- 0. 데이터 준비 완료 ---\n")


# ===================================================================
# 1. 시계열 데이터셋 생성 (수정된 로직)
# ===================================================================
print("--- 1. 시계열 데이터셋 생성 시작 ---")

def create_sequences_by_location(df, sequence_length=3):
    X_list, y_list = [], []
    # ✅ 수정: 위도와 경도를 기준으로 데이터를 그룹화
    #    이렇게 하면 각 지역별로 데이터를 따로 처리할 수 있습니다.
    grouped = df.groupby(['LAT', 'LON'])
    
    print(f"총 {grouped.ngroups}개의 고유한 위치 그룹을 찾았습니다.")
    
    # 각 위치 그룹별로 반복
    for _, group_df in grouped:
        # 각 그룹 내에서 시간순으로 정렬
        group_df = group_df.sort_values(by=['YEAR', 'MONTH']).reset_index(drop=True)
        
        # 기후 데이터 컬럼만 
        #  (위치/시간/수확량 정보 제외)
        climate_features = [col for col in group_df.columns if col not in ['CODE', 'YEAR', 'MONTH', 'LAT', 'LON', 'yield']]
        
        # 그룹 안에 데이터가 충분할 경우에만 시퀀스 생성
        if len(group_df) >= sequence_length:
            for i in range(len(group_df) - sequence_length + 1):
                sequence = group_df.iloc[i : i + sequence_length]
                
                # 월이 연속적인지 확인 (같은 그룹이므로 위치는 항상 동일함)
                is_continuous = all(np.diff(sequence['MONTH'].values) % 12 == 1)
                
                if is_continuous:
                    X_list.append(sequence[climate_features].values)
                    last_month_data = sequence.iloc[-1]
                    y_list.append({
                        'is_harvest': 1 if last_month_data['yield'] > 0 else 0,
                        'yield': last_month_data['yield']
                    })

    if not X_list:
        return np.array([]), pd.DataFrame()
        
    return np.array(X_list), pd.DataFrame(y_list)

# ✅ 수정된 함수를 호출
X_seq, y_df = create_sequences_by_location(merged_df, sequence_length=3)

# ✅ X_seq가 비어있는지 확인하는 방어 코드 (기존과 동일)
if X_seq.shape[0] == 0:
    print("⚠️ 오류: 생성된 3개월 시퀀스 데이터가 없습니다.")
    print("데이터 파일의 내용을 확인하여 한 지역에 연속된 3개월 데이터가 충분히 있는지 확인해주세요.")
    exit()
else:
    print(f"✅ 생성된 3개월 시퀀스 데이터 형태 (X): {X_seq.shape}")
    print(f"✅ 생성된 정답 데이터 형태 (y): {y_df.shape}")

# --- 데이터 분리 (훈련/테스트) ---
# (이 부분은 수정할 필요 없이 그대로 두시면 됩니다.)
# 1. XGBoost용 입력 데이터 (3개월치 데이터를 1차원으로 펼침)
X_flat = X_seq.reshape(X_seq.shape[0], -1)
# 2. LSTM용 입력 데이터 (원본 3D 형태 유지)

# 정답 데이터 분리
y_clf = y_df['is_harvest'].values
y_reg = y_df['yield'].values

# 훈련셋과 테스트셋으로 분리 (stratify로 클래스 비율 유지)
X_flat_train, X_flat_test, X_seq_train, X_seq_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
    train_test_split(X_flat, X_seq, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf)

print(f"훈련 데이터 형태 (X_flat_train): {X_flat_train.shape}")
print(f"테스트 데이터 형태 (X_flat_test): {X_flat_test.shape}")
print("--- 1. 시계열 데이터셋 생성 완료 ---\n")


# ===================================================================
# 2. 모델 1: 수확월 예측 (XGBoost 분류 모델)
# ===================================================================
print("--- 2. XGBoost 분류 모델 학습 시작 ---")

# 데이터 불균형 처리를 위한 가중치 계산
scale_pos_weight = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()
print(f"적용될 가중치 (scale_pos_weight): {scale_pos_weight:.2f}")

# XGBoost 분류 모델 생성 및 학습
classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)
classifier.fit(X_flat_train, y_clf_train)
print("✅ 분류 모델 학습 완료")

# --- 분류 모델 성능 평가 ---
y_clf_pred = classifier.predict(X_flat_test)
print("\n--- 📝 분류 모델 성능 리포트 ---")
print(classification_report(y_clf_test, y_clf_pred, target_names=['수확 안함 (0)', '수확함 (1)']))

# 혼동 행렬 시각화
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_clf_test, y_clf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['수확 안함', '수확함'], yticklabels=['수확 안함', '수확함'])
plt.title('분류 모델 혼동 행렬 (Confusion Matrix)')
plt.ylabel('실제 값')
plt.xlabel('예측 값')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✅ 'confusion_matrix.png' 파일로 혼동 행렬 그래프 저장 완료")
print("--- 2. XGBoost 분류 모델 학습 완료 ---\n")


# ===================================================================
# 3. 모델 2: 생산량 예측 (LSTM 회귀 모델)
# ===================================================================
print("--- 3. LSTM 회귀 모델 학습 시작 ---")

# --- LSTM용 데이터 준비 (수확이 있는 데이터만 필터링 및 정규화) ---
X_seq_train_harvest = X_seq_train[y_clf_train == 1]
y_reg_train_harvest = y_reg_train[y_clf_train == 1]

if len(X_seq_train_harvest) > 0:
    # 정규화 (Scaler)
    # 3D 데이터를 2D로 펼쳐서 scaler 학습 후, 다시 3D로 복원
    scaler_X = MinMaxScaler()
    X_train_2d = X_seq_train_harvest.reshape(-1, X_seq_train_harvest.shape[2])
    scaler_X.fit(X_train_2d)
    X_train_scaled_2d = scaler_X.transform(X_train_2d)
    X_train_scaled_3d = X_train_scaled_2d.reshape(X_seq_train_harvest.shape)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_reg_train_harvest.reshape(-1, 1))
    
    # PyTorch 텐서로 변환
    X_reg_tensor = torch.from_numpy(X_train_scaled_3d).float()
    y_reg_tensor = torch.from_numpy(y_train_scaled).float()

    # --- LSTM 모델 정의 및 학습 ---
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size=50):
            super(LSTMRegressor, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    regressor = LSTMRegressor(input_size=X_seq.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.005)

    epochs = 150
    print("LSTM 모델 학습 중...")
    for epoch in range(epochs):
        regressor.train()
        outputs = regressor(X_reg_tensor)
        loss = criterion(outputs, y_reg_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 30 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    print("✅ 회귀 모델 학습 완료")
else:
    regressor = None
    print("⚠️ 경고: LSTM 모델을 학습할 수확 데이터가 부족합니다.")
print("--- 3. LSTM 회귀 모델 학습 완료 ---\n")


# ===================================================================
# 4. 통합 예측 및 최종 평가
# ===================================================================
print("--- 4. 통합 예측 및 최종 평가 시작 ---")

final_predictions = []
regressor.eval() if regressor else None

with torch.no_grad():
    for i in range(len(X_flat_test)):
        # 1단계: 분류 모델로 수확 여부 예측
        is_harvest_pred = classifier.predict(X_flat_test[[i]])[0]
        
        predicted_yield = 0.0
        if is_harvest_pred == 1 and regressor:
            # 2단계: LSTM으로 생산량 예측
            # 테스트 데이터도 훈련 시 사용한 scaler로 정규화
            X_seq_sample_scaled = scaler_X.transform(X_seq_test[i].reshape(-1, X_seq.shape[2]))
            X_seq_sample_tensor = torch.from_numpy(X_seq_sample_scaled.reshape(1, X_seq.shape[1], X_seq.shape[2])).float()
            
            yield_pred_scaled = regressor(X_seq_sample_tensor)
            # 예측값을 원래 스케일로 복원
            predicted_yield = scaler_y.inverse_transform(yield_pred_scaled.numpy())[0][0]
        
        final_predictions.append(predicted_yield)

# --- 최종 성능 요약 ---
results_df = pd.DataFrame({
    'Actual_Harvest': y_clf_test,
    'Actual_Yield': y_reg_test,
    'Predicted_Yield': final_predictions
})

# 실제 수확이 있었던 데이터만 필터링하여 회귀 성능 평가
harvest_samples_df = results_df[results_df['Actual_Harvest'] == 1]
if not harvest_samples_df.empty:
    mae = mean_absolute_error(harvest_samples_df['Actual_Yield'], harvest_samples_df['Predicted_Yield'])
    r2 = r2_score(harvest_samples_df['Actual_Yield'], harvest_samples_df['Predicted_Yield'])
    print("\n--- 📝 회귀 모델 최종 성능 (실제 수확이 있었던 경우) ---")
    print(f"평균 절대 오차 (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.4f}")

    # 최종 결과 시각화
    plt.figure(figsize=(8, 8))
    plt.scatter(harvest_samples_df['Actual_Yield'], harvest_samples_df['Predicted_Yield'], alpha=0.6)
    plt.plot([harvest_samples_df['Actual_Yield'].min(), harvest_samples_df['Actual_Yield'].max()],
             [harvest_samples_df['Actual_Yield'].min(), harvest_samples_df['Actual_Yield'].max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.title(f'{TARGET_CROP} 실제 생산량 vs 예측 생산량')
    plt.xlabel('실제 생산량')
    plt.ylabel('예측 생산량')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('final_prediction_scatter.png')
    print("✅ 'final_prediction_scatter.png' 파일로 최종 예측 결과 그래프 저장 완료")
else:
    print("\n[회귀 모델 성능 요약]")
    print("평가할 수확 샘플이 테스트 세트에 없습니다.")

print("--- 4. 통합 예측 및 최종 평가 완료 ---")