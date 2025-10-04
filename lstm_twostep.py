import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# ===================================================================
# 0. 데이터 준비 (SPN_Climate.csv 파일 불러오기 및 전처리)
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")

file_path = 'Screwed-Backend/data/MERGED_Climate.csv'
df = pd.read_csv(file_path)

columns_to_drop = ['CODE', 'YEAR', 'LAT', 'LON']
# 파일에 해당 컬럼이 없을 경우 에러가 나지 않도록 errors='ignore' 옵션 추가
df = df.drop(columns=columns_to_drop, errors='ignore')

# 생산량 컬럼 이름을 'yield'로 변경
if 'YIELD' in df.columns:
    df = df.rename(columns={'YIELD': 'yield'})

# ✅ 1단계: 'yield' 컬럼의 NaN 값만 0으로 먼저 채웁니다.
print(f"처리 전, yield 컬럼의 NaN 개수: {df['yield'].isnull().sum()}")
df['yield'].fillna(0, inplace=True)
print(f"처리 후, yield 컬럼의 NaN 개수: {df['yield'].isnull().sum()}")

# ✅ 2단계: 나머지 컬럼들의 NaN 값은 각 컬럼의 평균으로 채웁니다.
# 'yield'는 이미 NaN이 없으므로 영향을 받지 않습니다.
df.fillna(df.mean(), inplace=True)
print(f"전체 결측치 처리 후, NaN 총 개수: {df.isnull().sum().sum()}")

# 변수의 개수를 데이터에 맞춰 자동으로 설정
num_features = df.shape[1]
print(f"원본 데이터 형태: {df.shape}")
print("--- 0. 데이터 준비 완료 ---\n")

# 데이터 정규화 (이후 코드는 동일)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)


# ===================================================================
# 1. 모델 1: 수확 여부 예측 (분류 모델)
# ===================================================================
print("--- 1. 분류 모델 학습 시작 ---")

# --- 1-1. 분류 모델용 데이터 준비 ---
# 입력(X): 생산량('yield')과 'MONTH'를 제외한 나머지 변수
X_clf_df = scaled_df.drop(['yield', 'MONTH'], axis=1) 
# 정답(y): 생산량이 0보다 크면 1, 아니면 0
y_clf_df = (df['yield'] > 0).astype(np.float32)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf_df.values, y_clf_df.values, test_size=0.2, shuffle=False
)

# PyTorch 텐서로 변환
X_clf_train_tensor = torch.from_numpy(X_clf_train).float() # ✅ .float() 추가
y_clf_train_tensor = torch.from_numpy(y_clf_train).unsqueeze(1).float() # ✅ .float() 추가
X_clf_test_tensor = torch.from_numpy(X_clf_test).float() # ✅ .float() 추가
y_clf_test_tensor = torch.from_numpy(y_clf_test).unsqueeze(1).float() # ✅ .float() 추가

# --- 1-2. 분류 모델 정의 ---
class ClassifierModel(nn.Module):
    def __init__(self, input_size):
        super(ClassifierModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

# ✅ 수정된 부분: input_size를 num_features - 2로 변경
# 원본(30개)에서 'yield'와 'MONTH' 2개를 제외했으므로 입력 크기는 28이 됩니다.
classifier = ClassifierModel(input_size=num_features - 2)

# ✅ 가중치 계산 추가
# '수확 없음' 클래스의 개수 / '수확 있음' 클래스의 개수를 계산하여 가중치로 사용
# y_clf_train은 numpy 배열입니다.
count_neg = len(y_clf_train[y_clf_train == 0])
count_pos = len(y_clf_train[y_clf_train == 1])
weight = torch.tensor(count_neg / count_pos).float()
print(f"\n✅ 적용될 가중치 (pos_weight): {weight:.2f}") 
print(f"▶ 해석: '수확하는 달(1)'을 예측할 때 약 {weight:.2f}배의 중요도를 부여합니다.")

# ✅ 손실 함수에 pos_weight 적용
criterion_clf = nn.BCEWithLogitsLoss(pos_weight=weight)
optimizer_clf = torch.optim.Adam(classifier.parameters(), lr=0.001)

# --- 1-3. 분류 모델 학습 ---
epochs = 20
for epoch in range(epochs):
    outputs = classifier(X_clf_train_tensor)
    loss = criterion_clf(outputs, y_clf_train_tensor)
    optimizer_clf.zero_grad()
    loss.backward()
    optimizer_clf.step()
    if (epoch + 1) % 5 == 0:
        print(f'분류 모델 Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- 1-4. 분류 모델 평가 (정밀 진단) ---
# 추가 임포트
from sklearn.metrics import classification_report, confusion_matrix

classifier.eval()
with torch.no_grad():
    test_outputs = classifier(X_clf_test_tensor)
    predicted = (torch.sigmoid(test_outputs) > 0.5).float()
    
    y_true = y_clf_test_tensor.numpy()
    y_pred = predicted.numpy()

    print("\n--- 분류 모델 정밀 진단 리포트 ---")
    # classification_report는 precision, recall, f1-score를 한 번에 보여줍니다.
    # target_names는 클래스 0과 1의 이름을 지정해줍니다.
    print(classification_report(y_true, y_pred, target_names=['No_Harvest (0)', 'Harvest (1)']))
    
    print("--- 혼동 행렬 (Confusion Matrix) ---")
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("해석: Harvest(1) 클래스의 recall 점수가 0.0이면, 모델이 수확하는 달을 전혀 맞추지 못하고 있다는 의미입니다.")


# ===================================================================
# 2. 모델 2: 생산량 예측 (회귀 LSTM 모델)
# ===================================================================
print("--- 2. 회귀 모델 학습 시작 ---")

# --- 2-1. 회귀 모델용 데이터 준비 ---
# 생산량이 0보다 큰 데이터만 필터링
harvest_only_df = scaled_df[scaled_df['yield'] > 0]

# LSTM 입력 형태로 변환
X_reg, y_reg = [], []
for i in range(len(harvest_only_df) - 1):
    X_reg.append(harvest_only_df.iloc[i].values)
    y_reg.append(harvest_only_df.iloc[i + 1].values)

X_reg = np.array(X_reg).reshape(-1, 1, num_features)
y_reg = np.array(y_reg)

X_reg_tensor = torch.from_numpy(X_reg).float() # ✅ .float() 추가
y_reg_tensor = torch.from_numpy(y_reg).float() # ✅ .float() 추가

# --- 2-2. LSTM 모델 정의 (이전과 동일) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

regressor = LSTMModel(input_size=num_features, hidden_size=50, output_size=num_features)
criterion_reg = nn.MSELoss()
optimizer_reg = torch.optim.Adam(regressor.parameters(), lr=0.001)

# --- 2-3. 회귀 모델 학습 ---
if len(X_reg_tensor) > 0: # 학습할 데이터가 있을 경우에만
    for epoch in range(epochs):
        outputs = regressor(X_reg_tensor)
        loss = criterion_reg(outputs, y_reg_tensor)
        optimizer_reg.zero_grad()
        loss.backward()
        optimizer_reg.step()
        if (epoch + 1) % 5 == 0:
            print(f'회귀 모델 Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    print("✅ 회귀 모델 MAE:", mean_absolute_error(y_reg, regressor(X_reg_tensor).detach().numpy()))
else:
    print("회귀 모델 학습 데이터가 부족합니다.")
print("--- 2. 회귀 모델 학습 완료 ---\n")


# ===================================================================
# 3. 통합 예측 결과 상세 비교
# ===================================================================
print("--- 3. 통합 예측 결과 상세 비교 시작 ---")

num_samples_to_test = 24 # 2년치(24달) 데이터를 테스트
results_list = []

classifier.eval()
regressor.eval()
with torch.no_grad():
    for i in range(min(num_samples_to_test, len(X_clf_test_tensor))):
        
        # 1. 예측에 사용할 입력 데이터 준비
        original_data_index = X_clf_train.shape[0] + i
        
        # 분류 모델 입력 (yield와 MONTH 제외)
        input_clf_np = scaled_df.iloc[original_data_index].drop(['yield', 'MONTH']).values
        input_clf = torch.from_numpy(input_clf_np).float()
        
        # 회귀 모델 입력 (yield, MONTH 포함)
        input_reg_np = scaled_df.iloc[original_data_index].values
        input_reg = torch.from_numpy(input_reg_np).float()

        # 2. 1단계: 분류 모델로 수확 여부 예측
        is_harvest_prob = torch.sigmoid(classifier(input_clf)).item()
        
        # 3. 2단계: 수확 여부에 따라 생산량 예측
        predicted_yield = 0.0
        if is_harvest_prob > 0.5:
            next_state_scaled = regressor(input_reg.reshape(1, 1, num_features))
            next_state_original = scaler.inverse_transform(next_state_scaled.numpy())
            predicted_yield = next_state_original[0, df.columns.get_loc('yield')]

        # 4. 결과 저장을 위해 원본 데이터 가져오기
        original_row = df.iloc[original_data_index]
        month = original_row['MONTH']
        actual_yield = original_row['yield']
        
        results_list.append({
            'Month': int(month),
            'Actual_Yield': actual_yield,
            'Predicted_Yield': predicted_yield
        })

# 5. 최종 결과를 Pandas DataFrame으로 변환하여 출력
results_df = pd.DataFrame(results_list)
print("\n--- 최종 예측 결과 ---")
print(results_df.to_string())

print("\n--- 3. 통합 예측 결과 상세 비교 완료 ---")