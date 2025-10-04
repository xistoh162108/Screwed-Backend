import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ===================================================================
# 0. 데이터 준비
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")

file_path = 'Screwed-Backend/data/SPN_Climate.csv'
df = pd.read_csv(file_path)

# 생산량 컬럼 이름을 'yield'로 통일
if 'YIELD' in df.columns:
    df = df.rename(columns={'YIELD': 'yield'})

# 'yield' 컬럼의 NaN은 0으로, 나머지 컬럼의 NaN은 평균으로 채우기
df['yield'].fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)

# 💡 **수확이 있는 데이터만으로 학습**
# 생산량이 0인 데이터는 회귀 모델 학습에 방해가 될 수 있으므로,
# 실제 수확이 발생한 데이터만 사용해 '얼마나' 생산되는지를 학습합니다.
df_harvest = df[df['yield'] > 0].copy()

if df_harvest.empty:
    print("오류: 학습에 사용할 수확 데이터가 없습니다.")
else:
    # --- 입력 변수(X)와 정답(y) 분리 ---
    # X: 생산량('yield')을 제외한 모든 변수
    X = df_harvest.drop('yield', axis=1)
    # y: 생산량('yield') 변수
    y = df_harvest['yield']

    print(f"총 {len(df)}개 데이터 중 수확이 있는 데이터 {len(df_harvest)}개를 학습에 사용합니다.")
    print(f"입력 변수(X)의 형태: {X.shape}")
    print(f"정답(y)의 형태: {y.shape}")

    # --- 훈련 데이터와 테스트 데이터 분리 ---
    # shuffle=True로 데이터를 무작위로 섞어주는 것이 좋습니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print("--- 0. 데이터 준비 완료 ---\n")


    # ===================================================================
    # 1. XGBoost 모델 학습
    # ===================================================================
    print("--- 1. XGBoost 모델 학습 시작 ---")
    
    # XGBoost 회귀 모델 생성
    # n_estimators: 만들 트리의 개수
    # max_depth: 트리의 최대 깊이
    # learning_rate: 학습률
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror')

    # 모델 학습
    model.fit(X_train, y_train)
    
    print("--- 1. XGBoost 모델 학습 완료 ---\n")


    # ===================================================================
    # 2. 모델 성능 평가
    # ===================================================================
    print("--- 2. 모델 성능 평가 시작 ---")
    
    # 테스트 데이터로 예측 수행
    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ 평균 절대 오차 (MAE): {mae:.4f}")
    print(f"▶ 해석: 모델의 예측값은 실제값과 평균적으로 약 {mae:.4f} 만큼 차이납니다.")
    print(f"✅ R-squared (R²): {r2:.4f}")
    print(f"▶ 해석: 모델이 생산량의 분산을 약 {r2*100:.2f}% 설명합니다. (1에 가까울수록 좋음)")

    print("--- 2. 모델 성능 평가 완료 ---\n")


    # ===================================================================
    # 3. 최종 예측 결과 비교
    # ===================================================================
    print("--- 3. 최종 예측 결과 비교 시작 ---")
    
    # 예측 결과를 보기 쉽게 DataFrame으로 만듭니다.
    results_df = pd.DataFrame({
        'Actual_Yield': y_test,
        'Predicted_Yield': y_pred
    })
    
    # 소수점 2자리까지만 표시하도록 설정
    pd.options.display.float_format = '{:.2f}'.format
    
    print("\n--- 실제 생산량 vs 예측 생산량 비교 (샘플 10개) ---")
    print(results_df.head(10).to_string())

    print("\n--- 3. 최종 예측 결과 비교 완료 ---")