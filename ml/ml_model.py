# ==========================================
# 3month_Climate_and_{target_crop}.csv 구조 전용
# LightGBM (native API) + custom feval(R^2)
# - 8:2 split
# - 100 에폭마다 R^2 로그 출력
# - 조기종료(early_stopping)
# - R^2 학습 곡선 + (Train/Valid) 예측-실측 산점도
# - 모델/메트릭/예측/그래프 저장
# ==========================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import platform
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation

# -----------------------------
# 0) 설정
# -----------------------------
CSV_PATH   = r"C:\Users\HJ\OneDrive\Desktop\screwed\Data_for_ML\3month_Climate_and_SOYBEAN.csv"
OUT_DIR    = r"C:\Users\HJ\OneDrive\Desktop\ml\output\SOYBEAN"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(OUT_DIR, "LightGBM_PpA_model.pkl")
METRICS_PATH  = os.path.join(OUT_DIR, "metrics.json")
PRED_PATH     = os.path.join(OUT_DIR, "validation_predictions.csv")
FEATURES_PATH = os.path.join(OUT_DIR, "feature_names.json")

CURVE_PNG       = os.path.join(OUT_DIR, "r2_learning_curve.png")
TRAIN_SCATTER   = os.path.join(OUT_DIR, "train_pred_vs_true.png")
VALID_SCATTER   = os.path.join(OUT_DIR, "valid_pred_vs_true.png")

SEED        = 42
NUM_ROUNDS  = 5000
EARLY_STOP  = 200
LOG_PERIOD  = 100

# -----------------------------
# 1) 데이터 로드
# -----------------------------
df = pd.read_csv(CSV_PATH)
print(f"✅ 데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

# -----------------------------
# 2) X / y 분리
# -----------------------------
if "PpA" not in df.columns:
    raise ValueError("❌ 'PpA' 컬럼이 존재하지 않습니다. 타깃 컬럼명을 확인하세요.")

X = df.drop(columns=["PpA"])
y = df["PpA"]
feature_names = list(X.columns)

# -----------------------------
# 3) 8:2 분할
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)
print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

# -----------------------------
# 4) 결측치 처리
# -----------------------------
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)

# -----------------------------
# 5) Dataset 생성
# -----------------------------
lgb_train = lgb.Dataset(X_train_imp, label=y_train, feature_name=feature_names, free_raw_data=False)
lgb_valid = lgb.Dataset(X_val_imp,   label=y_val,   reference=lgb_train, feature_name=feature_names, free_raw_data=False)

# -----------------------------
# 6) custom feval(R^2)
# -----------------------------
def r2_metric(y_pred, dataset):
    y_true = dataset.get_label()
    return ("R2", r2_score(y_true, y_pred), True)

# -----------------------------
# 7) 학습 (R^2만 로그)
# -----------------------------
params = dict(
    objective="regression",
    metric="None",
    learning_rate=0.02,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    seed=SEED,
    verbose=-1,
)

evals_result = {}
booster = lgb.train(
    params=params,
    train_set=lgb_train,
    num_boost_round=NUM_ROUNDS,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train", "valid_1"],
    feval=r2_metric,
    callbacks=[
        early_stopping(stopping_rounds=EARLY_STOP),
        log_evaluation(period=LOG_PERIOD),
        record_evaluation(evals_result)
    ]
)

best_iter = booster.best_iteration if booster.best_iteration is not None else NUM_ROUNDS

# -----------------------------
# 8) 최종 성능
# -----------------------------
y_train_pred = booster.predict(X_train_imp, num_iteration=best_iter)
y_val_pred   = booster.predict(X_val_imp,   num_iteration=best_iter)

train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
train_mae  = mean_absolute_error(y_train, y_train_pred)
train_r2   = r2_score(y_train, y_train_pred)

val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
val_mae  = mean_absolute_error(y_val, y_val_pred)
val_r2   = r2_score(y_val, y_val_pred)

print("\n📊 학습/검증 성능 요약 (최종 모델)")
print(f"Train → RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R^2: {train_r2:.4f}")
print(f"Valid → RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R^2: {val_r2:.4f}")
print(f"(best_iteration = {best_iter})")

# -----------------------------
# 9) 폰트 설정 (그래프용, 한글/유니코드)
# -----------------------------
def setup_fonts():
    sys = platform.system()
    if sys == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif sys == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
setup_fonts()

# -----------------------------
# 10) R^2 꺾은선 그래프 (Epoch별)
# -----------------------------
r2_train_hist = np.array(evals_result["train"]["R2"])
r2_valid_hist = np.array(evals_result["valid_1"]["R2"])
epochs        = np.arange(1, len(r2_train_hist) + 1)

plt.figure(figsize=(7.5, 4.5))
plt.plot(epochs, r2_train_hist, label="Train R^2")
plt.plot(epochs, r2_valid_hist, label="Validation R^2")
plt.xlabel("Epoch (Trees)")
plt.ylabel("R^2")
plt.title("R^2 vs Epoch (LightGBM)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(CURVE_PNG, dpi=150)
plt.show()
print(f"🖼️ R^2 학습 곡선 저장: {CURVE_PNG}")

# -----------------------------
# 11) (Train/Valid) 예측값 vs 실제값 산점도 + R^2
# -----------------------------
def plot_pred_vs_true(y_true, y_pred, title, save_path, color):
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(5.5, 5))
    plt.scatter(y_pred, y_true, s=8, alpha=0.35, color=color, edgecolors="none")
    vmin = min(np.min(y_true), np.min(y_pred))
    vmax = max(np.max(y_true), np.max(y_pred))
    plt.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.2, label="y = x 기준선")
    plt.xlabel("예측값 (Prediction)")
    plt.ylabel("실측값 (Actual)")
    plt.title(f"{title}  |  R² = {r2:.4f}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"🖼️ {title} 저장: {save_path}  (R² = {r2:.4f})")

plot_pred_vs_true(y_train, y_train_pred, "Train: 예측 vs 실측", TRAIN_SCATTER, "#4caf50")
plot_pred_vs_true(y_val,   y_val_pred,   "Validation: 예측 vs 실측", VALID_SCATTER, "#2196f3")

# -----------------------------
# 12) 결과 저장
# -----------------------------
joblib.dump({"imputer": imputer, "booster": booster, "best_iteration": best_iter}, MODEL_PATH)

metrics = {
    "final": {
        "train": {"RMSE": float(train_rmse), "MAE": float(train_mae), "R2": float(train_r2)},
        "validation": {"RMSE": float(val_rmse), "MAE": float(val_mae), "R2": float(val_r2)},
        "best_iteration": int(best_iter)
    }
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=2, ensure_ascii=False)

val_out = X_val.copy()
val_out["PpA_true"] = y_val.values
val_out["PpA_pred"] = y_val_pred
val_out.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

print("\n✅ 저장 완료")
print(f"- 모델: {MODEL_PATH}")
print(f"- 메트릭: {METRICS_PATH}")
print(f"- 피처: {FEATURES_PATH}")
print(f"- 검증 예측 CSV: {PRED_PATH}")
print(f"- 그래프: {CURVE_PNG}, {TRAIN_SCATTER}, {VALID_SCATTER}")
