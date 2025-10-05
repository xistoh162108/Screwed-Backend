# -*- coding: utf-8 -*-
# 실행:
#   python -m src.app.services.test_yield_inference
# 또는
#   python src/app/services/test_yield_inference.py
# -*- coding: utf-8 -*-
# make it runnable: python src/app/services/test_yield_inference.py
if __name__ == "__main__" and __package__ is None:
    import os, sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))  # <-- src를 PYTHONPATH에 추가
from app.services.yield_inference import YieldPredictorLGBM, MultiCropYieldService
import os
import numpy as np
import pandas as pd
from app.services.yield_inference import YieldPredictorLGBM, MultiCropYieldService
from pathlib import Path



def main_single_artifacts():
    # 단일 곡물 디렉토리 (예: soybean)
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]   # Screwed-Backend 루트
    ART_DIR = PROJECT_ROOT / "artifacts" / "yieldInference" / "soybean"      # ✅ 중복 없음
    pred = YieldPredictorLGBM(str(ART_DIR))
    print(f"loaded features ({len(pred.feature_names)}):", pred.feature_names[:8], "...")

    # 1) 단건 예측 (데모: 모든 피처 0)
    row = {f: 0.0 for f in pred.feature_names}
    y_hat = pred.predict_one(row)
    print("single PpA_pred:", y_hat)

    # 2) 배치 예측 (임의 난수)
    X = np.random.randn(5, len(pred.feature_names))
    df = pd.DataFrame(X, columns=pred.feature_names)
    out = pred.predict(df, return_with_inputs=True)
    print(out.head())


def main_multicrop():
    # 멀티 곡물 서비스
    ROOT = os.path.abspath("Screwed-Backend/output")
    svc = MultiCropYieldService(ROOT)

    for crop in ["MAIZE", "RICE", "SOYBEAN", "WHEAT"]:
        try:
            model = svc.get(crop)
        except Exception as e:
            print(f"[{crop}] load failed:", e)
            continue
        row = {f: 0.0 for f in model.feature_names}
        y_hat = model.predict_one(row)
        print(f"[{crop}] PpA_pred (all zeros):", y_hat)


if __name__ == "__main__":
    main_single_artifacts()
    print("\n---- multi-crop check ----")
    main_multicrop()