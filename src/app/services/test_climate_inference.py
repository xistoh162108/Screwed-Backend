# -*- coding: utf-8 -*-
# --- make it runnable directly: python src/app/services/test_climate_inference.py ---
if __name__ == "__main__" and __package__ is None:
    import os, sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
import os, json
import numpy as np
from app.services.climate_inference import ClimatePredictor

FEATURES = [
    'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
    'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
    'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
    'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
    'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
]

ARTDIR = os.path.join(os.path.dirname(__file__), "../../../artifacts/predictClimate")
ARTDIR = os.path.abspath(ARTDIR)  # 절대경로로 변환

MODEL_PATH = os.path.join(ARTDIR, "model.pt")
NORM_PATH  = os.path.join(ARTDIR, "normalizer_stats.json")
CALIB_PATH = os.path.join(ARTDIR, "calibration.json")

print("Model path:", MODEL_PATH)
print("Normalizer path:", NORM_PATH)
print("Calibration path:", CALIB_PATH)

def main():
    # 1) Predictor 준비
    predictor = ClimatePredictor(
        model_path=MODEL_PATH,
        normalizer_stats_path=NORM_PATH,
        calibration_path=CALIB_PATH,
        feature_names=FEATURES,
        sequence_length=6
    )

    # 2) x_window 준비 (실제 서비스에서는 최근 6개월 원스케일 데이터를 채우세요)
    #    여기선 데모로 random; 실전: DB/CSV에서 최근 6개 행을 F열 순서대로 뽑아 넣으면 됨.
    x_window = np.random.rand(6, len(FEATURES)).astype(np.float32)

    month_now = 8   # 현재 월 (예: 8월 → 다음 상태는 9월)
    lat, lon  = 37.5, 127.0

    # 3) 한 스텝 예측 (정규화 공간 그대로; 원스케일 보고 싶으면 return_original_scale=True)
    # 3) 한 스텝 예측
    mu, sigma, P68, P95, P997, month_next = predictor.predict_next(
        x_window, month_now, lat, lon, return_original_scale=False
    )
    print("=== 1-step prediction (normalized) ===")
    print("next month =", month_next)
    print("mu[:5]   =", np.round(mu[:5], 6))
    print("sigma[:5]=", np.round(sigma[:5], 6))
    print("P68[0]   =", np.round(P68[0], 6), "(lo, hi for the 1st feature)")
    
    
    mu, sigma, P68, P95, P997, month_next = predictor.predict_next(
        x_window, month_now, lat, lon, return_original_scale=True
    )
    print("=== 1-step prediction (ORIGINAL scale) ===")
    print("mu[:5]   =", np.round(mu[:5], 6))
    print("P68[0]   =", np.round(P68[0], 6))  # (lo, hi) in original units

    # 4) 6개월 롤아웃 (월만 진전, 입력창은 고정 / autoregressive=True면 mu로 창을 갱신)
    roll = predictor.rollout(
        x_window, month_now, lat, lon,
        steps=6, autoregressive=False, return_original_scale=False
    )
    print("\n=== 6-step rollout (normalized) ===")
    print("months   =", roll["month"].tolist())
    print("mu.shape =", roll["mu"].shape, "sigma.shape =", roll["sigma"].shape)
    print("mu[0,:5] =", np.round(roll["mu"][0, :5], 6))

    # 5) (옵션) 보정 파라미터 확인
    # 5) (옵션) 보정 파라미터 확인
    with open(CALIB_PATH, "r") as f:
        calib = json.load(f)
    print("\n=== calibration.json ===")
    print("dist     =", calib.get("dist"))
    print("scalar_s =", calib.get("scalar_s"))
    print("log_t_f[:8] =", (calib.get("log_t_f") or [])[:8])

from pathlib import Path

def debug_normalizer_json():
    print("Normalizer JSON path:", NORM_PATH)
    with open(NORM_PATH, "r") as f:
        stats = json.load(f)

    print("=== Normalizer JSON top-level keys (first 10) ===")
    keys = list(stats.keys())
    print(keys[:10], " ... total:", len(keys))

    if "__global__" in stats:
        g = stats["__global__"]
        print("\n=== __global__ feature keys (first 10) ===")
        fkeys = list(g.keys())
        print(fkeys[:10], " ... total:", len(fkeys))
        need = "ALLSKY_SFC_LW_DWN"
        if need in g:
            print(f"✅ __global__ has {need}: snippet =", g[need])
        else:
            print(f"❌ {need} not in __global__")
    else:
        print("❌ '__global__' key missing in normalizer_stats.json")

if __name__ == "__main__":
    debug_normalizer_json()
    main()