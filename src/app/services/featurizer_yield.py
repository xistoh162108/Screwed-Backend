#수확량 모델(LightGBM)이 먹는 입력 벡터를 만들어 주는 피처 엔지니어링 모듈
# src/app/services/featurizer_yield.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Sequence, Union

# === (1) 너희 기상 피처 28개 (순서 주의 — climate_inference와 동일)
FEATURES: List[str] = [
    'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
    'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
    'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
    'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
    'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
]

# === (2) 학습 CSV에 맞춘 집계 규칙 (예시)
# 실제 CSV 컬럼명을 확인해서 여기를 "정답"으로 바꿔줘야 함!
AGG_SPECS = [
    # (feature_name, op, out_name)
    ("T2M",             "mean", "T2M_mean_3m"),
    ("T2M_MAX",         "max",  "T2M_MAX_max_3m"),
    ("T2M_MIN",         "min",  "T2M_MIN_min_3m"),
    ("T2M_RANGE",       "mean", "T2M_RANGE_mean_3m"),
    ("T2MWET",          "mean", "T2MWET_mean_3m"),
    ("T2MDEW",          "mean", "T2MDEW_mean_3m"),

    ("PRECTOTCORR",     "sum",  "PRECTOTCORR_sum_3m"),
    ("PRECTOTCORR_SUM", "sum",  "PRECTOTCORR_SUM_sum_3m"),

    ("ALLSKY_SFC_SW_DWN", "mean", "SW_DWN_mean_3m"),
    ("ALLSKY_SFC_SW_DNI", "mean", "SW_DNI_mean_3m"),
    ("ALLSKY_SFC_SW_DIFF","mean", "SW_DIFF_mean_3m"),
    ("CLRSKY_SFC_SW_DWN", "mean", "CLR_SW_DWN_mean_3m"),

    ("ALLSKY_SFC_PAR_TOT","mean", "PAR_TOT_mean_3m"),
    ("ALLSKY_SFC_LW_DWN", "mean", "LW_DWN_mean_3m"),
    ("TOA_SW_DWN",        "mean", "TOA_SW_DWN_mean_3m"),

    ("CLOUD_AMT",         "mean", "CLOUD_AMT_mean_3m"),
    ("ALLSKY_SRF_ALB",    "mean", "ALB_mean_3m"),

    ("QV2M",              "mean", "QV2M_mean_3m"),
    ("RH2M",              "mean", "RH2M_mean_3m"),
    ("PS",                "mean", "PS_mean_3m"),
    ("TS",                "mean", "TS_mean_3m"),
    # 필요시 더 추가
]

ArrayLike = Union[np.ndarray, Sequence[float], Dict[str, float]]

def _to_vector(x: ArrayLike) -> np.ndarray:
    """dict 또는 배열을 28차원 벡터로 변환 (FEATURES 순서 준수)"""
    if isinstance(x, dict):
        return np.array([float(x[f]) for f in FEATURES], dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features, got {arr.shape[-1]}")
    return arr

def make_feature_row_3m(window_3m: Sequence[ArrayLike]) -> pd.DataFrame:
    """
    window_3m: 길이 3 시퀀스 [M-2, M-1, M] (또는 [M-1, M, M+1pred])
               각 원소는 dict({feat:value}) 또는 28차원 벡터
    return: 학습 CSV와 동일 스키마의 1행 DataFrame
    """
    if len(window_3m) != 3:
        raise ValueError("window_3m must have length 3")
    mats = np.stack([_to_vector(x) for x in window_3m], axis=0)  # (3, 28)

    out = {}
    for feat, op, out_name in AGG_SPECS:
        idx = FEATURES.index(feat)
        col = mats[:, idx]  # (3,)
        if op == "mean":
            out[out_name] = float(np.mean(col))
        elif op == "sum":
            out[out_name] = float(np.sum(col))
        elif op == "max":
            out[out_name] = float(np.max(col))
        elif op == "min":
            out[out_name] = float(np.min(col))
        else:
            raise ValueError(f"Unsupported op: {op}")

    # (선택) 월/위경도, 코드 등 메타 추가 컬럼 필요하면 여기서 추가
    # out["month_next"] = month_next
    # out["lat"] = lat
    # out["lon"] = lon
    # out["code"] = code

    return pd.DataFrame([out])  # (1, D)