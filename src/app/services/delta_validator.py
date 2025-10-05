# src/app/services/delta_validator.py
from typing import Dict, Tuple, List
import numpy as np

# 예시: 피처 인덱스/단위/허용범위
FEAT_IDX = {"T2M": 0, "ALLSKY_SFC_PAR_TOT": 1, "CLRSKY_SFC_SW_DWN": 2}  # 실제 28개 매핑 필요
BOUNDS: Dict[str, Tuple[float,float]] = {
    "T2M": (-60.0, 60.0),
    "ALLSKY_SFC_PAR_TOT": (0.0, 10_000.0),
    "CLRSKY_SFC_SW_DWN": (0.0, 1_500.0),
}
UNITS: Dict[str, str] = {"T2M": "K", "ALLSKY_SFC_PAR_TOT": "W/m^2", "CLRSKY_SFC_SW_DWN": "W/m^2"}

def apply_deltas(x_t: np.ndarray, deltas: dict, mode: str) -> np.ndarray:
    x_new = x_t.copy()
    for feat, spec in deltas.items():
        if feat not in FEAT_IDX:
            continue
        i = FEAT_IDX[feat]
        v = float(spec["value"])
        if mode == "relative":
            x_new[i] = x_new[i] * (1.0 + v)
        else:
            x_new[i] = x_new[i] + v
        lo, hi = BOUNDS.get(feat, (-np.inf, np.inf))
        x_new[i] = float(np.clip(x_new[i], lo, hi))
    # 간단한 상관 보정(예시): CLRSKY_SFC_SW_DWN ↑ → ALLSKY_SFC_PAR_TOT 도 완만히 ↑
    if "CLRSKY_SFC_SW_DWN" in deltas and "ALLSKY_SFC_PAR_TOT" in FEAT_IDX:
        j = FEAT_IDX["ALLSKY_SFC_PAR_TOT"]
        x_new[j] = float(np.clip(x_new[j] * 1.01, *BOUNDS["ALLSKY_SFC_PAR_TOT"]))
    return x_new