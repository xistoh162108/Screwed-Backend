# app/services/prediction_service.py (신규)
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import datetime as dt

def run_prediction(
    *,
    variables: List[str],
    location: Dict[str, float] | None,
    horizon_days: int,
    context: Dict[str, Any],
    user_text: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(42)
    days = horizon_days if horizon_days and horizon_days > 0 else 30

    # 더미: 핵심 변수 3개만 사용해 물 스트레스 대략치 생성
    t = 20 + 5*np.sin(np.linspace(0, 2*np.pi, days)) + rng.normal(0, 0.6, days)   # T2M
    rh = 60 + 8*np.sin(np.linspace(0, 4*np.pi, days)) + rng.normal(0, 2.0, days)  # RH2M
    pr = np.maximum(0, rng.gamma(shape=2.0, scale=1.1, size=days) - 0.8)          # PRECTOTCORR

    stress = (
        np.maximum(0, (t - 22)/8.0) * 0.5 +
        np.maximum(0, (65 - rh)/20.0) * 0.3 +
        np.maximum(0, 1.0 - np.tanh(pr)) * 0.2
    )
    stress = np.clip(stress, 0, 1)

    contrib_raw = [
        ("T2M", float(np.var(t))),
        ("RH2M", float(np.var(rh))),
        ("PRECTOTCORR", float(np.var(pr))),
    ]
    s = sum(v for _, v in contrib_raw) or 1.0
    top_features = [{"name": k, "contrib": v/s} for k, v in contrib_raw]
    top_features.sort(key=lambda x: x["contrib"], reverse=True)

    pred = {
        "target": "next_month_water_stress_index",
        "yhat": [float(x) for x in stress.tolist()],
        "ts_start": dt.date.today().isoformat(),
        "freq": "D",
        "variables_used": variables,
        "location": location or {},
        "horizon_days": days,
    }
    impact = {"top_features": top_features}
    delta_stats = {
        "baseline": 0.25,
        "pred_mean": float(np.mean(stress)),
        "delta": float(np.mean(stress) - 0.25),
    }
    return pred, impact, delta_stats