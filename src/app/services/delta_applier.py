# app/services/delta_applier.py
"""
델타 적용 서비스: 농업 행동에 따른 환경 변수 변화 적용
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.core.delta_rules import (
    DELTA_RULES,
    INTENT_TO_DELTA_KEY,
    VARIABLE_PHYSICAL_LIMITS,
    CROP_NAME_MAPPING,
)


class DeltaApplier:
    """델타 적용 및 클램핑"""

    def __init__(self):
        self.delta_rules = DELTA_RULES
        self.intent_mapping = INTENT_TO_DELTA_KEY
        self.limits = VARIABLE_PHYSICAL_LIMITS

    def map_intent_to_deltas(self, proc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        procedureAnalyzer 결과를 델타 룰로 변환

        Args:
            proc: {intent, crop, area, irrigation, shading, mulching, ...}

        Returns:
            {
                "GWETROOT": {"op": "+", "value": 0.02, "clamp": [0,1]},
                ...
            }
        """
        deltas = {}

        # 1. Intent 기반 매핑
        intent = proc.get("intent", "").lower()
        for key, delta_key in self.intent_mapping.items():
            if key in intent:
                rule = self.delta_rules.get(delta_key, {})
                deltas.update(rule)
                break

        # 2. 세부 파라미터 기반 매핑
        # 관개 방식
        irrigation = proc.get("irrigation", "").lower()
        if "drip" in irrigation or "드립" in irrigation or "점적" in irrigation:
            deltas.update(self.delta_rules.get("irrigation=drip", {}))
        elif "sprinkler" in irrigation or "스프링클러" in irrigation:
            deltas.update(self.delta_rules.get("irrigation=sprinkler", {}))
        elif "night" in irrigation or "야간" in irrigation:
            deltas.update(self.delta_rules.get("irrigation=night", {}))
        elif "stop" in irrigation or "중단" in irrigation:
            deltas.update(self.delta_rules.get("irrigation=stop", {}))

        # 차광
        shading = proc.get("shading", "").lower()
        if "on" in shading or "설치" in shading or "차광" in shading:
            deltas.update(self.delta_rules.get("shading=on", {}))

        # 멀칭
        mulching = proc.get("mulching", "").lower()
        if "straw" in mulching or "짚" in mulching or "볏짚" in mulching:
            deltas.update(self.delta_rules.get("mulching=straw", {}))
        elif "plastic" in mulching or "비닐" in mulching or "플라스틱" in mulching:
            deltas.update(self.delta_rules.get("mulching=plastic", {}))

        # 풍력차단
        if proc.get("windbreak") or "방풍" in intent or "windbreak" in intent:
            deltas.update(self.delta_rules.get("windbreak=installed", {}))

        return deltas

    def apply_deltas(
        self, df: pd.DataFrame, deltas: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        데이터프레임에 델타 적용 및 클램핑

        Args:
            df: 원본 윈도우 데이터
            deltas: 델타 룰 딕셔너리

        Returns:
            델타 적용 후 데이터프레임 (복사본)
        """
        df_adj = df.copy()

        for col, rule in deltas.items():
            if col not in df_adj.columns:
                continue

            op = rule.get("op", "*")
            value = rule.get("value", 1.0)
            clamp = rule.get("clamp")

            # 연산 적용
            if op == "+":
                df_adj[col] = df_adj[col] + value
            elif op == "*":
                df_adj[col] = df_adj[col] * value
            else:
                raise ValueError(f"Unknown operation: {op}")

            # 클램핑
            if clamp:
                lower, upper = clamp
                if lower is not None:
                    df_adj[col] = df_adj[col].clip(lower=lower)
                if upper is not None:
                    df_adj[col] = df_adj[col].clip(upper=upper)

            # 물리적 한계 클램핑
            if col in self.limits:
                lower, upper = self.limits[col]
                if lower is not None:
                    df_adj[col] = df_adj[col].clip(lower=lower)
                if upper is not None:
                    df_adj[col] = df_adj[col].clip(upper=upper)

        return df_adj

    def normalize_crop_name(self, crop: str) -> Optional[str]:
        """작물명 정규화 (한글 → 영문)"""
        crop_lower = crop.lower()
        return CROP_NAME_MAPPING.get(crop_lower, crop.upper())

    def extract_action_params(self, proc: Dict[str, Any]) -> Dict[str, Any]:
        """
        procedureAnalyzer 결과에서 시뮬레이션 파라미터 추출

        Returns:
            {
                "crop": "MAIZE",
                "area": 100,
                "unit": "acre",
                "irrigation": "drip",
                "deltas": {...}
            }
        """
        crop_raw = proc.get("crop", "")
        crop = self.normalize_crop_name(crop_raw) if crop_raw else None

        area = proc.get("area", 0)
        unit = proc.get("unit", "acre")

        # 단위 변환 (에이커 → 헥타르)
        if unit.lower() in ["acre", "에이커"]:
            area_ha = area * 0.404686  # 1 acre = 0.404686 ha
        else:
            area_ha = area

        deltas = self.map_intent_to_deltas(proc)

        return {
            "crop": crop,
            "area_acre": area,
            "area_ha": area_ha,
            "unit": unit,
            "irrigation": proc.get("irrigation"),
            "shading": proc.get("shading"),
            "mulching": proc.get("mulching"),
            "deltas": deltas,
        }


# 싱글톤 인스턴스
_applier = None


def get_delta_applier() -> DeltaApplier:
    """글로벌 DeltaApplier 인스턴스 반환"""
    global _applier
    if _applier is None:
        _applier = DeltaApplier()
    return _applier


# 편의 함수
def apply_action_to_window(
    df: pd.DataFrame, proc: Dict[str, Any]
) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    행동을 윈도우에 적용

    Args:
        df: 원본 윈도우 데이터
        proc: procedureAnalyzer 결과

    Returns:
        (조정된 데이터프레임, 액션 파라미터, 적용된 델타)
    """
    applier = get_delta_applier()

    # 액션 파라미터 추출
    action_params = applier.extract_action_params(proc)
    deltas = action_params["deltas"]

    # 델타 적용
    df_adjusted = applier.apply_deltas(df, deltas)

    return df_adjusted, action_params, deltas


def format_deltas_for_response(deltas: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    델타를 API 응답 형식으로 변환

    Args:
        deltas: 내부 델타 룰 형식

    Returns:
        {"col": {"op": "*", "value": 1.02}}
    """
    formatted = {}
    for col, rule in deltas.items():
        formatted[col] = {
            "op": rule.get("op", "*"),
            "value": rule.get("value", 1.0),
        }
    return formatted


def get_delta_summary(deltas: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    델타를 사람이 읽을 수 있는 텍스트로 변환

    Args:
        deltas: 델타 룰

    Returns:
        ["GWETROOT +0.02로 토양수분 개선", ...]
    """
    summary = []
    for col, rule in deltas.items():
        op = rule.get("op", "*")
        value = rule.get("value", 1.0)

        if op == "+":
            if value > 0:
                summary.append(f"{col} +{value:.3f} 증가")
            else:
                summary.append(f"{col} {value:.3f} 감소")
        elif op == "*":
            pct = (value - 1.0) * 100
            if pct > 0:
                summary.append(f"{col} {pct:+.1f}% 증가")
            else:
                summary.append(f"{col} {pct:.1f}% 감소")

    return summary
