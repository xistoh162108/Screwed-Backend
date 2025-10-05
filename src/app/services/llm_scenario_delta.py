# src/app/services/llm_scenario_delta.py
from typing import Dict, Any
# 실제 LLM 클라이언트는 프로젝트 환경에 맞게 주입

DELTA_SCHEMA_EXAMPLE = {
  "mode": "relative",
  "deltas": {
    "ALLSKY_SFC_PAR_TOT": {"value": -0.05, "unit": "ratio"},
    "T2M": {"value": -0.8, "unit": "K"}
  },
  "notes": "백색화→단기 복사 상승, 녹지→지표온도 하강 가정...",
  "confidence": 0.62
}

SYSTEM_PROMPT = """당신은 기상 입력 피처(28개)의 정의/단위/허용범위를 아는 과학 보조원입니다.
입력으로 사용자 자연어, 현재 월의 원시 관측치, 피처 설명표를 받으면
다음 JSON 스키마로만 출력하세요:
{ "mode": "relative|absolute", "deltas": {<feat>: {"value": number, "unit": "..."}}, "notes": "...", "confidence": float }
규칙:
- 상대(relative)는 비율(ratio), 절대(absolute)는 물리 단위 사용.
- 정의되지 않은 피처는 제외.
- JSON만 출력.
"""

def extract(nl_command: str, x_t: list, month_now: int, lat: float, lon: float, feature_table: Dict[str, Any]) -> Dict[str, Any]:
    """
    nl_command + 현재 상태를 바탕으로 델타 JSON을 생성.
    실제 구현에서는 LLM 호출. 여기서는 더미/후킹 포인트 제공.
    """
    # TODO: LLM 호출 → JSON 파싱
    # return parsed_json
    return DELTA_SCHEMA_EXAMPLE.copy()