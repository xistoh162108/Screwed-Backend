# app/services/llm_to_ml_service.py

import json
from typing import Dict, Any, List, Tuple

# --------------------------------------------------------------------------
# 이 함수가 이 파일의 핵심입니다.
# --------------------------------------------------------------------------
def prepare_ml_inputs(
    structured_command: Dict[str, Any],
    game_state: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    LLM이 분석한 명령과 현재 게임 상태를 바탕으로,
    두 ML 모델에 입력으로 전달할 데이터를 생성합니다.

    :param structured_command: procedureAnalyzer가 생성한 JSON 객체
    :param game_state: 현재 게임의 상태 (돈, 턴, 기후 기록 등)
    :return: (기후 예측 모델 입력, 생산량 예측 모델 입력) 튜플
    """
    intent = structured_command.get("intent")
    
    # --- 현재 턴의 기후 데이터를 먼저 계산합니다. ---
    # 기록된 기후 중 가장 최신 데이터를 가져옵니다.
    # game_state에 'climate_history'가 있다고 가정합니다.
    current_climate = game_state.get("climate_history", [])[-1].copy()

    # 사용자가 기후를 변경하는 명령을 내렸을 경우, 현재 턴 기후에 반영합니다.
    if intent == "modify_climate":
        params = structured_command.get("parameters", {})
        variable = params.get("variable") # 예: "temperature"
        value = float(params.get("value", 0))  # 예: "+2" -> 2.0

        if variable in current_climate:
            current_climate[variable] += value

    # --- ML 모델 입력 데이터 생성 ---

    # 1. 기후 예측 모델 입력 데이터 (현재 턴의 기후 데이터 1개)
    climate_model_input = current_climate

    # 2. 생산량 예측 모델 입력 데이터 (최근 3개월치 기후 데이터)
    #    - 과거 2개월치 + 현재 턴의 변경된 기후 데이터
    past_two_months = game_state.get("climate_history", [])[-2:]
    yield_model_input = past_two_months + [current_climate]
    
    # 생성된 두 종류의 입력 데이터를 반환합니다.
    return climate_model_input, yield_model_input


# --- 테스트를 위한 예시 ---
if __name__ == '__main__':
    # 1. LLM (procedureAnalyzer)이 이런 결과를 줬다고 가정
    sample_structured_command = {
        "intent": "modify_climate",
        "parameters": {"variable": "temperature", "value": "+2"}
    }

    # 2. 현재 게임 상태가 이렇다고 가정
    sample_game_state = {
        "money": 1000,
        "turn": 3,
        "climate_history": [
            {"month": 1, "temperature": 18, "precipitation": 80}, # 1개월 전
            {"month": 2, "temperature": 20, "precipitation": 85}, # 2개월 전
        ]
    }

    # 3. 번역기 함수 호출
    climate_input, yield_input = prepare_ml_inputs(sample_structured_command, sample_game_state)

    print("--- LLM 명령 번역 결과 ---")
    print("\n[기후 예측 모델에 전달될 데이터 (현재 턴의 기후)]: ")
    print(json.dumps(climate_input, indent=4, ensure_ascii=False))

    print("\n[생산량 예측 모델에 전달될 데이터 (최근 3개월)]: ")
    print(json.dumps(yield_input, indent=4, ensure_ascii=False))