# app/services/command_utils.py
from __future__ import annotations
import json
from typing import Optional

def extract_user_text_from_command(cmd) -> Optional[str]:
    """
    Command 스키마 기준:
      - text: 필수 문자열(최우선 사용)
      - payload: dict | None (보조 정보)
    """
    # 1) 최우선: text
    if getattr(cmd, "text", None):
        return cmd.text

    # 2) 보조: payload에서 text/user_input 키를 시도
    payload = getattr(cmd, "payload", None)
    if payload is None:
        return None

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            # 문자열 그대로는 구조를 모르므로 반환 불가
            return None

    if isinstance(payload, dict):
        return payload.get("text") or payload.get("user_input")

    return None