# app/core/delta_rules.py
"""
델타 룰 설정: 농업 행동 → 환경 변수 변화 매핑
"""

# 델타 룰 상수
# 행동(intent) → 환경 변수 변화량 및 연산 방식
DELTA_RULES = {
    # 드립 관개 설치
    "irrigation=drip": {
        "GWETTOP": {"op": "+", "value": 0.02, "clamp": [0, 1]},
        "GWETROOT": {"op": "+", "value": 0.02, "clamp": [0, 1]},
        "GWETPROF": {"op": "+", "value": 0.01, "clamp": [0, 1]},
        "PRECTOTCORR": {"op": "*", "value": 0.98},  # 지표 유출 감소
        "CLOUD_AMT": {"op": "*", "value": 0.98, "clamp": [0, 1]},
    },

    # 스프링클러 관개
    "irrigation=sprinkler": {
        "GWETTOP": {"op": "+", "value": 0.03, "clamp": [0, 1]},
        "GWETROOT": {"op": "+", "value": 0.015, "clamp": [0, 1]},
        "GWETPROF": {"op": "+", "value": 0.01, "clamp": [0, 1]},
        "RH2M": {"op": "+", "value": 2.0},  # 상대습도 증가
    },

    # 야간 관개 전환
    "irrigation=night": {
        "TS": {"op": "*", "value": 0.98},  # 표면 과열 완화
        "T2M": {"op": "*", "value": 0.99},  # 약간의 냉각 효과
    },

    # 차광막 설치
    "shading=on": {
        "ALLSKY_SFC_SW_DWN": {"op": "*", "value": 0.97},
        "ALLSKY_SFC_PAR_TOT": {"op": "*", "value": 0.95},
        "ALLSKY_SFC_UV_INDEX": {"op": "*", "value": 0.95, "clamp": [0, None]},
        "TS": {"op": "*", "value": 0.98},
        "T2M_MAX": {"op": "*", "value": 0.97},
    },

    # 멀칭 (짚)
    "mulching=straw": {
        "TS": {"op": "*", "value": 0.97},
        "ALLSKY_SRF_ALB": {"op": "+", "value": 0.005, "clamp": [0, 1]},
        "GWETTOP": {"op": "+", "value": 0.02, "clamp": [0, 1]},
        "T2M_RANGE": {"op": "*", "value": 0.95},  # 일교차 감소
    },

    # 멀칭 (플라스틱)
    "mulching=plastic": {
        "TS": {"op": "*", "value": 1.02},  # 지온 상승
        "ALLSKY_SRF_ALB": {"op": "+", "value": 0.01, "clamp": [0, 1]},
        "GWETTOP": {"op": "+", "value": 0.03, "clamp": [0, 1]},
    },

    # 풍력차단막 설치
    "windbreak=installed": {
        "T2M": {"op": "*", "value": 1.005},  # 미세한 온도 상승
        "RH2M": {"op": "+", "value": 1.5},
    },

    # 관개 중단
    "irrigation=stop": {
        "GWETTOP": {"op": "*", "value": 0.95, "clamp": [0, 1]},
        "GWETROOT": {"op": "*", "value": 0.97, "clamp": [0, 1]},
    },

    # 작물 심기 (환경 변수 직접 변경 없음, 생산량 모델만 사용)
    "planting": {
        # 환경 변수 변화 없음
    },

    # 수확 (환경 변수 직접 변경 없음)
    "harvesting": {
        # 환경 변수 변화 없음
    },

    # 비료 살포 (N/P/K 변수가 없으므로 간접 효과만)
    "fertilizer=applied": {
        # 28개 환경 변수에는 직접 영향 없음
        # 지속가능성 점수에서 텍스트로만 반영
    },

    # 병해충 방제
    "pesticide=applied": {
        # 환경 변수 직접 변경 없음
    },
}


# 의도(intent) → 델타 룰 키 매핑
INTENT_TO_DELTA_KEY = {
    # 관개 관련
    "드립_관개": "irrigation=drip",
    "드립관개": "irrigation=drip",
    "점적관개": "irrigation=drip",
    "스프링클러": "irrigation=sprinkler",
    "살수": "irrigation=sprinkler",
    "야간관개": "irrigation=night",
    "관개중단": "irrigation=stop",
    "물주기중단": "irrigation=stop",

    # 피복 관련
    "차광막": "shading=on",
    "차광": "shading=on",
    "그늘막": "shading=on",
    "짚멀칭": "mulching=straw",
    "볏짚": "mulching=straw",
    "플라스틱멀칭": "mulching=plastic",
    "비닐멀칭": "mulching=plastic",

    # 기타
    "방풍망": "windbreak=installed",
    "풍력차단": "windbreak=installed",
    "심기": "planting",
    "파종": "planting",
    "수확": "harvesting",
    "비료": "fertilizer=applied",
    "비료살포": "fertilizer=applied",
    "농약": "pesticide=applied",
    "방제": "pesticide=applied",
}


# 변수별 물리적 범위 (클램핑용)
VARIABLE_PHYSICAL_LIMITS = {
    # 0-1 범위
    "CLOUD_AMT": [0, 1],
    "GWETPROF": [0, 1],
    "GWETROOT": [0, 1],
    "GWETTOP": [0, 1],
    "ALLSKY_SRF_ALB": [0, 1],

    # 0 이상
    "ALLSKY_SFC_LW_DWN": [0, None],
    "ALLSKY_SFC_PAR_TOT": [0, None],
    "ALLSKY_SFC_SW_DIFF": [0, None],
    "ALLSKY_SFC_SW_DNI": [0, None],
    "ALLSKY_SFC_SW_DWN": [0, None],
    "ALLSKY_SFC_UVA": [0, None],
    "ALLSKY_SFC_UVB": [0, None],
    "ALLSKY_SFC_UV_INDEX": [0, None],
    "CLRSKY_SFC_PAR_TOT": [0, None],
    "CLRSKY_SFC_SW_DWN": [0, None],
    "PRECTOTCORR": [0, None],
    "PRECTOTCORR_SUM": [0, None],
    "TOA_SW_DWN": [0, None],

    # 습도 0-100
    "RH2M": [0, 100],
    "QV2M": [0, None],

    # 온도 (물리적 하한 -100, 상한 60°C 가정)
    "T2M": [-100, 60],
    "T2MDEW": [-100, 60],
    "T2MWET": [-100, 60],
    "T2M_MAX": [-100, 70],
    "T2M_MIN": [-100, 60],
    "T2M_RANGE": [0, 80],
    "TS": [-100, 80],

    # 기압 (kPa, 정상 범위 80-110)
    "PS": [80, 110],
}


# 작물별 기본 파라미터
CROP_DEFAULTS = {
    "MAIZE": {
        "name_kr": "옥수수",
        "optimal_temp": 25,
        "water_demand": "medium-high",
    },
    "RICE": {
        "name_kr": "쌀",
        "optimal_temp": 28,
        "water_demand": "very-high",
    },
    "SOYBEAN": {
        "name_kr": "콩",
        "optimal_temp": 26,
        "water_demand": "medium",
    },
    "WHEAT": {
        "name_kr": "밀",
        "optimal_temp": 20,
        "water_demand": "medium",
    },
}


# 한글 작물명 → 영문 매핑
CROP_NAME_MAPPING = {
    "옥수수": "MAIZE",
    "corn": "MAIZE",
    "maize": "MAIZE",
    "쌀": "RICE",
    "벼": "RICE",
    "rice": "RICE",
    "콩": "SOYBEAN",
    "대두": "SOYBEAN",
    "soybean": "SOYBEAN",
    "밀": "WHEAT",
    "wheat": "WHEAT",
}
