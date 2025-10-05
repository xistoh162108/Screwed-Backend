# app/services/sustainability_scorer.py
"""
지속가능성 점수 계산 시스템
NASA 28개 기후 변수를 기반으로 환경 적합성 및 스트레스 관리 점수 산출
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================
# 1. 상수 및 설정
# ============================================

class SustainabilityGrade(str, Enum):
    S = "S"  # ≥90
    A = "A"  # ≥80
    B = "B"  # ≥65
    C = "C"  # ≥50
    D = "D"  # <50

# 기본 가중치 (총합 = 1.0)
DEFAULT_WEIGHTS = {
    "water": 0.25,      # 수분 스트레스
    "heat": 0.20,       # 온도 스트레스
    "radiation": 0.20,  # 광합성 가용성
    "surface": 0.20,    # 토양/표면 상태
    "air_uv": 0.15,     # 대기질/UV
}

# 등급 컷라인
GRADE_CUTOFFS = {
    "S": 90,
    "A": 80,
    "B": 65,
    "C": 50,
    "D": 0,
}

# 작물별 최적 온도 범위 (°C)
CROP_OPTIMAL_TEMP = {
    "MAIZE": {"min": 18, "optimal": 25, "max": 32},
    "RICE": {"min": 20, "optimal": 28, "max": 35},
    "SOYBEAN": {"min": 20, "optimal": 26, "max": 32},
    "WHEAT": {"min": 12, "optimal": 20, "max": 28},
}

# ============================================
# 2. 데이터 클래스
# ============================================

@dataclass
class ClimateData:
    """28개 NASA 기후 변수"""
    # Radiation (7 variables)
    ALLSKY_SFC_LW_DWN: float
    ALLSKY_SFC_PAR_TOT: float
    ALLSKY_SFC_SW_DIFF: float
    ALLSKY_SFC_SW_DNI: float
    ALLSKY_SFC_SW_DWN: float
    ALLSKY_SFC_UVA: float
    ALLSKY_SFC_UVB: float
    ALLSKY_SFC_UV_INDEX: float

    # Surface (3 variables)
    ALLSKY_SRF_ALB: float
    CLOUD_AMT: float
    TS: float

    # Clear sky (2 variables)
    CLRSKY_SFC_PAR_TOT: float
    CLRSKY_SFC_SW_DWN: float

    # Soil moisture (3 variables)
    GWETPROF: float
    GWETROOT: float
    GWETTOP: float

    # Precipitation (2 variables)
    PRECTOTCORR: float
    PRECTOTCORR_SUM: float

    # Pressure & Humidity (4 variables)
    PS: float
    QV2M: float
    RH2M: float
    T2MDEW: float

    # Temperature (6 variables)
    T2M: float
    T2MWET: float
    T2M_MAX: float
    T2M_MIN: float
    T2M_RANGE: float
    TOA_SW_DWN: float


@dataclass
class SustainabilityComponents:
    """지속가능성 하위 점수"""
    water_stress: float  # 0-100
    heat_stress: float   # 0-100
    radiation_score: float  # 0-100
    surface_score: float  # 0-100
    air_uv_score: float  # 0-100


@dataclass
class SustainabilityResult:
    """지속가능성 평가 결과"""
    overall_score: float  # 0-100
    grade: SustainabilityGrade
    components: SustainabilityComponents
    indicators: Dict[str, float]  # 세부 지표
    metadata: Dict[str, Any]  # 계산 메타데이터


# ============================================
# 3. 파생 지표 계산 함수
# ============================================

def calculate_vpd_like(t2m: float, rh2m: float) -> float:
    """
    증기압차(VPD) 근사치 계산
    VPD = (1 - RH/100) * 0.611 * exp(17.27 * T / (T + 237.3))
    """
    if rh2m < 0 or rh2m > 100:
        rh2m = np.clip(rh2m, 0, 100)

    svp = 0.611 * np.exp(17.27 * t2m / (t2m + 237.3))  # kPa
    vpd = (1 - rh2m / 100) * svp
    return vpd


def calculate_water_stress_indicator(data: ClimateData) -> float:
    """
    수분 스트레스 지표 계산 (0-100, 높을수록 좋음)
    고려 요소:
    - 토양 수분 (GWETROOT, GWETTOP, GWETPROF)
    - 강수량 (PRECTOTCORR_SUM)
    - VPD (증기압차)
    """
    # 1. 토양 수분 (0-1 범위, 0.4-0.8이 최적)
    soil_moisture = (data.GWETROOT + data.GWETTOP + data.GWETPROF) / 3
    soil_score = 100 * (1 - abs(soil_moisture - 0.6) / 0.6)  # 0.6이 최적
    soil_score = np.clip(soil_score, 0, 100)

    # 2. 강수량 (window 기간 누적, mm)
    # 가정: 월 60-150mm가 최적 범위
    precip = data.PRECTOTCORR_SUM
    if precip < 60:
        precip_score = (precip / 60) * 100
    elif precip > 150:
        precip_score = max(0, 100 - (precip - 150) / 2)
    else:
        precip_score = 100

    # 3. VPD (kPa, 0.5-1.5가 최적)
    vpd = calculate_vpd_like(data.T2M, data.RH2M)
    if vpd < 0.5:
        vpd_score = (vpd / 0.5) * 100
    elif vpd > 1.5:
        vpd_score = max(0, 100 - (vpd - 1.5) * 40)
    else:
        vpd_score = 100

    # 가중 평균
    water_score = 0.4 * soil_score + 0.35 * precip_score + 0.25 * vpd_score
    return np.clip(water_score, 0, 100)


def calculate_heat_stress_indicator(data: ClimateData, crop: str = "MAIZE") -> float:
    """
    온도 스트레스 지표 계산 (0-100, 높을수록 좋음)
    """
    optimal = CROP_OPTIMAL_TEMP.get(crop, CROP_OPTIMAL_TEMP["MAIZE"])

    # 1. 평균 온도 적합성
    t_mean = data.T2M
    if t_mean < optimal["min"]:
        temp_score = max(0, 100 - (optimal["min"] - t_mean) * 10)
    elif t_mean > optimal["max"]:
        temp_score = max(0, 100 - (t_mean - optimal["max"]) * 10)
    else:
        # 최적 범위 내: 중심에 가까울수록 높은 점수
        distance_from_optimal = abs(t_mean - optimal["optimal"])
        temp_score = 100 - (distance_from_optimal / (optimal["max"] - optimal["optimal"])) * 20

    # 2. 일교차 (T2M_RANGE, 10-15°C가 최적)
    range_val = data.T2M_RANGE
    if 10 <= range_val <= 15:
        range_score = 100
    elif range_val < 10:
        range_score = (range_val / 10) * 100
    else:
        range_score = max(0, 100 - (range_val - 15) * 5)

    # 3. 극한 온도 체크
    if data.T2M_MAX > optimal["max"] + 5:
        extreme_penalty = min(30, (data.T2M_MAX - optimal["max"] - 5) * 5)
    elif data.T2M_MIN < optimal["min"] - 5:
        extreme_penalty = min(30, (optimal["min"] - 5 - data.T2M_MIN) * 5)
    else:
        extreme_penalty = 0

    heat_score = 0.6 * temp_score + 0.4 * range_score - extreme_penalty
    return np.clip(heat_score, 0, 100)


def calculate_radiation_indicator(data: ClimateData) -> float:
    """
    광합성 가용 방사선 지표 (0-100, 높을수록 좋음)
    """
    # 1. PAR (Photosynthetically Active Radiation)
    # 가정: 400-600 W/m²가 최적
    par = data.ALLSKY_SFC_PAR_TOT
    if 400 <= par <= 600:
        par_score = 100
    elif par < 400:
        par_score = (par / 400) * 100
    else:
        par_score = max(80, 100 - (par - 600) / 10)

    # 2. 전천일사량 (SW_DWN)
    sw_dwn = data.ALLSKY_SFC_SW_DWN
    if 150 <= sw_dwn <= 300:
        sw_score = 100
    elif sw_dwn < 150:
        sw_score = (sw_dwn / 150) * 100
    else:
        sw_score = max(80, 100 - (sw_dwn - 300) / 20)

    # 3. 구름량 (CLOUD_AMT, 0-1, 낮을수록 좋음)
    cloud_score = (1 - data.CLOUD_AMT) * 100

    # 가중 평균
    radiation_score = 0.4 * par_score + 0.4 * sw_score + 0.2 * cloud_score
    return np.clip(radiation_score, 0, 100)


def calculate_surface_indicator(data: ClimateData) -> float:
    """
    표면/토양 상태 지표 (0-100)
    """
    # 1. 지표면 온도 (TS)
    # 가정: 15-30°C가 최적
    ts = data.TS
    if 15 <= ts <= 30:
        ts_score = 100
    elif ts < 15:
        ts_score = max(0, (ts / 15) * 100)
    else:
        ts_score = max(0, 100 - (ts - 30) * 3)

    # 2. 반사율 (Albedo, 0.15-0.25가 건강한 작물)
    albedo = data.ALLSKY_SRF_ALB
    if 0.15 <= albedo <= 0.25:
        albedo_score = 100
    else:
        albedo_score = max(0, 100 - abs(albedo - 0.20) * 200)

    # 3. 토양 수분 균형 (GWETPROF)
    gwet_score = (1 - abs(data.GWETPROF - 0.6) / 0.6) * 100
    gwet_score = np.clip(gwet_score, 0, 100)

    surface_score = 0.35 * ts_score + 0.30 * albedo_score + 0.35 * gwet_score
    return np.clip(surface_score, 0, 100)


def calculate_air_uv_indicator(data: ClimateData) -> float:
    """
    대기질 및 UV 지표 (0-100)
    """
    # 1. UV Index (3-6이 적정, 너무 높으면 스트레스)
    uv = data.ALLSKY_SFC_UV_INDEX
    if 3 <= uv <= 6:
        uv_score = 100
    elif uv < 3:
        uv_score = (uv / 3) * 90  # 약간 감점
    else:
        uv_score = max(50, 100 - (uv - 6) * 10)

    # 2. 상대습도 (40-70%가 최적)
    rh = data.RH2M
    if 40 <= rh <= 70:
        rh_score = 100
    elif rh < 40:
        rh_score = (rh / 40) * 100
    else:
        rh_score = max(60, 100 - (rh - 70) * 2)

    # 3. 기압 (PS, 정상 범위 체크)
    # 가정: 95-105 kPa가 정상
    ps = data.PS
    if 95 <= ps <= 105:
        ps_score = 100
    else:
        ps_score = max(80, 100 - abs(ps - 100) * 5)

    air_score = 0.5 * uv_score + 0.3 * rh_score + 0.2 * ps_score
    return np.clip(air_score, 0, 100)


# ============================================
# 4. 메인 점수 계산 함수
# ============================================

def calculate_sustainability_score(
    data: ClimateData,
    crop: str = "MAIZE",
    weights: Optional[Dict[str, float]] = None,
    production_tonnes: float = 0,
    money_balance: float = 0,
) -> SustainabilityResult:
    """
    지속가능성 종합 점수 계산

    Args:
        data: 28개 NASA 기후 변수
        crop: 작물 종류 (MAIZE, RICE, SOYBEAN, WHEAT)
        weights: 커스텀 가중치 (없으면 기본값 사용)
        production_tonnes: 누적 생산량 (효율성 보정용)
        money_balance: 현재 잔액 (경제성 보정용)

    Returns:
        SustainabilityResult
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 1. 각 카테고리별 점수 계산
    water_stress = calculate_water_stress_indicator(data)
    heat_stress = calculate_heat_stress_indicator(data, crop)
    radiation_score = calculate_radiation_indicator(data)
    surface_score = calculate_surface_indicator(data)
    air_uv_score = calculate_air_uv_indicator(data)

    # 2. 가중 평균으로 baseline 점수 계산
    baseline_score = (
        weights["water"] * water_stress +
        weights["heat"] * heat_stress +
        weights["radiation"] * radiation_score +
        weights["surface"] * surface_score +
        weights["air_uv"] * air_uv_score
    )

    # 3. 효율성 보정 (생산량 기반)
    # 생산량이 많을수록 약간의 보너스 (최대 +5점)
    if production_tonnes > 0:
        efficiency_bonus = min(5, np.log1p(production_tonnes) * 0.5)
    else:
        efficiency_bonus = 0

    # 4. 경제성 보정 (잔액 기반)
    # 잔액이 양수면 약간의 보너스, 음수면 페널티 (최대 ±3점)
    if money_balance > 0:
        money_bonus = min(3, np.log1p(money_balance / 10000) * 0.3)
    else:
        money_bonus = max(-3, money_balance / 100000 * 3)

    # 5. 최종 점수
    overall_score = baseline_score + efficiency_bonus + money_bonus
    overall_score = np.clip(overall_score, 0, 100)

    # 6. 등급 결정
    grade = _determine_grade(overall_score)

    # 7. 세부 지표
    indicators = {
        "vpd": calculate_vpd_like(data.T2M, data.RH2M),
        "soil_moisture_avg": (data.GWETROOT + data.GWETTOP + data.GWETPROF) / 3,
        "temp_deviation_from_optimal": abs(data.T2M - CROP_OPTIMAL_TEMP[crop]["optimal"]),
        "par_efficiency": data.ALLSKY_SFC_PAR_TOT / (data.CLRSKY_SFC_PAR_TOT + 1e-6),
        "cloud_coverage": data.CLOUD_AMT,
        "precipitation_adequacy": min(100, (data.PRECTOTCORR_SUM / 100) * 100),
    }

    # 8. 메타데이터
    metadata = {
        "crop": crop,
        "weights_used": weights,
        "baseline_score": baseline_score,
        "efficiency_bonus": efficiency_bonus,
        "money_bonus": money_bonus,
        "production_tonnes": production_tonnes,
        "money_balance": money_balance,
    }

    return SustainabilityResult(
        overall_score=round(overall_score, 2),
        grade=grade,
        components=SustainabilityComponents(
            water_stress=round(water_stress, 2),
            heat_stress=round(heat_stress, 2),
            radiation_score=round(radiation_score, 2),
            surface_score=round(surface_score, 2),
            air_uv_score=round(air_uv_score, 2),
        ),
        indicators=indicators,
        metadata=metadata,
    )


def _determine_grade(score: float) -> SustainabilityGrade:
    """점수를 등급으로 변환"""
    if score >= GRADE_CUTOFFS["S"]:
        return SustainabilityGrade.S
    elif score >= GRADE_CUTOFFS["A"]:
        return SustainabilityGrade.A
    elif score >= GRADE_CUTOFFS["B"]:
        return SustainabilityGrade.B
    elif score >= GRADE_CUTOFFS["C"]:
        return SustainabilityGrade.C
    else:
        return SustainabilityGrade.D


# ============================================
# 5. 시계열 데이터 처리 (window 집계)
# ============================================

def aggregate_climate_window(
    timeseries_data: List[Dict[str, float]],
    window_days: int = 30
) -> ClimateData:
    """
    시계열 데이터를 window 기간으로 집계

    Args:
        timeseries_data: 일별 기후 데이터 리스트
        window_days: 집계 기간 (일)

    Returns:
        ClimateData (집계된 평균값)
    """
    # 최근 window_days 데이터만 사용
    recent_data = timeseries_data[-window_days:] if len(timeseries_data) > window_days else timeseries_data

    if not recent_data:
        raise ValueError("No data available for aggregation")

    # 각 변수별 평균 계산
    aggregated = {}
    for key in ClimateData.__annotations__.keys():
        values = [d.get(key, np.nan) for d in recent_data]

        # 누적 변수는 합계 (PRECTOTCORR_SUM)
        if key == "PRECTOTCORR_SUM":
            aggregated[key] = np.nansum(values)
        else:
            aggregated[key] = np.nanmean(values)

    return ClimateData(**aggregated)


# ============================================
# 6. 배치 점수 계산 (여러 작물)
# ============================================

def calculate_batch_sustainability(
    data: ClimateData,
    crops: List[str],
    production_data: Dict[str, float],
    money_balance: float = 0,
) -> Dict[str, SustainabilityResult]:
    """
    여러 작물에 대한 지속가능성 점수를 한 번에 계산

    Args:
        data: 기후 데이터
        crops: 작물 리스트
        production_data: {crop: production_tonnes} 딕셔너리
        money_balance: 현재 잔액

    Returns:
        {crop: SustainabilityResult} 딕셔너리
    """
    results = {}
    for crop in crops:
        production = production_data.get(crop, 0)
        results[crop] = calculate_sustainability_score(
            data=data,
            crop=crop,
            production_tonnes=production,
            money_balance=money_balance,
        )
    return results
