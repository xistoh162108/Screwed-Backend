# Usage Examples - Economy, Crops & Sustainability

## 1. Session Economy API

### Create/Update Economy (PUT)

```bash
# 세션 경제 정보 생성 또는 전체 업데이트
curl -X PUT http://localhost:8000/api/v1/sessions/s_abc123/economy \
  -H "Content-Type: application/json" \
  -d '{
    "currency": "KRW",
    "balance": 1500000.00
  }'
```

### Partial Update Economy (PATCH)

```bash
# 잔액만 부분 업데이트
curl -X PATCH http://localhost:8000/api/v1/sessions/s_abc123/economy \
  -H "Content-Type: application/json" \
  -d '{
    "balance": 1777000.50
  }'
```

### Get Economy

```bash
curl http://localhost:8000/api/v1/sessions/s_abc123/economy
```

**Response:**
```json
{
  "id": "eco_a1b2c3d4",
  "session_id": "s_abc123",
  "currency": "KRW",
  "balance": "1777000.50",
  "updated_at": "2025-10-05T12:00:00Z"
}
```

### Delete Economy

```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/s_abc123/economy
```

---

## 2. Session Crop Stats API

### List All Crops for Session

```bash
curl http://localhost:8000/api/v1/sessions/s_abc123/crops
```

**Response:**
```json
[
  {
    "id": "crop_x1y2z3",
    "session_id": "s_abc123",
    "crop": "MAIZE",
    "cumulative_production_tonnes": "12.345",
    "co2e_kg": "100.500",
    "water_m3": "80.000",
    "fert_n_kg": "5.200",
    "fert_p_kg": "1.100",
    "fert_k_kg": "2.200",
    "last_event_at": "2025-10-05T12:00:00Z"
  },
  {
    "id": "crop_a4b5c6",
    "session_id": "s_abc123",
    "crop": "WHEAT",
    "cumulative_production_tonnes": "8.500",
    "co2e_kg": "75.300",
    "water_m3": "65.000",
    "fert_n_kg": "4.100",
    "fert_p_kg": "0.900",
    "fert_k_kg": "1.800",
    "last_event_at": "2025-10-05T11:30:00Z"
  }
]
```

### Get Single Crop Stats

```bash
curl http://localhost:8000/api/v1/sessions/s_abc123/crops/MAIZE
```

### Update Crop Stats (Absolute Mode)

```bash
# 절대값 모드: 값을 덮어씀
curl -X PUT http://localhost:8000/api/v1/sessions/s_abc123/crops/MAIZE \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "MAIZE",
    "mode": "absolute",
    "cumulative_production_tonnes": 12.345,
    "co2e_kg": 100.5,
    "water_m3": 80.0,
    "fert_n_kg": 5.2,
    "fert_p_kg": 1.1,
    "fert_k_kg": 2.2
  }'
```

### Update Crop Stats (Delta Mode)

```bash
# 증분 모드: 기존 값에 더함
curl -X PATCH http://localhost:8000/api/v1/sessions/s_abc123/crops/MAIZE \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "MAIZE",
    "mode": "delta",
    "cumulative_production_tonnes": 2.000,
    "co2e_kg": 10.0,
    "water_m3": 6.0,
    "fert_n_kg": 0.5,
    "fert_p_kg": 0.2,
    "fert_k_kg": 0.3
  }'
```

### Batch Update Multiple Crops

```bash
curl -X PUT http://localhost:8000/api/v1/sessions/s_abc123/crops:batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "crop": "MAIZE",
      "mode": "delta",
      "cumulative_production_tonnes": 1.5,
      "co2e_kg": 8.0,
      "water_m3": 5.0,
      "fert_n_kg": 0.3,
      "fert_p_kg": 0.1,
      "fert_k_kg": 0.2
    },
    {
      "crop": "WHEAT",
      "mode": "delta",
      "cumulative_production_tonnes": 1.2,
      "co2e_kg": 6.5,
      "water_m3": 4.2,
      "fert_n_kg": 0.25,
      "fert_p_kg": 0.08,
      "fert_k_kg": 0.15
    }
  ]'
```

### Delete Crop Stats

```bash
# 특정 작물 통계 삭제
curl -X DELETE http://localhost:8000/api/v1/sessions/s_abc123/crops/MAIZE

# 모든 작물 통계 삭제
curl -X DELETE http://localhost:8000/api/v1/sessions/s_abc123/crops
```

---

## 3. Sustainability Scoring (Python)

### Basic Usage

```python
from app.services.sustainability_scorer import (
    ClimateData,
    calculate_sustainability_score
)

# 1. NASA 28개 변수로 ClimateData 생성
climate = ClimateData(
    # Radiation
    ALLSKY_SFC_LW_DWN=350.5,
    ALLSKY_SFC_PAR_TOT=500.2,
    ALLSKY_SFC_SW_DIFF=120.3,
    ALLSKY_SFC_SW_DNI=850.1,
    ALLSKY_SFC_SW_DWN=220.4,
    ALLSKY_SFC_UVA=15.5,
    ALLSKY_SFC_UVB=2.3,
    ALLSKY_SFC_UV_INDEX=5.2,

    # Surface
    ALLSKY_SRF_ALB=0.18,
    CLOUD_AMT=0.35,
    TS=22.5,

    # Clear sky
    CLRSKY_SFC_PAR_TOT=580.3,
    CLRSKY_SFC_SW_DWN=280.5,

    # Soil moisture
    GWETPROF=0.62,
    GWETROOT=0.58,
    GWETTOP=0.55,

    # Precipitation
    PRECTOTCORR=3.5,
    PRECTOTCORR_SUM=105.2,

    # Pressure & Humidity
    PS=101.3,
    QV2M=12.5,
    RH2M=65.0,
    T2MDEW=18.2,

    # Temperature
    T2M=25.3,
    T2MWET=23.1,
    T2M_MAX=32.5,
    T2M_MIN=18.7,
    T2M_RANGE=13.8,
    TOA_SW_DWN=450.2,
)

# 2. 지속가능성 점수 계산
result = calculate_sustainability_score(
    data=climate,
    crop="MAIZE",
    production_tonnes=12.5,
    money_balance=1500000
)

# 3. 결과 확인
print(f"Overall Score: {result.overall_score}")  # 82.35
print(f"Grade: {result.grade}")  # A

print(f"\nComponents:")
print(f"  Water Stress: {result.components.water_stress}")
print(f"  Heat Stress: {result.components.heat_stress}")
print(f"  Radiation: {result.components.radiation_score}")
print(f"  Surface: {result.components.surface_score}")
print(f"  Air/UV: {result.components.air_uv_score}")

print(f"\nIndicators:")
for key, value in result.indicators.items():
    print(f"  {key}: {value:.2f}")

print(f"\nMetadata:")
print(f"  Baseline: {result.metadata['baseline_score']:.2f}")
print(f"  Efficiency Bonus: {result.metadata['efficiency_bonus']:.2f}")
print(f"  Money Bonus: {result.metadata['money_bonus']:.2f}")
```

**Output:**
```
Overall Score: 82.35
Grade: A

Components:
  Water Stress: 78.50
  Heat Stress: 85.20
  Radiation: 88.40
  Surface: 81.30
  Air/UV: 79.60

Indicators:
  vpd: 1.23
  soil_moisture_avg: 0.58
  temp_deviation_from_optimal: 0.30
  par_efficiency: 0.86
  cloud_coverage: 0.35
  precipitation_adequacy: 100.00

Metadata:
  Baseline: 80.12
  Efficiency Bonus: 1.53
  Money Bonus: 0.70
```

### Aggregate Time Series Data

```python
from app.services.sustainability_scorer import aggregate_climate_window

# 일별 기후 데이터 (30일치)
daily_data = [
    {
        "T2M": 24.5,
        "RH2M": 62.0,
        "PRECTOTCORR": 2.5,
        # ... 다른 25개 변수
    },
    # ... 29 more days
]

# 30일 윈도우로 집계
climate = aggregate_climate_window(daily_data, window_days=30)

# 점수 계산
result = calculate_sustainability_score(data=climate, crop="RICE")
```

### Batch Calculation for Multiple Crops

```python
from app.services.sustainability_scorer import calculate_batch_sustainability

# 작물별 생산량
production_data = {
    "MAIZE": 12.5,
    "WHEAT": 8.3,
    "SOYBEAN": 5.2,
    "RICE": 10.1,
}

# 배치 계산
results = calculate_batch_sustainability(
    data=climate,
    crops=["MAIZE", "WHEAT", "SOYBEAN", "RICE"],
    production_data=production_data,
    money_balance=1500000
)

# 결과 확인
for crop, result in results.items():
    print(f"{crop}: {result.overall_score:.1f} ({result.grade})")
```

**Output:**
```
MAIZE: 82.3 (A)
WHEAT: 78.5 (B)
SOYBEAN: 85.1 (A)
RICE: 88.2 (A)
```

### Custom Weights

```python
# 커스텀 가중치 사용 (물 스트레스 중요도 증가)
custom_weights = {
    "water": 0.35,      # 기본 0.25 → 0.35
    "heat": 0.20,
    "radiation": 0.15,  # 기본 0.20 → 0.15
    "surface": 0.20,
    "air_uv": 0.10,     # 기본 0.15 → 0.10
}

result = calculate_sustainability_score(
    data=climate,
    crop="MAIZE",
    weights=custom_weights,
    production_tonnes=12.5,
    money_balance=1500000
)
```

---

## 4. Complete Workflow Example

```python
from app.db.session import get_db
from app.services import economy_service, crop_stats_service
from app.services.sustainability_scorer import (
    ClimateData,
    calculate_sustainability_score
)
from app.schemas.crop_stats import SessionCropStatsUpsert

# 1. 경제 정보 조회
db = next(get_db())
economy = economy_service.get(db, "s_abc123")
money_balance = float(economy.balance) if economy else 0

# 2. 작물 통계 조회
crop_stats = crop_stats_service.list_(db, "s_abc123")
production_data = {
    stat.crop: float(stat.cumulative_production_tonnes)
    for stat in crop_stats
}

# 3. 최신 기후 데이터 로드 (예시)
climate = ClimateData(
    # ... NASA 28 variables
)

# 4. 지속가능성 점수 계산
result = calculate_sustainability_score(
    data=climate,
    crop="MAIZE",
    production_tonnes=production_data.get("MAIZE", 0),
    money_balance=money_balance
)

# 5. 작물 통계 업데이트 (새로운 수확)
new_harvest = SessionCropStatsUpsert(
    crop="MAIZE",
    mode="delta",
    cumulative_production_tonnes=2.5,  # +2.5 톤 수확
    co2e_kg=15.0,  # +15kg CO2e 배출
    water_m3=10.0,  # +10m³ 물 사용
    fert_n_kg=1.2,
    fert_p_kg=0.4,
    fert_k_kg=0.6,
)
crop_stats_service.upsert_one(db, "s_abc123", new_harvest)

# 6. 비용 차감 (경제 업데이트)
from app.schemas.economy import SessionEconomyUpdate
economy_update = SessionEconomyUpdate(
    balance=money_balance - 50000  # -50,000원 비용
)
economy_service.upsert(db, "s_abc123", economy_update)

print(f"Sustainability Score: {result.overall_score} ({result.grade})")
print(f"New Balance: {money_balance - 50000:,.0f} KRW")
```

---

## 5. API Integration Example (Full Game Turn)

```bash
#!/bin/bash

SESSION_ID="s_abc123"
API_URL="http://localhost:8000/api/v1"

# 1. 턴 시작 - 현재 상태 조회
echo "=== 현재 경제 상태 ==="
curl -s $API_URL/sessions/$SESSION_ID/economy | jq

echo "\n=== 현재 작물 통계 ==="
curl -s $API_URL/sessions/$SESSION_ID/crops | jq

# 2. 플레이어 명령 처리 (예: "옥수수 2톤 수확")
# ... command processing ...

# 3. 작물 통계 업데이트
echo "\n=== 작물 통계 업데이트 (수확) ==="
curl -s -X PATCH $API_URL/sessions/$SESSION_ID/crops/MAIZE \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "MAIZE",
    "mode": "delta",
    "cumulative_production_tonnes": 2.0,
    "co2e_kg": 12.5,
    "water_m3": 8.0,
    "fert_n_kg": 0.8,
    "fert_p_kg": 0.3,
    "fert_k_kg": 0.5
  }' | jq

# 4. 경제 업데이트 (수확 수익 - 비용)
REVENUE=150000
COST=50000
NET=$(($REVENUE - $COST))

echo "\n=== 경제 업데이트 (잔액 증가) ==="
CURRENT_BALANCE=$(curl -s $API_URL/sessions/$SESSION_ID/economy | jq -r '.balance')
NEW_BALANCE=$(echo "$CURRENT_BALANCE + $NET" | bc)

curl -s -X PATCH $API_URL/sessions/$SESSION_ID/economy \
  -H "Content-Type: application/json" \
  -d "{\"balance\": $NEW_BALANCE}" | jq

echo "\n=== 턴 완료 ==="
```

---

## 6. Grade Thresholds

| Grade | Score Range | Description |
|-------|-------------|-------------|
| S | ≥90 | 최고 등급: 매우 지속가능한 농업 |
| A | 80-89 | 우수: 지속가능성 높음 |
| B | 65-79 | 양호: 개선 여지 있음 |
| C | 50-64 | 보통: 주의 필요 |
| D | <50 | 미흡: 즉각적인 개선 필요 |

---

## 7. Sustainability Components Explanation

### Water Stress (25% weight)
- 토양 수분 (GWET*)
- 강수량 (PRECTOTCORR_SUM)
- 증기압차 (VPD)

### Heat Stress (20% weight)
- 평균 온도 적합성 (T2M)
- 일교차 (T2M_RANGE)
- 극한 온도 (T2M_MAX/MIN)

### Radiation (20% weight)
- 광합성 유효 방사선 (PAR)
- 전천일사량 (SW_DWN)
- 구름량 (CLOUD_AMT)

### Surface (20% weight)
- 지표면 온도 (TS)
- 반사율 (Albedo)
- 토양 수분 균형 (GWETPROF)

### Air & UV (15% weight)
- UV 지수 (UV_INDEX)
- 상대습도 (RH2M)
- 기압 (PS)
