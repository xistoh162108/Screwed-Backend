import pandas as pd

# ===================================================================
# 0. 데이터 준비 (ML_src.csv 파일 불러오기 및 전처리)
# ===================================================================
print("--- 0. 데이터 준비 시작 ---")

file_path = 'Screwed-Backend/data/ML_src.csv'
df = pd.read_csv(file_path)
print(df.head())


# 0) 타겟 작물 지정.
target_crop = 'SOYBEAN'

# 1) 수확량 > 0 인 (수확 시점) 표본에 영향을 주는 행만을 따로 추출한 테이블 생성.
harvest_rows = df.loc[df[target_crop] > 0, ['LAT', 'LON', 'YEAR', 'MONTH']].rename(columns={'YEAR': 'YEAR_t', 'MONTH': 'MONTH_t'})
merged = df.merge(harvest_rows.drop_duplicates(), on=['LAT', 'LON'], how='inner')
print(merged.head())
merged['month_diff'] = (merged['YEAR_t'] - merged['YEAR']) * 12 + (merged['MONTH_t'] - merged['MONTH'])

df1 = merged[(merged['month_diff'] >= 0) & (merged['month_diff'] <= 3)].reset_index(drop=True)
print(df1.head())
# 2) 평탄화 진행
keys = ['CODE', 'LAT', 'LON', 'YEAR_t', 'MONTH_t']

climate_cols = [
    'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
    'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
    'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
    'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
    'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
]

df1 = df1[df1['month_diff'].between(0, 2)]

agg_cols = climate_cols + [target_crop]
df1_agg = df1.groupby(keys + ['month_diff'], as_index=False)[agg_cols].mean()

wide = df1_agg.set_index(keys + ['month_diff'])[agg_cols].unstack('month_diff')

wide.columns = [f"{var}_{mdiff}" for var, mdiff in wide.columns]
result = wide.reset_index()
result = result.drop(columns=['CODE', 'LAT', 'LON', 'YEAR_t', 'MONTH_t',f'{target_crop}_1',f'{target_crop}_2'], errors='ignore')
result = result.rename(columns={f'{target_crop}_0': 'PpA'})

print(result.head())
# 3) csv로 저장
save_path = f"Screwed-Backend/data/3month_Climate_and_{target_crop}.csv"
result.to_csv(save_path, index=False, encoding='utf-8-sig')