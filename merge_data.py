import pandas as pd

f_climate = pd.read_csv('Screwed-Backend/data/NONAN_file.csv')
f_crop = pd.read_csv('Screwed-Backend/data/MERGED_CropsPpA.csv')

f_merged = pd.merge(f_climate, f_crop, on=['CODE', 'YEAR', 'MONTH'], how='left')
f_merged.fillna(0, inplace=True)

f_merged.to_csv('Screwed-Backend/data/MERGED_NONAN_file.csv', index=False)