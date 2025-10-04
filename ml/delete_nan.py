import pandas as pd

f_nonan = pd.read_csv('Screwed-Backend/data/MERGED_Climate.csv')
f_nonan = f_nonan.dropna()

f_nonan.to_csv('Screwed-Backend/data/NONAN_file.csv', index=False)