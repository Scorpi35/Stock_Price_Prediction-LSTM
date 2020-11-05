import pandas as pd

# Creating Intial_Data.csv from Trained_Data.csv
df = pd.read_csv('Initial_Data.csv')
keep_col = ['Date', 'Open']
new_f = df[keep_col]
new_f.to_csv('Initial_Data.csv', index=False)


