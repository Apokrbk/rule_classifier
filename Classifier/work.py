import pandas as pd
df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/work1.csv',
                 encoding='utf-8', delimiter=';')
numeric_cols = df._get_numeric_data().columns
char_cols = df.select_dtypes('object').columns
print(numeric_cols)
print(char_cols)
for i in range(0, len(numeric_cols)):
    avg = df[numeric_cols[i]].mean()
    df[numeric_cols[i]] = df[numeric_cols[i]].fillna(value=avg)
for i in range(0, len(char_cols)):
    df[char_cols[i]] = df[char_cols[i]].fillna(value='null')
print(df)