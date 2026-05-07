import pandas as pd
df = pd.read_excel('dataset_fiis_2019_2024_brapi_v2.xlsx')
df = df[df['Tipo_do_Fundo'].isin(['Tijolo','Papel'])]
print(df.groupby(['Tipo_do_Fundo','Segmento'])['Sigla'].nunique().to_string())