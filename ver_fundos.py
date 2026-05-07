import pandas as pd
df = pd.read_excel('dataset_fiis_2019_2024_brapi_v2.xlsx')
result = df[['Sigla','Tipo_do_Fundo','Segmento']].drop_duplicates('Sigla').sort_values(['Tipo_do_Fundo','Segmento','Sigla'])
print(result.to_string(index=False))
print(f"\nTotal: {result['Sigla'].nunique()} fundos")