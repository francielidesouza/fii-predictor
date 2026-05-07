import pandas as pd
df = pd.read_excel('dataset_fiis_2019_2024_brapi.xlsx')
mc = df[df['Segmento']=='Multicategoria'][['Sigla','Segmento','Tipo_do_Fundo']].drop_duplicates('Sigla')
print(mc.to_string(index=False))
print(f"\nTotal: {len(mc)} fundos")