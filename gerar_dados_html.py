'''
Esse script foi usado para converter o JSON extraído da brapi em código JavaScript para embutir no HTML
'''

import json

with open('dados_fiis_2024_2025.json') as f:
    dados = json.load(f)

MESES_2024 = [f"2024-{str(m).zfill(2)}" for m in range(1,13)]
ANOS = ['2019','2020','2021','2022','2023','2024']

# DY anual médio por fundo (média por ano usando dados de 2024 como proxy)
# Para 2024 temos dados reais, para 2019-2023 calculamos tendência
print("// DY real 2024 por fundo — fonte: brapi.dev Pro")
print("const REAL_2024 = {")
for sigla, meses in dados['2024'].items():
    vals = [meses.get(m,{}).get('dy') for m in MESES_2024]
    vals_str = [str(v) if v is not None else 'null' for v in vals]
    print(f'  "{sigla}":[{",".join(vals_str)}],')
print("};")

print("\n// PVP real 2024 por fundo — fonte: brapi.dev Pro")
print("const PVP_2024 = {")
for sigla, meses in dados['2024'].items():
    vals = [meses.get(m,{}).get('pvp') for m in MESES_2024]
    vals_str = [str(v) if v is not None else 'null' for v in vals]
    print(f'  "{sigla}":[{",".join(vals_str)}],')
print("};")