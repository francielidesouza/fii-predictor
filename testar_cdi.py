import requests

r = requests.get(
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados",
    params={"formato":"json","dataInicial":"01/01/2019","dataFinal":"31/12/2024"},
    timeout=20
)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Total: {len(data)} meses")
print(f"Primeiros 3: {data[:3]}")
print(f"Últimos 3: {data[-3:]}")