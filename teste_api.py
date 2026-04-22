import requests

headers = {'User-Agent': 'Mozilla/5.0'}

# Testa busca de proventos do HGLG11 no Status Invest
url = 'https://statusinvest.com.br/fundo-imobiliario/companytickerprovents'
params = {
    'ticker': 'HGLG11',
    'chartprovents': 'false',
    'startat': '2019-01-01',
    'endat': '2024-12-31'
}
r = requests.get(url, params=params, headers=headers, timeout=15)
print('Status:', r.status_code)
print('Resposta:', r.text[:2000])