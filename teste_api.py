# import requests

# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# # Yahoo Finance — histórico de cotação HGLG11
# url = 'https://query1.finance.yahoo.com/v8/finance/chart/HGLG11.SA'
# params = {
#     'interval': '1mo',
#     'range': '6y',
#     'events': 'dividends'
# }
# r = requests.get(url, params=params, headers=headers, timeout=15)
# print('Status:', r.status_code)
# import json
# data = r.json()
# # Mostra dividendos
# events = data.get('chart', {}).get('result', [{}])[0].get('events', {})
# divs = events.get('dividends', {})
# print('Dividendos encontrados:', len(divs))
# print('Amostra:', list(divs.items())[:3])
# # Mostra cotações
# meta = data.get('chart', {}).get('result', [{}])[0].get('meta', {})
# print('Moeda:', meta.get('currency'))
# print('Símbolo:', meta.get('symbol'))

sed -i 's/dt\.to_period("M")\.astype(str)/str(dt.to_period("M"))/g' montar_dataset.py