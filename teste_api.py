import requests, os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('BRAPI_TOKEN')
r = requests.get('https://brapi.dev/api/quote/HGLG11', params={'token': token, 'fundamental': 'true', 'dividends': 'true'})
print('Status:', r.status_code)
import json
data = r.json()
print(json.dumps(data, indent=2)[:3000])