import requests, os, json
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("BRAPI_TOKEN", "")
headers = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://brapi.dev/api/v2/fii"

# HGLG11 é fundo de tijolo — o mais provável de ter vacância
for simbolo in ["HGLG11", "XPML11", "VISC11"]:
    print(f"\n=== {simbolo} — indicadores atuais ===")
    r = requests.get(f"{BASE}/indicators",
                     headers=headers,
                     params={"symbols": simbolo},
                     timeout=30)
    if r.status_code == 200:
        data = r.json()
        fii = data.get("fiis", [{}])[0]
        print("Todos os campos:")
        for k, v in fii.items():
            if 'vacan' in k.lower() or 'vaga' in k.lower() or 'ocup' in k.lower():
                print(f"  *** VACÂNCIA: {k} = {v}")
        print(json.dumps({k: v for k, v in fii.items()
                          if any(x in k.lower() for x in ['vacan','vaga','ocup','vacant'])},
                         indent=2, ensure_ascii=False) or "  Nenhum campo de vacância encontrado")
        print(f"  Campos disponíveis: {list(fii.keys())}")
    else:
        print(f"Erro: {r.status_code}")