import requests, os, json, time
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("BRAPI_TOKEN", "")
headers = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://brapi.dev/api/v2/fii"

# FIIs que queremos buscar o real de 2025
FIIS = ["HGLG11","MXRF11","XPML11","HGCR11","VISC11","KNCR11","IRDM11","CPTS11"]

resultado = {}

for fii in FIIS:
    r = requests.get(f"{BASE}/indicators/history", headers=headers, params={
        "symbols":   fii,
        "startDate": "2025-01-01",
        "endDate":   "2025-12-31",
        "sortOrder": "asc"
    }, timeout=30)

    if r.status_code == 200:
        history = r.json().get("history", [])
        dados = {}
        for item in history:
            mes = item.get("referenceDate","")[:7]  # YYYY-MM
            dy  = item.get("dividendYield1m")
            pvp = item.get("priceToNav")
            if mes and dy:
                dados[mes] = {"dy": round(dy, 6), "pvp": round(pvp,4) if pvp else None}
        resultado[fii] = dados
        print(f"✅ {fii}: {len(dados)} meses ({list(dados.keys())})")
    else:
        print(f"❌ {fii}: HTTP {r.status_code} — {r.text[:100]}")
    time.sleep(0.3)

print("\n\n=== JSON para colar no HTML ===")
print(json.dumps(resultado, indent=2, ensure_ascii=False))