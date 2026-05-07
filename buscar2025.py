import requests, os, json
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("BRAPI_TOKEN","")
# Todos os fundos que aparecem na interface
FIIS = [
    "HGLG11","XPLG11","VILG11","BRCO11","BTLG11","HLOG11","LGVT11","LVBI11",
    "XPML11","VISC11","HSML11","HGBS11","MALL11","ABCP11","PQDP11","SHDP11","SHPH11","VPSI11","VSHO11","WPLZ11","ELDO11","FIGS11","FLRP11",
    "KNRI11","ALZR11","BRCR11","RBRD11","PATB11","VERE11","BRRI11",
    "BBPO11","HGRE11","CBOP11","CEOC11","CNES11","GTWR11","RNGO11","TRNT11","FLMA11","FPAB11",
    "EDGA11","FPNG11"
]

resultado = {}
for fii in FIIS:
    r = requests.get(
        "https://brapi.dev/api/v2/fii/indicators/history",
        headers={"Authorization": f"Bearer {TOKEN}"},
        params={"symbols":fii,"startDate":"2025-01-01","endDate":"2025-12-31","sortOrder":"asc"},
        timeout=30
    )
    if r.status_code == 200:
        h = r.json().get("history",[])
        dys = [round(item["dividendYield1m"],6) for item in h if item.get("dividendYield1m")]
        if dys:
            resultado[fii] = dys
            print(f"✅ {fii}: {len(dys)} meses")
        else:
            print(f"⚠ {fii}: sem DY")
    else:
        print(f"❌ {fii}: HTTP {r.status_code}")

print("\n=== COLE NO HTML ===")
print("const REAL_2025 = " + json.dumps(resultado, indent=2) + ";")