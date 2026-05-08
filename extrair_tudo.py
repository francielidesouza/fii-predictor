import requests, os, json, time
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("BRAPI_TOKEN","")
headers = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://brapi.dev/api/v2/fii"

# Todos os fundos do dataset
FIIS = [
    # Logístico
    "HGLG11","XPLG11","VILG11","BRCO11","BTLG11","HLOG11","LGVT11","LVBI11","CXTL11","EURO11","FIIB11","LVBI11",
    # Shoppings
    "XPML11","VISC11","HSML11","HGBS11","MALL11","ABCP11","PQDP11","SHDP11","SHPH11","VPSI11","VSHO11","WPLZ11","ELDO11","FIGS11","FLRP11",
    # Híbrido
    "KNRI11","ALZR11","BRCR11","RBRD11","PATB11","VERE11","BRRI11","MXRF11",
    # Escritórios
    "BBPO11","HGRE11","CBOP11","CEOC11","CNES11","GTWR11","RNGO11","TRNT11","FLMA11","FPAB11",
    # Lajes Corporativas
    "EDGA11","FPNG11",
    # Títulos e Val. Mob (para referência)
    "KNCR11","KNHY11","KNIP11","MCCI11","NCHB11","OUJP11","PLRI11","PORD11","RBRR11","SADI11","VCJR11","VGIR11","VOTS11","CVBI11","FEXC11","HGCR11","IRDM11","VRTA11","CPTS11",
]
FIIS = list(dict.fromkeys(FIIS))  # remove duplicatas

resultado = {}

for periodo, ini, fim in [
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
]:
    resultado[periodo] = {}
    print(f"\n=== Extraindo {periodo} ===")
    for fii in FIIS:
        r = requests.get(
            f"{BASE}/indicators/history",
            headers=headers,
            params={"symbols":fii,"startDate":ini,"endDate":fim,"sortOrder":"asc"},
            timeout=30
        )
        if r.status_code == 200:
            h = r.json().get("history",[])
            dados = {}
            for item in h:
                mes = item.get("referenceDate","")[:7]
                dy  = item.get("dividendYield1m")
                pvp = item.get("priceToNav")
                if mes and dy is not None:
                    dados[mes] = {
                        "dy":  round(float(dy), 6),
                        "pvp": round(float(pvp), 4) if pvp else None,
                    }
            resultado[periodo][fii] = dados
            print(f"  ✅ {fii}: {len(dados)} meses")
        else:
            print(f"  ❌ {fii}: HTTP {r.status_code}")
        time.sleep(0.2)

# Salva JSON completo
with open("dados_fiis_2024_2025.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

print(f"\n✅ Salvo em dados_fiis_2024_2025.json")
print(f"   Fundos: {len(FIIS)}")
print(f"   Períodos: 2024 e 2025")