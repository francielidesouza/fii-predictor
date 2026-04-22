"""
montar_dataset.py
─────────────────
Monta dataset profissional de FIIs (jan/2019 – dez/2024).

Fontes:
  - SELIC  → Banco Central do Brasil API SGS série 4390 (sem autenticação)
  - IFIX   → Banco Central do Brasil API SGS série 12466 (sem autenticação)
  - FIIs   → brapi.dev (token via .env)

Instalação:
    pip install requests pandas openpyxl python-dotenv
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
BRAPI_TOKEN = os.getenv("BRAPI_TOKEN")

PERIODO_INI = "2019-01"
PERIODO_FIM = "2024-12"

# ─────────────────────────────────────────────────────────────────────────────
# Lista de FIIs
# ─────────────────────────────────────────────────────────────────────────────
FIIS = [
    {"sigla": "BBAM11", "segmento": "Agencia Bancaria",              "tipo": "Fundo de Tijolo"},
    {"sigla": "CXAG11", "segmento": "Agencia Bancaria",              "tipo": "Fundo de Tijolo"},
    {"sigla": "GARE11", "segmento": "Galpoes",                       "tipo": "Fundo de Tijolo"},
    {"sigla": "TRBL11", "segmento": "Galpoes",                       "tipo": "Fundo de Tijolo"},
    {"sigla": "RBVA11", "segmento": "Hibrido",                       "tipo": "Fundo de Tijolo"},
    {"sigla": "KISU11", "segmento": "Hibrido",                       "tipo": "Fundo de Tijolo"},
    {"sigla": "SDIL11", "segmento": "Industria",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "HGRE11", "segmento": "Lajes Corporativas",            "tipo": "Fundo de Tijolo"},
    {"sigla": "JSRE11", "segmento": "Lajes Corporativas",            "tipo": "Fundo de Tijolo"},
    {"sigla": "BRCR11", "segmento": "Lajes Corporativas",            "tipo": "Fundo de Tijolo"},
    {"sigla": "HGLG11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "XPLG11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "VILG11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "BRCO11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "MALL11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "XPML11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "VISC11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "HSML11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "HGCR11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "KNCR11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "MXRF11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "IRDM11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "CPTS11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "BCFF11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Fundos"},
    {"sigla": "RBFF11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Fundos"},
    {"sigla": "HCTR11", "segmento": "Hibrido",                       "tipo": "Fundo de Desenvolvimento"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_get(url, params=None, timeout=30):
    """GET com tratamento de erro — retorna None se falhar."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        # Testa se é JSON válido antes de retornar
        r.json()
        return r
    except Exception:
        return None


def safe_json(r):
    """Extrai JSON de forma segura."""
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def to_float(val):
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "."))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. SELIC — BCB série 4390 (% a.m.)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_selic():
    print("📡 Buscando SELIC mensal do BCB (série 4390)...")
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados"
    r = safe_get(url, params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"})
    data = safe_json(r)
    if not data:
        print("   ⚠ Falha ao buscar SELIC. Preenchendo com NaN.")
        return {}
    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    df["SELIC"] = df["valor"].apply(lambda v: round(to_float(v) / 100, 6) if to_float(v) else None)
    print(f"   ✓ {len(df)} meses carregados")
    return dict(zip(df["ano_mes"], df["SELIC"]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. IFIX — BCB série 12466 (variação mensal do índice IFIX)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_ifix():
    print("📡 Buscando IFIX mensal do BCB (série 12466)...")
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12466/dados"
    r = safe_get(url, params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"})
    data = safe_json(r)
    if not data:
        print("   ⚠ Série 12466 não disponível. Tentando série alternativa 12469...")
        # Série alternativa: IFIX - índice de fundos imobiliários
        url2 = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12469/dados"
        r2 = safe_get(url2, params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"})
        data = safe_json(r2)
    if not data:
        print("   ⚠ IFIX não disponível via BCB. Preenchendo com NaN.")
        return {}
    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    df = df[(df["ano_mes"] >= PERIODO_INI) & (df["ano_mes"] <= PERIODO_FIM)]
    df["IFIX"] = df["valor"].apply(lambda v: round(to_float(v) / 100, 6) if to_float(v) else None)
    print(f"   ✓ {len(df)} meses carregados")
    return dict(zip(df["ano_mes"], df["IFIX"]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cotação histórica mensal — brapi.dev
# ─────────────────────────────────────────────────────────────────────────────
def buscar_cotacao(sigla):
    url = f"https://brapi.dev/api/quote/{sigla}"
    r = safe_get(url, params={"range": "6y", "interval": "1mo", "token": BRAPI_TOKEN})
    data = safe_json(r)
    if not data:
        return {}
    hist = data.get("results", [{}])[0].get("historicalDataPrice", [])
    result = {}
    for h in hist:
        try:
            dt = pd.to_datetime(h["date"], unit="s")
            mes = dt.to_period("M").astype(str)
            if PERIODO_INI <= mes <= PERIODO_FIM:
                result[mes] = to_float(h.get("close"))
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dividendos históricos — brapi.dev
# ─────────────────────────────────────────────────────────────────────────────
def buscar_dividendos(sigla):
    url = f"https://brapi.dev/api/funds/{sigla}"
    r = safe_get(url, params={"token": BRAPI_TOKEN})
    data = safe_json(r)
    if not data:
        return {}
    results = data.get("results", [])
    if not results:
        return {}
    divs = results[0].get("dividendsData", {}).get("cashDividends", [])
    por_mes = {}
    for d in divs:
        try:
            dt = pd.to_datetime(
                d.get("paymentDate") or d.get("lastDatePrior"), errors="coerce"
            )
            if pd.isna(dt):
                continue
            mes = dt.to_period("M").astype(str)
            if PERIODO_INI <= mes <= PERIODO_FIM:
                val = to_float(d.get("rate", 0)) or 0
                por_mes[mes] = por_mes.get(mes, 0) + val
        except Exception:
            continue
    return por_mes


# ─────────────────────────────────────────────────────────────────────────────
# 5. P/VP e Vacância — brapi.dev (valor atual como proxy histórico)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_indicadores(sigla):
    url = f"https://brapi.dev/api/funds/{sigla}"
    r = safe_get(url, params={"token": BRAPI_TOKEN})
    data = safe_json(r)
    if not data:
        return None, None
    results = data.get("results", [])
    if not results:
        return None, None
    f = results[0]
    pvp = to_float(f.get("priceToBook") or f.get("pvp"))
    vac = to_float(f.get("physicalVacancy") or f.get("financialVacancy"))
    if pvp:
        pvp = round(pvp, 4)
    if vac:
        vac = round(vac / 100, 4)
    return pvp, vac


# ─────────────────────────────────────────────────────────────────────────────
# 6. Montar dataset completo
# ─────────────────────────────────────────────────────────────────────────────
def montar_dataset():
    if not BRAPI_TOKEN:
        print("⚠ BRAPI_TOKEN não encontrado no .env. Adicione: BRAPI_TOKEN=seu_token")
        return

    meses = [str(m) for m in pd.period_range(PERIODO_INI, PERIODO_FIM, freq="M")]

    selic = buscar_selic()
    ifix  = buscar_ifix()

    todas_linhas = []
    resumo = []

    for fii in FIIS:
        sigla    = fii["sigla"]
        segmento = fii["segmento"]
        tipo     = fii["tipo"]
        print(f"\n🔍 {sigla} ({segmento})...", end=" ", flush=True)

        cotacoes  = buscar_cotacao(sigla);   time.sleep(0.4)
        dividendos= buscar_dividendos(sigla); time.sleep(0.4)
        pvp, vac  = buscar_indicadores(sigla); time.sleep(0.4)

        meses_com_dy = 0
        for mes in meses:
            preco = cotacoes.get(mes)
            div   = dividendos.get(mes, 0)
            dy    = round(div / preco, 6) if (preco and preco > 0 and div > 0) else None

            todas_linhas.append({
                "Data":             mes,
                "Sigla":            sigla,
                "Segmento":         segmento,
                "Tipo_do_Fundo":    tipo,
                "Dividendos_Yield": dy,
                "P_VP":             pvp,
                "Vacancia":         vac,
                "SELIC":            selic.get(mes),
                "IFIX":             ifix.get(mes),
            })
            if dy:
                meses_com_dy += 1

        print(f"✓ {meses_com_dy} meses com DY")
        resumo.append({"sigla": sigla, "meses_com_dy": meses_com_dy})

    df = pd.DataFrame(todas_linhas)
    df_completo = df.dropna(subset=["Dividendos_Yield"]).copy()
    df_completo = df_completo.sort_values(["Sigla", "Data"]).reset_index(drop=True)

    print(f"\n{'─'*50}")
    print(f"📊 Dataset final: {len(df_completo)} linhas · {df_completo['Sigla'].nunique()} fundos")
    print("\nFundos com menos de 12 meses de dados (podem ter iniciado após 2019):")
    for r in resumo:
        if r["meses_com_dy"] < 12:
            print(f"   {r['sigla']}: {r['meses_com_dy']} meses")

    print("\nDistribuição por segmento:")
    print(df_completo.groupby("Segmento")["Sigla"].nunique().to_string())

    nome = "dataset_fiis_2019_2024.xlsx"
    df_completo.to_excel(nome, index=False)
    print(f"\n✅ Arquivo salvo: {nome}")
    print("\nAmostra (5 linhas):")
    print(df_completo.head(5).to_string(index=False))

    return df_completo


if __name__ == "__main__":
    montar_dataset()