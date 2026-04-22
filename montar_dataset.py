"""
montar_dataset.py
─────────────────
Monta dataset profissional de FIIs (jan/2019 – dez/2024).

Fontes (todas gratuitas, sem autenticação):
  - Dividendos + Cotação → Yahoo Finance (query1.finance.yahoo.com)
  - SELIC mensal         → Banco Central do Brasil API SGS série 4390
  - IFIX mensal          → Banco Central do Brasil API SGS série 12466

Instalação:
    pip install requests pandas openpyxl
"""

import requests
import pandas as pd
import time

PERIODO_INI = "2019-01"
PERIODO_FIM = "2024-12"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ─────────────────────────────────────────────────────────────────────────────
# Lista de FIIs por segmento
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
def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        return r
    except Exception:
        return None


def safe_json(r):
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
        print("   ⚠ Falha. Preenchendo SELIC com NaN.")
        return {}
    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["ano_mes"] = df["data"].str(dt.to_period("M"))
    df["SELIC"] = df["valor"].apply(lambda v: round(to_float(v) / 100, 6) if to_float(v) else None)
    print(f"   ✓ {len(df)} meses carregados")
    return dict(zip(df["ano_mes"], df["SELIC"]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. IFIX — BCB série 12466
# ─────────────────────────────────────────────────────────────────────────────
def buscar_ifix():
    print("📡 Buscando IFIX mensal do BCB (série 12466)...")
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12466/dados"
    r = safe_get(url, params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"})
    data = safe_json(r)
    if not data:
        print("   ⚠ Falha. Preenchendo IFIX com NaN.")
        return {}
    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["ano_mes"] = df["data"].str(dt.to_period("M"))
    df = df[(df["ano_mes"] >= PERIODO_INI) & (df["ano_mes"] <= PERIODO_FIM)]
    df["IFIX"] = df["valor"].apply(lambda v: round(to_float(v) / 100, 6) if to_float(v) else None)
    print(f"   ✓ {len(df)} meses carregados")
    return dict(zip(df["ano_mes"], df["IFIX"]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Yahoo Finance — cotação mensal + dividendos
# ─────────────────────────────────────────────────────────────────────────────
def buscar_yahoo(sigla):
    """
    Retorna:
      cotacoes  = {ano_mes: preco_fechamento}
      dividendos = {ano_mes: valor_total_dividendos}
      pvp_proxy  = None (Yahoo não fornece P/VP)
    """
    ticker = f"{sigla}.SA"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = safe_get(url, params={"interval": "1mo", "range": "7y", "events": "dividends"})
    data = safe_json(r)

    if not data:
        return {}, {}, None

    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        return {}, {}, None

    # ── Cotações mensais ──────────────────────────────────────────────────────
    timestamps = result.get("timestamp", [])
    closes = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

    cotacoes = {}
    for ts, preco in zip(timestamps, closes):
        if preco is None:
            continue
        dt = pd.to_datetime(ts, unit="s")
        mes = str(dt.to_period("M"))
        if PERIODO_INI <= mes <= PERIODO_FIM:
            cotacoes[mes] = round(float(preco), 4)

    # ── Dividendos ────────────────────────────────────────────────────────────
    divs_raw = result.get("events", {}).get("dividends", {})
    dividendos = {}
    for ts_str, info in divs_raw.items():
        try:
            dt = pd.to_datetime(int(ts_str), unit="s")
            mes = str(dt.to_period("M"))
            if PERIODO_INI <= mes <= PERIODO_FIM:
                valor = float(info.get("amount", 0))
                dividendos[mes] = dividendos.get(mes, 0) + valor
        except Exception:
            continue

    return cotacoes, dividendos


# ─────────────────────────────────────────────────────────────────────────────
# 4. P/VP — estimado via Yahoo (bookValue)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_pvp_yahoo(sigla):
    """P/VP estimado: cotação atual / valor patrimonial via Yahoo summary."""
    ticker = f"{sigla}.SA"
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    r = safe_get(url, params={"modules": "defaultKeyStatistics,summaryDetail"})
    data = safe_json(r)
    if not data:
        return None
    try:
        stats = data["quoteSummary"]["result"][0]
        pvp = stats.get("defaultKeyStatistics", {}).get("priceToBook", {}).get("raw")
        return round(float(pvp), 4) if pvp else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Montar dataset completo
# ─────────────────────────────────────────────────────────────────────────────
def montar_dataset():
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

        cotacoes, dividendos = buscar_yahoo(sigla)[:2]
        time.sleep(0.5)

        pvp = buscar_pvp_yahoo(sigla)
        time.sleep(0.3)

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
                "Vacancia":         None,   # não disponível gratuitamente
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

    print("\nFundos com menos de 12 meses de dados:")
    for r in resumo:
        if r["meses_com_dy"] < 12:
            print(f"   {r['sigla']}: {r['meses_com_dy']} meses")

    print("\nDistribuição por segmento:")
    print(df_completo.groupby("Segmento")[["Sigla"]].nunique().to_string())

    nome = "dataset_fiis_2019_2024.xlsx"
    df_completo.to_excel(nome, index=False)
    print(f"\n✅ Arquivo salvo: {nome}")

    print("\nAmostra (5 linhas):")
    print(df_completo.head(5).to_string(index=False))

    return df_completo


if __name__ == "__main__":
    montar_dataset()