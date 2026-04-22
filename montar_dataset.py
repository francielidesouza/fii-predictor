"""
montar_dataset.py
─────────────────
Monta dataset profissional de FIIs (jan/2019 – dez/2024) com:
  - Dividendos_Yield, P_VP, Vacancia  → brapi.dev (dados de mercado)
  - SELIC mensal                       → Banco Central do Brasil (API oficial)
  - IFIX mensal                        → brapi.dev (histórico do índice)
  - Segmento, Tipo_do_Fundo            → tabela local (CVM / B3)

Fontes oficiais:
  - BCB/SGS série 4390: https://api.bcb.gov.br  (dados abertos, sem autenticação)
  - brapi.dev: https://brapi.dev                (gratuito com cadastro)

Instalação:
    pip install requests pandas openpyxl

Uso:
    1. Crie uma conta gratuita em https://brapi.dev e copie seu token
    2. Cole o token na variável BRAPI_TOKEN abaixo
    3. python montar_dataset.py
"""

import requests
import pandas as pd
from datetime import datetime, date
import time
import json



# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — cole seu token brapi aqui
# Cadastro gratuito em: https://brapi.dev
# ─────────────────────────────────────────────────────────────────────────────
BRAPI_TOKEN = API_TOKEN_BRAPI
##############
# from dotenv import load_dotenv
# import os

# load_dotenv()
# BRAPI_TOKEN = os.getenv("BRAPI_TOKEN")
###############

# Período do dataset
DATA_INICIO = "2019-01-01"
DATA_FIM    = "2024-12-31"

# ─────────────────────────────────────────────────────────────────────────────
# LISTA DE FIIs por segmento e tipo
# Fonte: B3 / CVM — fundos com histórico completo 2019-2024
# ─────────────────────────────────────────────────────────────────────────────
FIIS = [
    # Agência Bancária
    {"sigla": "BBAM11", "segmento": "Agencia Bancaria",              "tipo": "Fundo de Tijolo"},
    {"sigla": "CXAG11", "segmento": "Agencia Bancaria",              "tipo": "Fundo de Tijolo"},

    # Galpões
    {"sigla": "GARE11", "segmento": "Galpoes",                       "tipo": "Fundo de Tijolo"},
    {"sigla": "TRBL11", "segmento": "Galpoes",                       "tipo": "Fundo de Tijolo"},

    # Híbrido
    {"sigla": "RBVA11", "segmento": "Hibrido",                       "tipo": "Fundo de Tijolo"},
    {"sigla": "KISU11", "segmento": "Hibrido",                       "tipo": "Fundo de Tijolo"},

    # Indústria
    {"sigla": "SDIL11", "segmento": "Industria",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "FIIP11B","segmento": "Industria",                     "tipo": "Fundo de Tijolo"},

    # Lajes Corporativas
    {"sigla": "HGRE11", "segmento": "Lajes Corporativas",            "tipo": "Fundo de Tijolo"},
    {"sigla": "JSRE11", "segmento": "Lajes Corporativas",            "tipo": "Fundo de Tijolo"},
    {"sigla": "BRCR11", "segmento": "Lajes Corporativas",            "tipo": "Fundo de Tijolo"},

    # Logístico
    {"sigla": "HGLG11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "XPLG11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "VILG11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},
    {"sigla": "BRCO11", "segmento": "Logistico",                     "tipo": "Fundo de Tijolo"},

    # Shopping e Varejo
    {"sigla": "MALL11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "XPML11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "VISC11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},
    {"sigla": "HSML11", "segmento": "Shopping e Varejo",             "tipo": "Fundo de Tijolo"},

    # Títulos e Valores Mobiliários (Papel)
    {"sigla": "HGCR11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "KNCR11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "MXRF11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "IRDM11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},
    {"sigla": "CPTS11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Papel"},

    # Fundo de Fundos
    {"sigla": "BCFF11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Fundos"},
    {"sigla": "RBFF11", "segmento": "Titulos e Valores Mobiliarios", "tipo": "Fundo de Fundos"},

    # Fundo de Desenvolvimento
    {"sigla": "HCTR11", "segmento": "Hibrido",                       "tipo": "Fundo de Desenvolvimento"},
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. SELIC MENSAL — Banco Central do Brasil (API pública, sem token)
# Série 4390: Taxa Selic acumulada no mês (% a.m.)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_selic():
    print("📡 Buscando SELIC mensal do Banco Central (série 4390)...")
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados"
        "?formato=json&dataInicial=01/01/2019&dataFinal=31/12/2024"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dados = r.json()
    df = pd.DataFrame(dados)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    df["SELIC"] = pd.to_numeric(df["valor"].str.replace(",", "."), errors="coerce") / 100
    print(f"   ✓ {len(df)} meses de SELIC carregados")
    return df[["ano_mes", "SELIC"]].set_index("ano_mes")


# ─────────────────────────────────────────────────────────────────────────────
# 2. IFIX MENSAL — brapi.dev
# ─────────────────────────────────────────────────────────────────────────────
def buscar_ifix():
    print("📡 Buscando IFIX histórico via brapi.dev...")
    url = f"https://brapi.dev/api/quote/IFIX11?range=5y&interval=1mo&token={BRAPI_TOKEN}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"   ⚠ IFIX não disponível ({r.status_code}). Preenchendo com NaN.")
        return pd.Series(dtype=float, name="IFIX")

    data = r.json()
    hist = data.get("results", [{}])[0].get("historicalDataPrice", [])
    if not hist:
        print("   ⚠ Histórico IFIX vazio. Preenchendo com NaN.")
        return pd.Series(dtype=float, name="IFIX")

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df["ano_mes"] = df["date"].dt.to_period("M").astype(str)
    df = df.sort_values("date")

    # Variação mensal = (close_atual / close_anterior) - 1
    df["IFIX"] = df["close"].pct_change().round(6)
    df = df.dropna(subset=["IFIX"])
    df = df[(df["ano_mes"] >= "2019-01") & (df["ano_mes"] <= "2024-12")]
    print(f"   ✓ {len(df)} meses de IFIX carregados")
    return df[["ano_mes", "IFIX"]].set_index("ano_mes")["IFIX"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. DADOS DO FII — brapi.dev (dividendos, cotação, P/VP)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_dividendos_fii(sigla):
    """Retorna DataFrame com Data (ano_mes) e Dividendos_Yield mensal."""
    url = f"https://brapi.dev/api/funds/{sigla}?token={BRAPI_TOKEN}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    fundo = results[0]

    # Dividendos históricos
    dividendos = fundo.get("dividendsData", {}).get("cashDividends", [])
    if not dividendos:
        return pd.DataFrame()

    rows = []
    for d in dividendos:
        try:
            dt = pd.to_datetime(d.get("paymentDate") or d.get("lastDatePrior"), errors="coerce")
            if pd.isna(dt):
                continue
            ano_mes = dt.to_period("M").astype(str)
            if ano_mes < "2019-01" or ano_mes > "2024-12":
                continue
            valor = float(str(d.get("rate", 0)).replace(",", "."))
            rows.append({"ano_mes": ano_mes, "dividendo_valor": valor})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.groupby("ano_mes")["dividendo_valor"].sum().reset_index()
    return df


def buscar_historico_cotacao(sigla):
    """Retorna DataFrame com ano_mes, preco_medio, pvp_estimado."""
    url = (
        f"https://brapi.dev/api/quote/{sigla}"
        f"?range=6y&interval=1mo&fundamental=true&token={BRAPI_TOKEN}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    hist = data.get("results", [{}])[0].get("historicalDataPrice", [])
    if not hist:
        return pd.DataFrame()

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df["ano_mes"] = df["date"].dt.to_period("M").astype(str)
    df = df[(df["ano_mes"] >= "2019-01") & (df["ano_mes"] <= "2024-12")]
    df = df.rename(columns={"close": "preco_fechamento"})
    return df[["ano_mes", "preco_fechamento"]].set_index("ano_mes")


# ─────────────────────────────────────────────────────────────────────────────
# 4. P/VP E VACÂNCIA — brapi.dev (indicadores do fundo)
# ─────────────────────────────────────────────────────────────────────────────
def buscar_indicadores_fii(sigla):
    """Retorna P/VP e vacância atual (será propagado historicamente como proxy)."""
    url = f"https://brapi.dev/api/funds/{sigla}?token={BRAPI_TOKEN}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None, None

    data = r.json()
    results = data.get("results", [])
    if not results:
        return None, None

    fundo = results[0]
    pvp = fundo.get("priceToBook") or fundo.get("pvp")
    vacancia = fundo.get("physicalVacancy") or fundo.get("financialVacancy")

    try:
        pvp = round(float(pvp), 4) if pvp else None
    except Exception:
        pvp = None
    try:
        vacancia = round(float(vacancia) / 100, 4) if vacancia else None
    except Exception:
        vacancia = None

    return pvp, vacancia


# ─────────────────────────────────────────────────────────────────────────────
# 5. MONTAR DATASET COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def montar_dataset():
    # Gera todos os meses do período
    meses = pd.period_range("2019-01", "2024-12", freq="M")
    meses_str = [str(m) for m in meses]

    # Busca SELIC e IFIX (fontes macro — uma vez só)
    selic = buscar_selic()
    ifix  = buscar_ifix()

    all_rows = []

    for fii in FIIS:
        sigla    = fii["sigla"]
        segmento = fii["segmento"]
        tipo     = fii["tipo"]

        print(f"\n🔍 Processando {sigla} ({segmento})...")

        # Cotação histórica mensal
        cotacao_df = buscar_historico_cotacao(sigla)
        time.sleep(0.5)  # respeita rate limit da API

        # Dividendos históricos
        div_df = buscar_dividendos_fii(sigla)
        time.sleep(0.5)

        # Indicadores atuais (P/VP e vacância como proxy histórico)
        pvp_atual, vac_atual = buscar_indicadores_fii(sigla)
        time.sleep(0.5)

        for mes in meses_str:
            # Preço de fechamento do mês
            preco = None
            if not cotacao_df.empty and mes in cotacao_df.index:
                preco = cotacao_df.loc[mes, "preco_fechamento"]

            # Dividendo do mês
            dividendo = 0.0
            if not div_df.empty:
                linha = div_df[div_df["ano_mes"] == mes]
                if not linha.empty:
                    dividendo = float(linha["dividendo_valor"].iloc[0])

            # DY = dividendo / preço
            dy = round(dividendo / preco, 6) if (preco and preco > 0 and dividendo > 0) else None

            # SELIC do mês
            selic_val = selic["SELIC"].get(mes) if mes in selic.index else None

            # IFIX do mês
            ifix_val = ifix.get(mes) if mes in ifix.index else None

            all_rows.append({
                "Data":              mes,
                "Sigla":             sigla,
                "Segmento":          segmento,
                "Tipo_do_Fundo":     tipo,
                "Dividendos_Yield":  dy,
                "P_VP":              pvp_atual,   # proxy: valor mais recente
                "Vacancia":          vac_atual,   # proxy: valor mais recente
                "SELIC":             selic_val,
                "IFIX":              ifix_val,
            })

        print(f"   ✓ {sigla} — {len([r for r in all_rows if r['Sigla']==sigla and r['Dividendos_Yield']])} meses com DY")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["Sigla", "Data"]).reset_index(drop=True)

    # Remove meses sem DY (fundo ainda não existia ou sem dados)
    df_completo = df.dropna(subset=["Dividendos_Yield"]).copy()

    print(f"\n📊 Dataset final: {len(df_completo)} linhas · {df_completo['Sigla'].nunique()} fundos")
    print(df_completo.groupby("Segmento")["Sigla"].nunique().to_string())

    # Salva Excel
    nome = "dataset_fiis_2019_2024.xlsx"
    df_completo.to_excel(nome, index=False)
    print(f"\n✅ Arquivo salvo: {nome}")

    # Preview
    print("\nPrimeiras linhas:")
    print(df_completo.head(10).to_string(index=False))

    return df_completo


if __name__ == "__main__":
    if BRAPI_TOKEN == API_TOKEN_BRAPI:
        print("⚠ ATENÇÃO: Substitua API_TOKEN_BRAPI pelo seu token da brapi.dev")
        print("  Cadastro gratuito em: https://brapi.dev")
        print("  O token gratuito permite até 1.000 requisições/dia — suficiente para este script.")
    else:
        montar_dataset()