"""
montar_dataset_brapi.py
───────────────────────
Monta dataset de FIIs (jan/2019 – dez/2024) via brapi.dev Pro + BCB.

Colunas finais:
  Data, Sigla, Segmento, Tipo_do_Fundo, Dividendos_Yield, P_VP, SELIC

Fontes:
  - brapi.dev Pro → Sigla, Segmento, Tipo_do_Fundo, DY mensal, P/VP histórico
  - BCB SGS 4390  → SELIC mensal (fonte primária oficial)

Nota metodológica:
  Vacância foi investigada em todos os endpoints da brapi.dev Pro
  (indicators, indicators/history, reports) e não foi encontrada em
  nenhum campo retornado, apesar de mencionada na documentação descritiva.
  A variável foi excluída do dataset e declarada como limitação do trabalho.

Pré-requisito .env:
    BRAPI_TOKEN=seu_token

Uso:
    python montar_dataset_brapi.py
"""

import os, time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BRAPI_TOKEN = os.getenv("BRAPI_TOKEN", "")
if not BRAPI_TOKEN:
    raise RuntimeError("BRAPI_TOKEN não encontrado no .env")

BRAPI_BASE = "https://brapi.dev/api/v2/fii"
BCB_BASE   = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"

MES_INI  = "2019-01"
MES_FIM  = "2024-12"
DATA_INI = "2019-01-01"
DATA_FIM = "2024-12-31"

headers_brapi = {"Authorization": f"Bearer {BRAPI_TOKEN}"}

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(url, headers=None, params=None, tentativas=3):
    for i in range(tentativas):
        try:
            r = requests.get(url, headers=headers or {}, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                print("⏳ rate limit...", end=" ")
                time.sleep(10)
            else:
                print(f"  ⚠ HTTP {r.status_code}: {r.text[:100]}")
                return None
        except Exception as e:
            print(f"  ⚠ Erro ({i+1}/{tentativas}): {e}")
            time.sleep(2)
    return None

def mes_fmt(dt) -> str:
    return pd.to_datetime(dt).strftime("%Y-%m")

# ─────────────────────────────────────────────────────────────────────────────
# 1. SELIC — BCB SGS série 4390 (fonte primária oficial)
# ─────────────────────────────────────────────────────────────────────────────

def buscar_selic() -> dict:
    print("📡 SELIC (BCB SGS 4390)...", end=" ")
    r = safe_get(
        f"{BCB_BASE}.4390/dados",
        params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"}
    )
    if not r:
        print("⚠ falhou")
        return {}
    result = {}
    for item in r.json():
        try:
            mes = mes_fmt(pd.to_datetime(item["data"], format="%d/%m/%Y"))
            result[mes] = round(float(item["valor"]) / 100, 6)
        except Exception:
            continue
    print(f"✓ {len(result)} meses")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 2. Lista todos os FIIs disponíveis na brapi
# ─────────────────────────────────────────────────────────────────────────────

def listar_fiis() -> pd.DataFrame:
    print("📡 Listando FIIs (brapi)...", end=" ")
    r = safe_get(f"{BRAPI_BASE}/list", headers_brapi, params={"limit": 10000})
    if not r:
        print("⚠ falhou")
        return pd.DataFrame()

    fiis = r.json().get("fiis", [])
    df = pd.DataFrame(fiis)

    cols = ["symbol", "segmentoAtuacao", "segmentType", "mandate"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.rename(columns={
        "symbol":          "Sigla",
        "segmentoAtuacao": "Segmento",
        "segmentType":     "Tipo_do_Fundo",
        "mandate":         "Mandato",
    })

    # Remove registros inválidos
    df = df.dropna(subset=["Sigla", "Segmento"])
    df = df[df["Sigla"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df = df[df["Segmento"] != ""]

    print(f"✓ {len(df)} FIIs encontrados")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. Histórico mensal de indicadores (DY e P/VP)
# ─────────────────────────────────────────────────────────────────────────────

def buscar_historico_indicadores(siglas: list) -> pd.DataFrame:
    # Remove siglas inválidas
    siglas = [s for s in siglas if isinstance(s, str) and s.strip() != ""]

    print(f"\n📡 Histórico de indicadores (brapi) — {len(siglas)} FIIs...")
    todos = []
    lotes = [siglas[i:i+20] for i in range(0, len(siglas), 20)]

    for idx, lote in enumerate(lotes):
        lote_valido = [s for s in lote if isinstance(s, str) and s.strip() != ""]
        if not lote_valido:
            continue

        symbols = ",".join(lote_valido)
        print(f"  Lote {idx+1}/{len(lotes)}: {symbols[:60]}...", end=" ")

        r = safe_get(
            f"{BRAPI_BASE}/indicators/history",
            headers_brapi,
            params={
                "symbols":   symbols,
                "startDate": DATA_INI,
                "endDate":   DATA_FIM,
                "sortOrder": "asc",
            }
        )

        if r:
            history = r.json().get("history", [])
            todos.extend(history)
            print(f"✓ {len(history)} registros")
        else:
            print("⚠ falhou")

        time.sleep(0.5)

    if not todos:
        return pd.DataFrame()

    df = pd.DataFrame(todos)

    # Formata mês
    df["Data"] = df["referenceDate"].apply(lambda x: mes_fmt(str(x)[:10]))

    # Filtra período
    df = df[(df["Data"] >= MES_INI) & (df["Data"] <= MES_FIM)]

    # Renomeia colunas
    df = df.rename(columns={
        "symbol":          "Sigla",
        "dividendYield1m": "Dividendos_Yield",
        "priceToNav":      "P_VP",
    })

    colunas = ["Data", "Sigla", "Dividendos_Yield", "P_VP"]
    df = df[[c for c in colunas if c in df.columns]]

    print(f"\n  ✓ Total: {len(df)} registros históricos")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. Montar dataset final
# ─────────────────────────────────────────────────────────────────────────────

def montar_dataset():
    print("🚀 Iniciando coleta de dados...\n")

    selic   = buscar_selic()
    df_fiis = listar_fiis()

    if df_fiis.empty:
        print("❌ Falha ao listar FIIs")
        return None

    siglas  = df_fiis["Sigla"].tolist()
    df_hist = buscar_historico_indicadores(siglas)

    if df_hist.empty:
        print("❌ Falha ao buscar histórico")
        return None

    # ── Junta histórico com info cadastral ────────────────────────────────────
    print("\n📊 Montando dataset final...")

    df = df_hist.merge(
        df_fiis[["Sigla", "Segmento", "Tipo_do_Fundo"]],
        on="Sigla", how="left"
    )

    # Adiciona SELIC
    df["SELIC"] = df["Data"].map(selic)

    # Remove registros sem DY
    df = df.dropna(subset=["Dividendos_Yield"])
    df = df[df["Dividendos_Yield"] > 0]

    # Remove fundos sem segmento
    df = df.dropna(subset=["Segmento"])

    # Filtra mínimo 60 meses por fundo
    contagem = df.groupby("Sigla")["Data"].nunique()
    siglas_ok = contagem[contagem >= 60].index
    df_full = df[df["Sigla"].isin(siglas_ok)]

    if df_full.empty:
        print("⚠ Nenhum fundo com 60+ meses — reduzindo para 48...")
        siglas_ok = contagem[contagem >= 48].index
        df_full = df[df["Sigla"].isin(siglas_ok)]

    df = df_full.copy()

    # Colunas finais
    colunas = ["Data", "Sigla", "Segmento", "Tipo_do_Fundo",
               "Dividendos_Yield", "P_VP", "SELIC"]
    df = df[[c for c in colunas if c in df.columns]]
    df = df.sort_values(["Sigla", "Data"]).reset_index(drop=True)

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"📊 Dataset final: {len(df)} linhas · {df['Sigla'].nunique()} fundos")
    print(f"   Período: {df['Data'].min()} → {df['Data'].max()}")
    print(f"   DY médio:  {df['Dividendos_Yield'].mean():.4f}")
    print(f"   P/VP médio: {df['P_VP'].mean():.4f}")

    print(f"\n   Fundos por segmento:")
    print(df.groupby("Segmento")["Sigla"].nunique()
            .sort_values(ascending=False).to_string())

    print(f"\n   Fundos por tipo:")
    print(df.groupby("Tipo_do_Fundo")["Sigla"].nunique()
            .sort_values(ascending=False).to_string())

    # ── Salvar ────────────────────────────────────────────────────────────────
    nome = "dataset_fiis_2019_2024_brapi.xlsx"
    df.to_excel(nome, index=False)
    print(f"\n✅ Salvo: {nome}")
    print(f"\nAmostra (5 linhas):")
    print(df.head(5).to_string(index=False))

    return df

if __name__ == "__main__":
    montar_dataset()