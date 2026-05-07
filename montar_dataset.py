"""
montar_dataset.py
─────────────────
Monta dataset de FIIs (jan/2019 – dez/2024).

Fontes:
  - Dados de Mercado API → FIIs, dividendos, cotações, SELIC
  - BCB SGS 12466        → IFIX mensal
  - CVM Informe Mensal   → Segmento e Tipo (fallback)

Colunas: Data, Sigla, Segmento, Tipo_do_Fundo,
         Dividendos_Yield, SELIC, IFIX

Pré-requisito .env:
    DADOS_MERCADO_TOKEN=seu_token

Uso:
    python montar_dataset.py
    python montar_dataset.py --minimo 48
"""

import os, time, argparse
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DDM_TOKEN = os.getenv("DADOS_MERCADO_TOKEN", "")
if not DDM_TOKEN:
    raise RuntimeError(
        "Token não encontrado.\n"
        "Adicione no .env:\n"
        "  DADOS_MERCADO_TOKEN=seu_token"
    )

DDM_BASE  = "https://api.dadosdemercado.com.br/v1"
BCB_BASE  = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"

MES_INI = "2019-01"
MES_FIM = "2024-12"
MINIMO_MESES = 48

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def mes_fmt(dt) -> str:
    return pd.to_datetime(dt).strftime("%Y-%m")

def safe_get(url, headers=None, params=None):
    for tentativa in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                print("⏳ rate limit...", end=" ")
                time.sleep(10)
            else:
                return None
        except Exception:
            time.sleep(2)
    return None

def ddm_get(path, params=None):
    """Requisição autenticada à API Dados de Mercado."""
    return safe_get(
        f"{DDM_BASE}{path}",
        headers={"Authorization": f"Bearer {DDM_TOKEN}"},
        params=params
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. SELIC — Dados de Mercado /macro/selic
# ─────────────────────────────────────────────────────────────────────────────

def buscar_selic() -> dict:
    print("📡 SELIC (Dados de Mercado)...", end=" ")
    r = ddm_get("/macro/selic")
    if not r:
        print("⚠ falhou")
        return {}
    result = {}
    for item in r.json():
        try:
            mes = mes_fmt(item["date"])
            if MES_INI <= mes <= MES_FIM:
                result[mes] = round(float(item["value"]) / 100, 6)
        except Exception:
            continue
    print(f"✓ {len(result)} meses")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 2. IFIX — BCB SGS 12466
# ─────────────────────────────────────────────────────────────────────────────

def buscar_ifix() -> dict:
    print("📡 IFIX (BCB)...", end=" ")
    r = safe_get(f"{BCB_BASE}.12466/dados", params={
        "formato": "json",
        "dataInicial": "01/12/2018",
        "dataFinal":   "31/12/2024"
    })
    if not r:
        print("⚠ falhou")
        return {}
    serie = {}
    for item in r.json():
        try:
            mes = mes_fmt(pd.to_datetime(item["data"], format="%d/%m/%Y"))
            serie[mes] = float(item["valor"])
        except Exception:
            continue
    meses = sorted(serie.keys())
    result = {}
    for i in range(1, len(meses)):
        m, ma = meses[i], meses[i-1]
        if MES_INI <= m <= MES_FIM and serie[ma] > 0:
            result[m] = round((serie[m] / serie[ma]) - 1, 6)
    print(f"✓ {len(result)} meses")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 3. Lista de FIIs — Dados de Mercado /reits
# ─────────────────────────────────────────────────────────────────────────────

def buscar_lista_fiis() -> list[dict]:
    """Retorna lista de FIIs com ticker, segmento e tipo."""
    print("📡 Lista de FIIs (Dados de Mercado)...", end=" ")
    r = ddm_get("/reits")
    if not r:
        print("⚠ falhou")
        return []
    fiis = r.json()
    # Filtra só os listados na B3
    listados = [f for f in fiis if f.get("is_b3_listed")]
    print(f"✓ {len(listados)} FIIs listados na B3")
    return listados

# ─────────────────────────────────────────────────────────────────────────────
# 4. Dividendos — Dados de Mercado /reits/{ticker}/dividends
# ─────────────────────────────────────────────────────────────────────────────

def buscar_dividendos(ticker: str) -> dict:
    """
    Retorna {mes: valor_dividendo} para o período 2019–2024.
    Soma dividendos do mesmo mês (alguns FIIs pagam 2x/mês).
    """
    r = ddm_get(f"/reits/{ticker}/dividends", params={"date_from": "2019-01-01"})
    if not r:
        return {}
    result = {}
    for item in r.json():
        try:
            # Usa payable_date como referência do mês
            data = (
                item.get("payable_date") or
                item.get("record_date") or
                item.get("ex_date", "")
            )
            if not data:
                continue
            mes = mes_fmt(data[:10])
            if MES_INI <= mes <= MES_FIM:
                val = float(item.get("amount") or 0)
                if val > 0:
                    result[mes] = result.get(mes, 0) + val
        except Exception:
            continue
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 5. Cotações — Dados de Mercado /funds/{id}/quotes
# ─────────────────────────────────────────────────────────────────────────────

def buscar_cotacoes(fund_id: str) -> dict:
    """
    Retorna {mes: valor_cota} — cotação patrimonial mensal.
    A API retorna valores em centavos, então dividimos por 100.
    """
    r = ddm_get(f"/funds/{fund_id}/quotes", params={
        "period_init": "2019-01-01",
        "period_end":  "2024-12-31"
    })
    if not r:
        return {}
    cotacoes_mes = {}
    for item in r.json():
        try:
            mes  = mes_fmt(item["date"])
            cota = float(item.get("quote") or 0) / 100  # centavos → reais
            if cota > 0 and MES_INI <= mes <= MES_FIM:
                # Mantém a última cotação do mês
                cotacoes_mes[mes] = cota
        except Exception:
            continue
    return cotacoes_mes

# ─────────────────────────────────────────────────────────────────────────────
# 6. Montar dataset
# ─────────────────────────────────────────────────────────────────────────────

def montar_dataset(minimo_meses: int = MINIMO_MESES):
    meses_periodo = [str(m) for m in pd.period_range(MES_INI, MES_FIM, freq="M")]

    selic = buscar_selic()
    ifix  = buscar_ifix()
    fiis  = buscar_lista_fiis()

    if not fiis:
        print("\n❌ Não foi possível obter a lista de FIIs.")
        return None

    print(f"\n{'─'*58}")
    print(f"📡 Processando {len(fiis)} FIIs | mínimo {minimo_meses} meses com DY")
    print(f"{'─'*58}\n")

    todas_linhas = []
    resumo       = []

    for fii in fiis:
        ticker   = fii.get("b3_trade_name") or fii.get("trade_name", "")
        segmento = fii.get("b3_segment") or fii.get("b3_subsector", "")
        tipo     = fii.get("b3_subsector") or fii.get("b3_sector", "")
        fund_id  = fii.get("id", "")

        if not ticker:
            continue

        print(f"  {ticker}...", end=" ", flush=True)

        dividendos = buscar_dividendos(ticker)
        time.sleep(0.3)
        cotacoes   = buscar_cotacoes(fund_id) if fund_id else {}
        time.sleep(0.3)

        meses_com_dy = 0
        for mes in meses_periodo:
            div  = dividendos.get(mes)
            cota = cotacoes.get(mes)
            dy   = round(div / cota, 6) if (div and cota and cota > 0) else None

            todas_linhas.append({
                "Data":             mes,
                "Sigla":            ticker,
                "Segmento":         segmento,
                "Tipo_do_Fundo":    tipo,
                "Dividendos_Yield": dy,
                "SELIC":            selic.get(mes),
                "IFIX":             ifix.get(mes),
            })
            if dy:
                meses_com_dy += 1

        resumo.append({"sigla": ticker, "meses": meses_com_dy})
        print(f"✓ {meses_com_dy} meses com DY")

    # ── Filtra fundos com mínimo de meses ─────────────────────────────────────
    aprovados = {r["sigla"] for r in resumo if r["meses"] >= minimo_meses}
    df = pd.DataFrame(todas_linhas)
    df = df[df["Sigla"].isin(aprovados)]
    df = df.dropna(subset=["Dividendos_Yield"])
    df = df.sort_values(["Sigla", "Data"]).reset_index(drop=True)

    if df.empty:
        print(f"\n⚠ Nenhum fundo com {minimo_meses}+ meses. Tentando com 36...")
        aprovados = {r["sigla"] for r in resumo if r["meses"] >= 36}
        df = pd.DataFrame(todas_linhas)
        df = df[df["Sigla"].isin(aprovados)]
        df = df.dropna(subset=["Dividendos_Yield"])
        df = df.sort_values(["Sigla", "Data"]).reset_index(drop=True)

    if df.empty:
        print("\n❌ Dataset vazio.")
        return None

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"📊 Dataset final: {len(df)} linhas · {df['Sigla'].nunique()} fundos")
    print(f"   Período: {df['Data'].min()} → {df['Data'].max()}")
    dy = df["Dividendos_Yield"]
    print(f"   DY — média: {dy.mean():.4f} | min: {dy.min():.4f} | max: {dy.max():.4f}")

    print(f"\n   Fundos por segmento:")
    print(df.groupby("Segmento")["Sigla"].nunique()
            .sort_values(ascending=False).to_string())

    excluidos = [r for r in resumo if r["sigla"] not in aprovados]
    print(f"\n   Excluídos: {len(excluidos)} fundos com menos de {minimo_meses} meses")

    # ── Salvar ────────────────────────────────────────────────────────────────
    nome = "dataset_fiis_2019_2024.xlsx"
    df.to_excel(nome, index=False)
    print(f"\n✅ Salvo: {nome}")
    print(f"\nAmostra (5 linhas):")
    print(df.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimo", type=int, default=MINIMO_MESES,
                        help=f"Mínimo de meses com DY (padrão: {MINIMO_MESES})")
    args = parser.parse_args()
    montar_dataset(args.minimo)