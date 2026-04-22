"""
api.py
------
API FastAPI para previsão de Dividend Yield de FIIs.

Variáveis de ambiente (.env):
    BRAPI_TOKEN = seu_token_aqui   ← nunca coloque no código

Uso local:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /             → status
    GET  /health       → status + métricas do modelo
    GET  /fundos       → lista fundos disponíveis com dados recentes
    POST /predict      → previsão de DY para 1 fundo
    GET  /fii/{sigla}  → busca dados atuais na brapi.dev (token protegido)
"""

import json
import os
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Carrega variáveis de ambiente (.env) ──────────────────────────────────────
# O token da brapi.dev fica aqui — nunca entra no código
load_dotenv()
BRAPI_TOKEN = os.getenv("BRAPI_TOKEN") or os.getenv("API_TOKEN_BRAPI", "")
BRAPI_BASE  = "https://brapi.dev/api"

# ── App ───────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("modelo")

app = FastAPI(
    title="FII Predictor API",
    description="Previsão de Dividend Yield para Fundos de Investimento Imobiliário",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Carregamento de artefatos ─────────────────────────────────────────────────

def carregar_artefatos():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            "modelo/meta.json não encontrado. "
            "Execute: python treinar_modelo.py --arquivo DatasetUsuario.xlsx"
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    modelos = {}
    for nome in meta["modelos"]:
        slug = nome.lower().replace(" ", "_")
        pkl  = MODEL_DIR / f"{slug}.pkl"
        if pkl.exists():
            modelos[nome] = joblib.load(pkl)
            print(f"[✓] Modelo carregado: {nome}")
        else:
            print(f"[!] Modelo não encontrado: {pkl}")

    fundos_path = MODEL_DIR / "fundos_recentes.csv"
    fundos_df = pd.read_csv(fundos_path) if fundos_path.exists() else None

    return meta, modelos, fundos_df


try:
    META, MODELOS, FUNDOS_DF = carregar_artefatos()
except Exception as e:
    print(f"[!] Erro ao carregar artefatos: {e}")
    META, MODELOS, FUNDOS_DF = {}, {}, None


# ── Schemas ───────────────────────────────────────────────────────────────────

class EntradaPredicao(BaseModel):
    sigla:        str            = Field(...,   example="HGLG11")
    dy_lag1:      float          = Field(...,   example=0.0083,  description="DY mês anterior (decimal, ex: 0.0083 = 0.83%)")
    dy_lag2:      float          = Field(...,   example=0.0079,  description="DY 2 meses atrás")
    dy_lag3:      float          = Field(...,   example=0.0081,  description="DY 3 meses atrás")
    p_vp:         Optional[float]= Field(None,  example=1.05)
    vacancia:     Optional[float]= Field(None,  example=0.031,   description="Vacância decimal, ex: 0.031 = 3.1%")
    selic:        Optional[float]= Field(None,  example=0.0113,  description="SELIC mensal decimal")
    ifix:         Optional[float]= Field(None,  example=0.0172,  description="Variação IFIX mensal decimal")
    segmento:     Optional[str]  = Field(None,  example="Logistico")
    tipo_do_fundo:Optional[str]  = Field(None,  example="Fundo de Tijolo")
    modelo:       str            = Field("Gradient Boosting", description="'Random Forest' ou 'Gradient Boosting'")


class SaidaPredicao(BaseModel):
    sigla:           str
    dy_previsto:     float
    dy_previsto_pct: float = Field(description="DY previsto em % ao mês")
    dy_previsto_aa:  float = Field(description="DY previsto em % ao ano (estimativa)")
    modelo_usado:    str
    r2:              Optional[float] = None
    mape:            Optional[float] = None


class InfoFundo(BaseModel):
    sigla:         str
    dy_recente:    Optional[float] = None
    dy_lag1:       Optional[float] = None
    dy_lag2:       Optional[float] = None
    dy_lag3:       Optional[float] = None
    p_vp:          Optional[float] = None
    vacancia:      Optional[float] = None
    selic:         Optional[float] = None
    ifix:          Optional[float] = None
    segmento:      Optional[str]   = None
    tipo_do_fundo: Optional[str]   = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def raiz():
    return {
        "status": "online",
        "modelos_disponiveis": list(MODELOS.keys()),
        "versao": "2.0.0",
        "brapi_configurado": bool(BRAPI_TOKEN),
    }


@app.get("/health", tags=["Status"])
def health():
    metricas = META.get("metricas", {}) if META else {}
    return {
        "status": "ok",
        "modelos": len(MODELOS),
        "melhor_modelo": META.get("melhor_modelo") if META else None,
        "metricas": metricas,
        "n_amostras_treino": META.get("n_amostras_treino") if META else None,
    }


@app.get("/fundos", tags=["Dados"], response_model=list[InfoFundo])
def listar_fundos():
    """Lista fundos disponíveis com os dados mais recentes para pré-preencher o formulário."""
    if FUNDOS_DF is None:
        return []

    resultado = []
    col_sigla = META.get("col_sigla", "Sigla") if META else "Sigla"
    col_dy    = META.get("col_dy",    "Dividendos_Yield") if META else "Dividendos_Yield"

    for _, row in FUNDOS_DF.iterrows():
        resultado.append(InfoFundo(
            sigla         = str(row.get(col_sigla, "")),
            dy_recente    = _safe_float(row.get(col_dy)),
            dy_lag1       = _safe_float(row.get("DY_lag1")),
            dy_lag2       = _safe_float(row.get("DY_lag2")),
            dy_lag3       = _safe_float(row.get("DY_lag3")),
            p_vp          = _safe_float(row.get("P_VP")),
            vacancia      = _safe_float(row.get("Vacancia")),
            selic         = _safe_float(row.get("SELIC")),
            ifix          = _safe_float(row.get("IFIX")),
            segmento      = _safe_str(row.get("Segmento")),
            tipo_do_fundo = _safe_str(row.get("Tipo_do_Fundo")),
        ))
    return resultado


@app.post("/predict", tags=["Previsão"], response_model=SaidaPredicao)
def prever(entrada: EntradaPredicao):
    """Previsão de DY para o próximo mês usando o modelo treinado."""
    if not MODELOS:
        raise HTTPException(503, detail="Nenhum modelo carregado. Execute treinar_modelo.py primeiro.")

    nome_modelo = entrada.modelo
    if nome_modelo not in MODELOS:
        raise HTTPException(400, detail=f"Modelo '{nome_modelo}' inválido. Use: {list(MODELOS.keys())}")

    pipe     = MODELOS[nome_modelo]
    num_cols = META.get("num_cols", ["DY_lag1", "DY_lag2", "DY_lag3"]) if META else ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = META.get("cat_cols", []) if META else []

    # Monta linha de entrada espelhando o treino
    row: dict = {
        "DY_lag1": entrada.dy_lag1,
        "DY_lag2": entrada.dy_lag2,
        "DY_lag3": entrada.dy_lag3,
    }
    mapa_opcional = {
        "P_VP":         entrada.p_vp,
        "Vacancia":     entrada.vacancia,
        "SELIC":        entrada.selic,
        "IFIX":         entrada.ifix,
        "Segmento":     entrada.segmento,
        "Tipo_do_Fundo":entrada.tipo_do_fundo,
    }
    for col, val in mapa_opcional.items():
        if col in num_cols + cat_cols:
            row[col] = val if val is not None else np.nan

    X = pd.DataFrame([row])[num_cols + cat_cols]

    try:
        dy_previsto = float(pipe.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, detail=f"Erro na predição: {str(e)}")

    # Pega métricas do modelo escolhido
    metricas_modelo = (META.get("metricas", {}).get(nome_modelo, {}) if META else {})

    return SaidaPredicao(
        sigla           = entrada.sigla.upper(),
        dy_previsto     = round(dy_previsto, 6),
        dy_previsto_pct = round(dy_previsto * 100, 4),
        dy_previsto_aa  = round(dy_previsto * 12 * 100, 2),
        modelo_usado    = nome_modelo,
        r2              = metricas_modelo.get("r2"),
        mape            = metricas_modelo.get("mape"),
    )


@app.get("/fii/{sigla}", tags=["brapi.dev"])
async def get_fii_brapi(sigla: str):
    """
    Busca dados atuais do FII na brapi.dev.
    O token fica protegido no servidor — nunca chega ao navegador.
    """
    if not BRAPI_TOKEN:
        raise HTTPException(503, detail="BRAPI_TOKEN não configurado. Defina no .env")

    url = f"{BRAPI_BASE}/funds/{sigla.upper()}?token={BRAPI_TOKEN}"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url)
            if r.status_code == 404:
                raise HTTPException(404, detail=f"FII {sigla.upper()} não encontrado na brapi.dev")
            if r.status_code != 200:
                raise HTTPException(r.status_code, detail=f"Erro brapi.dev: {r.status_code}")

            data    = r.json()
            results = data.get("results", [])
            if not results:
                raise HTTPException(404, detail=f"FII {sigla.upper()} sem dados na brapi.dev")

            fundo = results[0]
            return {
                "sigla":               sigla.upper(),
                "nome":                fundo.get("longName", ""),
                "cotacao":             fundo.get("regularMarketPrice"),
                "pvp":                 fundo.get("priceToBook") or fundo.get("pvp"),
                "vacancia_fisica":     fundo.get("physicalVacancy"),
                "vacancia_financeira": fundo.get("financialVacancy"),
                "ultimo_rendimento":   fundo.get("lastDividend"),
                "patrimonio_liq":      fundo.get("netWorth"),
                "fonte":               "brapi.dev",
            }
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Timeout ao acessar brapi.dev")


@app.get("/fii/{sigla}/historico", tags=["brapi.dev"])
async def get_fii_historico(sigla: str, meses: int = 3):
    """
    Busca histórico mensal do FII na brapi.dev para usar como lags no modelo.
    Retorna os últimos N meses de cotação.
    """
    if not BRAPI_TOKEN:
        raise HTTPException(503, detail="BRAPI_TOKEN não configurado. Defina no .env")

    url = f"{BRAPI_BASE}/quote/{sigla.upper()}?range=1y&interval=1mo&token={BRAPI_TOKEN}"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(r.status_code, detail=f"Erro brapi.dev: {r.status_code}")

            hist = r.json().get("results", [{}])[0].get("historicalDataPrice", [])
            if not hist:
                raise HTTPException(404, detail="Histórico não disponível")

            df = pd.DataFrame(hist[-meses:])
            df["date"] = pd.to_datetime(df["date"], unit="s")

            return {
                "sigla":    sigla.upper(),
                "meses":    meses,
                "historico": [
                    {
                        "data":       row["date"].strftime("%Y-%m"),
                        "fechamento": round(float(row.get("close", 0)), 4),
                    }
                    for _, row in df.iterrows()
                ],
                "fonte": "brapi.dev",
            }
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Timeout ao acessar brapi.dev")


# ── Utilidades ────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 6)
    except Exception:
        return None


def _safe_str(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip() or None


# ── Entry point local ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
