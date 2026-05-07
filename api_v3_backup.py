"""
api.py v3
---------
API FastAPI para previsão de Dividend Yield de FIIs.
Suporta Random Forest, Gradient Boosting, XGBoost e Prophet.
Features de calendário (mes/trimestre/semestre) incluídas automaticamente.

Variáveis de ambiente (.env):
    BRAPI_TOKEN = seu_token_aqui

Uso local:
    uvicorn api:app --reload --port 8000
    Docs: http://localhost:8000/docs
"""

import json
import os
from datetime import datetime
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

load_dotenv()
BRAPI_TOKEN = os.getenv("BRAPI_TOKEN") or os.getenv("API_TOKEN_BRAPI", "")
BRAPI_BASE  = "https://brapi.dev/api"

MODEL_DIR = Path("modelo")

app = FastAPI(
    title="FII Predictor API v3",
    description="Previsão de Dividend Yield com features de calendário e comparação de modelos",
    version="3.0.0",
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
    for nome in meta.get("modelos", []):
        slug = nome.lower().replace(" ", "_")
        pkl  = MODEL_DIR / f"{slug}.pkl"
        if pkl.exists():
            modelos[nome] = joblib.load(pkl)
            print(f"[✓] {nome} carregado ({pkl.stat().st_size/1024:.0f} KB)")
        else:
            print(f"[!] {pkl} não encontrado")

    fundos_path = MODEL_DIR / "fundos_recentes.csv"
    fundos_df   = pd.read_csv(fundos_path) if fundos_path.exists() else None

    return meta, modelos, fundos_df


try:
    META, MODELOS, FUNDOS_DF = carregar_artefatos()
except Exception as e:
    print(f"[!] Erro ao carregar artefatos: {e}")
    META, MODELOS, FUNDOS_DF = {}, {}, None


# ── Helpers ───────────────────────────────────────────────────────────────────

def features_calendario(data_ref: Optional[str] = None) -> dict:
    """
    Gera features de calendário para uma data.
    Se não informada, usa o mês seguinte ao atual (próxima previsão).
    """
    if data_ref:
        dt = pd.to_datetime(data_ref)
    else:
        hoje = datetime.now()
        # Próximo mês — que é o que o modelo vai prever
        mes_prox = hoje.month + 1 if hoje.month < 12 else 1
        ano_prox = hoje.year if hoje.month < 12 else hoje.year + 1
        dt = datetime(ano_prox, mes_prox, 1)

    return {
        "mes":       float(dt.month),
        "trimestre": float((dt.month - 1) // 3 + 1),
        "semestre":  float(1 if dt.month > 6 else 0),
    }


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


# ── Schemas ───────────────────────────────────────────────────────────────────

class EntradaPredicao(BaseModel):
    sigla:         str            = Field(...,   example="HGLG11")
    dy_lag1:       float          = Field(...,   example=0.0082,  description="DY mês anterior (decimal)")
    dy_lag2:       float          = Field(...,   example=0.0079,  description="DY 2 meses atrás")
    dy_lag3:       float          = Field(...,   example=0.0081,  description="DY 3 meses atrás")
    data_referencia: Optional[str]= Field(None,  example="2025-02-01", description="Data da previsão (YYYY-MM-DD). Padrão: próximo mês")
    selic:         Optional[float]= Field(None,  example=0.0113)
    ifix:          Optional[float]= Field(None,  example=0.0172)
    segmento:      Optional[str]  = Field(None,  example="Logistico")
    tipo_do_fundo: Optional[str]  = Field(None,  example="Fundo de Tijolo")
    modelo:        str            = Field("Gradient Boosting",
                                          description="'Random Forest', 'Gradient Boosting' ou 'XGBoost'")


class SaidaPredicao(BaseModel):
    sigla:           str
    dy_previsto:     float
    dy_previsto_pct: float
    dy_previsto_aa:  float
    modelo_usado:    str
    data_referencia: str
    features_calendario: dict
    r2:              Optional[float] = None
    mape:            Optional[float] = None


class InfoFundo(BaseModel):
    sigla:         str
    dy_recente:    Optional[float] = None
    dy_lag1:       Optional[float] = None
    dy_lag2:       Optional[float] = None
    dy_lag3:       Optional[float] = None
    selic:         Optional[float] = None
    ifix:          Optional[float] = None
    segmento:      Optional[str]   = None
    tipo_do_fundo: Optional[str]   = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def raiz():
    return {
        "status":              "online",
        "versao":              "3.0.0",
        "modelos_disponiveis": list(MODELOS.keys()),
        "features_calendario": META.get("features_calendario", []) if META else [],
        "brapi_configurado":   bool(BRAPI_TOKEN),
    }


@app.get("/health", tags=["Status"])
def health():
    metricas = META.get("metricas", {}) if META else {}
    return {
        "status":              "ok",
        "modelos":             len(MODELOS),
        "melhor_modelo":       META.get("melhor_modelo") if META else None,
        "metricas":            metricas,
        "n_amostras_treino":   META.get("n_amostras_treino") if META else None,
        "n_fundos":            META.get("n_fundos") if META else None,
        "versao_modelo":       META.get("versao") if META else None,
    }


@app.get("/fundos", tags=["Dados"], response_model=list[InfoFundo])
def listar_fundos():
    """Lista fundos disponíveis com dados mais recentes para pré-preencher o formulário."""
    if FUNDOS_DF is None:
        return []
    col_sigla = META.get("col_sigla", "Sigla") if META else "Sigla"
    col_dy    = META.get("col_dy", "Dividendos_Yield") if META else "Dividendos_Yield"
    resultado = []
    for _, row in FUNDOS_DF.iterrows():
        resultado.append(InfoFundo(
            sigla         = str(row.get(col_sigla, "")),
            dy_recente    = _safe_float(row.get(col_dy)),
            dy_lag1       = _safe_float(row.get("DY_lag1")),
            dy_lag2       = _safe_float(row.get("DY_lag2")),
            dy_lag3       = _safe_float(row.get("DY_lag3")),
            selic         = _safe_float(row.get("SELIC")),
            ifix          = _safe_float(row.get("IFIX")),
            segmento      = _safe_str(row.get("Segmento")),
            tipo_do_fundo = _safe_str(row.get("Tipo_do_Fundo")),
        ))
    return resultado


@app.post("/predict", tags=["Previsão"], response_model=SaidaPredicao)
def prever(entrada: EntradaPredicao):
    """
    Previsão de DY para o próximo mês.
    Features de calendário são geradas automaticamente com base na data de referência.
    """
    if not MODELOS:
        raise HTTPException(503, detail="Nenhum modelo carregado. Execute treinar_modelo.py primeiro.")

    nome_modelo = entrada.modelo
    if nome_modelo not in MODELOS:
        raise HTTPException(400, detail=f"Modelo '{nome_modelo}' inválido. Disponíveis: {list(MODELOS.keys())}")

    pipe     = MODELOS[nome_modelo]
    num_cols = META.get("num_cols", ["DY_lag1", "DY_lag2", "DY_lag3"]) if META else ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = META.get("cat_cols", []) if META else []

    # Features de calendário automáticas
    cal = features_calendario(entrada.data_referencia)

    # Monta linha de entrada
    row: dict = {
        "DY_lag1":   entrada.dy_lag1,
        "DY_lag2":   entrada.dy_lag2,
        "DY_lag3":   entrada.dy_lag3,
        "mes":       cal["mes"],
        "trimestre": cal["trimestre"],
        "semestre":  cal["semestre"],
    }

    # Opcionais — só inclui se o modelo foi treinado com elas
    mapa_opt = {
        "SELIC":        entrada.selic,
        "IFIX":         entrada.ifix,
        "Segmento":     entrada.segmento,
        "Tipo_do_Fundo":entrada.tipo_do_fundo,
    }
    for col, val in mapa_opt.items():
        if col in num_cols + cat_cols:
            row[col] = val if val is not None else np.nan

    X = pd.DataFrame([row])[num_cols + cat_cols]

    try:
        dy = float(pipe.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, detail=f"Erro na predição: {str(e)}")

    metricas = (META.get("metricas", {}).get(nome_modelo, {}) if META else {})
    data_ref = entrada.data_referencia or datetime.now().strftime("%Y-%m-01")

    return SaidaPredicao(
        sigla               = entrada.sigla.upper(),
        dy_previsto         = round(dy, 6),
        dy_previsto_pct     = round(dy * 100, 4),
        dy_previsto_aa      = round(dy * 12 * 100, 2),
        modelo_usado        = nome_modelo,
        data_referencia     = data_ref,
        features_calendario = cal,
        r2                  = metricas.get("r2"),
        mape                = metricas.get("mape"),
    )


@app.post("/predict/comparar", tags=["Previsão"])
def comparar_modelos(entrada: EntradaPredicao):
    """
    Roda todos os modelos disponíveis e retorna uma comparação lado a lado.
    Útil para a seção de resultados do TCC.
    """
    if not MODELOS:
        raise HTTPException(503, detail="Nenhum modelo carregado.")

    num_cols = META.get("num_cols", ["DY_lag1", "DY_lag2", "DY_lag3"]) if META else ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = META.get("cat_cols", []) if META else []
    cal      = features_calendario(entrada.data_referencia)

    row = {
        "DY_lag1": entrada.dy_lag1, "DY_lag2": entrada.dy_lag2,
        "DY_lag3": entrada.dy_lag3, "mes": cal["mes"],
        "trimestre": cal["trimestre"], "semestre": cal["semestre"],
    }
    for col, val in {
                     "SELIC": entrada.selic, "IFIX": entrada.ifix,
                     "Segmento": entrada.segmento, "Tipo_do_Fundo": entrada.tipo_do_fundo}.items():
        if col in num_cols + cat_cols:
            row[col] = val if val is not None else np.nan

    X = pd.DataFrame([row])[num_cols + cat_cols]
    resultados = []

    for nome, pipe in MODELOS.items():
        try:
            dy = float(pipe.predict(X)[0])
            metricas = META.get("metricas", {}).get(nome, {}) if META else {}
            resultados.append({
                "modelo":       nome,
                "dy_previsto":  round(dy, 6),
                "dy_pct_am":    round(dy * 100, 4),
                "dy_pct_aa":    round(dy * 12 * 100, 2),
                "r2":           metricas.get("r2"),
                "mape":         metricas.get("mape"),
                "cv_r2":        metricas.get("cv_r2"),
            })
        except Exception as e:
            resultados.append({"modelo": nome, "erro": str(e)})

    resultados.sort(key=lambda x: x.get("r2") or 0, reverse=True)
    return {
        "sigla":             entrada.sigla.upper(),
        "features_calendario": cal,
        "comparacao":        resultados,
        "melhor_modelo":     META.get("melhor_modelo") if META else None,
    }


@app.get("/fii/{sigla}", tags=["brapi.dev"])
async def get_fii_brapi(sigla: str):
    """Dados atuais do FII via brapi.dev. Token protegido no servidor."""
    if not BRAPI_TOKEN:
        raise HTTPException(503, detail="BRAPI_TOKEN não configurado no .env")
    url = f"{BRAPI_BASE}/funds/{sigla.upper()}?token={BRAPI_TOKEN}"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(r.status_code, detail=f"brapi.dev: {r.status_code}")
            results = r.json().get("results", [])
            if not results:
                raise HTTPException(404, detail=f"{sigla.upper()} não encontrado")
            f = results[0]
            return {
                "sigla":             sigla.upper(),
                "nome":              f.get("longName", ""),
                "cotacao":           f.get("regularMarketPrice"),
                "ultimo_rendimento": f.get("lastDividend"),
                "patrimonio_liq":    f.get("netWorth"),
                "fonte":             "brapi.dev",
            }
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Timeout na brapi.dev")


@app.get("/fii/{sigla}/historico", tags=["brapi.dev"])
async def get_fii_historico(sigla: str, meses: int = 3):
    """Histórico mensal do FII para usar como lags. Token protegido no servidor."""
    if not BRAPI_TOKEN:
        raise HTTPException(503, detail="BRAPI_TOKEN não configurado no .env")
    url = f"{BRAPI_BASE}/quote/{sigla.upper()}?range=1y&interval=1mo&token={BRAPI_TOKEN}"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(r.status_code, detail=f"brapi.dev: {r.status_code}")
            hist = r.json().get("results", [{}])[0].get("historicalDataPrice", [])
            if not hist:
                raise HTTPException(404, detail="Histórico não disponível")
            df = pd.DataFrame(hist[-meses:])
            df["date"] = pd.to_datetime(df["date"], unit="s")
            return {
                "sigla": sigla.upper(), "meses": meses, "fonte": "brapi.dev",
                "historico": [
                    {"data": row["date"].strftime("%Y-%m"), "fechamento": round(float(row.get("close", 0)), 4)}
                    for _, row in df.iterrows()
                ],
            }
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Timeout na brapi.dev")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
