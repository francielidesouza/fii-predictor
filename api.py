"""
api.py
------
API FastAPI para previsão de DY de FIIs.
Sem vacância. Aceita SELIC e IFIX como campos opcionais.

Uso local:
    uvicorn api:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


def carregar_artefatos():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError("modelo/meta.json não encontrado. Execute treinar_modelo.py primeiro.")
    with open(meta_path) as f:
        meta = json.load(f)
    modelos = {}
    for nome in meta["modelos"]:
        slug = nome.lower().replace(" ", "_")
        pkl = MODEL_DIR / f"{slug}.pkl"
        if pkl.exists():
            modelos[nome] = joblib.load(pkl)
    fundos_path = MODEL_DIR / "fundos_recentes.csv"
    fundos_df = pd.read_csv(fundos_path) if fundos_path.exists() else None
    return meta, modelos, fundos_df


try:
    META, MODELOS, FUNDOS_DF = carregar_artefatos()
    print(f"[✓] Modelos carregados: {list(MODELOS.keys())}")
except Exception as e:
    print(f"[!] Erro ao carregar artefatos: {e}")
    META, MODELOS, FUNDOS_DF = {}, {}, None


class EntradaPredicao(BaseModel):
    sigla: str          = Field(..., example="HGLG11")
    dy_lag1: float      = Field(..., example=0.0082, description="DY mês anterior (decimal)")
    dy_lag2: float      = Field(..., example=0.0079, description="DY 2 meses atrás (decimal)")
    dy_lag3: float      = Field(..., example=0.0081, description="DY 3 meses atrás (decimal)")
    p_vp: Optional[float]          = Field(None, example=1.05)
    selic: Optional[float]         = Field(None, example=0.0092, description="SELIC do mês (decimal)")
    ifix: Optional[float]          = Field(None, example=0.0058, description="Variação IFIX do mês (decimal)")
    segmento: Optional[str]        = Field(None, example="Logistico")
    tipo_do_fundo: Optional[str]   = Field(None, example="Fundo de Tijolo")
    modelo: str = Field("Gradient Boosting", description="'Random Forest' ou 'Gradient Boosting'")


class SaidaPredicao(BaseModel):
    sigla: str
    dy_previsto: float
    modelo_usado: str
    unidade: str = "decimal (% ao mês)"


class InfoFundo(BaseModel):
    sigla: str
    dy_recente: Optional[float] = None
    p_vp: Optional[float] = None
    segmento: Optional[str] = None
    tipo_do_fundo: Optional[str] = None


@app.get("/", tags=["Status"])
def raiz():
    return {"status": "online", "modelos_disponiveis": list(MODELOS.keys()), "versao": "2.0.0"}


@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "modelos": len(MODELOS)}


@app.get("/fundos", tags=["Dados"], response_model=list[InfoFundo])
def listar_fundos():
    if FUNDOS_DF is None:
        return []
    col_sigla = META.get("col_sigla", "Sigla")
    col_dy    = META.get("col_dy", "Dividendos_Yield")
    resultado = []
    for _, row in FUNDOS_DF.iterrows():
        resultado.append(InfoFundo(
            sigla         = str(row.get(col_sigla, "")),
            dy_recente    = _safe_float(row.get(col_dy)),
            p_vp          = _safe_float(row.get("P_VP")),
            segmento      = _safe_str(row.get("Segmento")),
            tipo_do_fundo = _safe_str(row.get("Tipo_do_Fundo")),
        ))
    return resultado


@app.post("/predict", tags=["Previsão"], response_model=SaidaPredicao)
def prever(entrada: EntradaPredicao):
    if not MODELOS:
        raise HTTPException(status_code=503, detail="Nenhum modelo carregado.")

    nome_modelo = entrada.modelo
    if nome_modelo not in MODELOS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{nome_modelo}' não disponível. Use: {list(MODELOS.keys())}"
        )

    pipe     = MODELOS[nome_modelo]
    num_cols = META.get("num_cols", ["DY_lag1", "DY_lag2", "DY_lag3"])
    cat_cols = META.get("cat_cols", [])

    row = {
        "DY_lag1": entrada.dy_lag1,
        "DY_lag2": entrada.dy_lag2,
        "DY_lag3": entrada.dy_lag3,
    }

    # Features opcionais — usa valor fornecido ou mediana do treino (NaN → preenchido no pipe)
    if "P_VP" in num_cols:
        row["P_VP"] = entrada.p_vp if entrada.p_vp is not None else np.nan
    if "SELIC" in num_cols:
        row["SELIC"] = entrada.selic if entrada.selic is not None else np.nan
    if "IFIX" in num_cols:
        row["IFIX"] = entrada.ifix if entrada.ifix is not None else np.nan
    if "Segmento" in cat_cols:
        row["Segmento"] = entrada.segmento or ""
    if "Tipo_do_Fundo" in cat_cols:
        row["Tipo_do_Fundo"] = entrada.tipo_do_fundo or ""

    X = pd.DataFrame([row])[num_cols + cat_cols]

    try:
        dy_previsto = float(pipe.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

    return SaidaPredicao(
        sigla        = entrada.sigla.upper(),
        dy_previsto  = round(dy_previsto, 6),
        modelo_usado = nome_modelo,
    )


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
