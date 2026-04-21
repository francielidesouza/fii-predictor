"""
api.py
------
API FastAPI para servir previsões de Dividend Yield de FIIs.
Gerada para deploy no Render.com ou Railway.app.

Estrutura esperada na pasta do projeto:
    api.py
    requirements.txt
    modelo/
        random_forest.pkl
        gradient_boosting.pkl
        meta.json
        fundos_recentes.csv   (opcional — para autocompletar)

Uso local:
    uvicorn api:app --reload --port 8000

Teste rápido:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"sigla":"HGLG11","dy_lag1":0.9,"dy_lag2":0.85,"dy_lag3":0.88}'
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

# ── Configuração ──────────────────────────────────────────────────────────────

MODEL_DIR = Path("modelo")

app = FastAPI(
    title="FII Predictor API",
    description="Previsão de Dividend Yield para Fundos de Investimento Imobiliário",
    version="1.0.0",
)

# CORS — permite que o frontend no Vercel chame esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # em produção: substitua pelo domínio Vercel exato
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Carregamento de artefatos ─────────────────────────────────────────────────

def carregar_artefatos():
    """Carrega modelos e metadados ao iniciar a aplicação."""
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            "Arquivo modelo/meta.json não encontrado. "
            "Execute treinar_modelo.py primeiro."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    modelos = {}
    for nome in meta["modelos"]:
        slug = nome.lower().replace(" ", "_")
        pkl = MODEL_DIR / f"{slug}.pkl"
        if pkl.exists():
            modelos[nome] = joblib.load(pkl)
        else:
            print(f"[!] Modelo não encontrado: {pkl}")

    # Carrega lista de fundos recentes (para o endpoint /fundos)
    fundos_path = MODEL_DIR / "fundos_recentes.csv"
    fundos_df = pd.read_csv(fundos_path) if fundos_path.exists() else None

    return meta, modelos, fundos_df


try:
    META, MODELOS, FUNDOS_DF = carregar_artefatos()
    print(f"[✓] Modelos carregados: {list(MODELOS.keys())}")
except Exception as e:
    print(f"[!] Erro ao carregar artefatos: {e}")
    META, MODELOS, FUNDOS_DF = {}, {}, None


# ── Schemas de entrada/saída ──────────────────────────────────────────────────

class EntradaPredicao(BaseModel):
    sigla: str = Field(..., example="HGLG11", description="Sigla do FII")
    dy_lag1: float = Field(..., example=0.90, description="DY do mês anterior (%)")
    dy_lag2: float = Field(..., example=0.85, description="DY de 2 meses atrás (%)")
    dy_lag3: float = Field(..., example=0.88, description="DY de 3 meses atrás (%)")
    p_vp: Optional[float] = Field(None, example=0.95, description="Preço / Valor Patrimonial")
    vacancia: Optional[float] = Field(None, example=5.2,  description="Vacância física (%)")
    segmento: Optional[str]  = Field(None, example="Logística", description="Segmento do fundo")
    tipo_do_fundo: Optional[str] = Field(None, example="Tijolo",  description="Tipo do fundo")
    modelo: str = Field("Random Forest", description="Modelo a usar: 'Random Forest' ou 'Gradient Boosting'")


class SaidaPredicao(BaseModel):
    sigla: str
    dy_previsto: float
    modelo_usado: str
    unidade: str = "% ao mês"


class InfoFundo(BaseModel):
    sigla: str
    dy_recente: Optional[float] = None
    p_vp: Optional[float] = None
    vacancia: Optional[float] = None
    segmento: Optional[str] = None
    tipo_do_fundo: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def raiz():
    """Verifica se a API está no ar."""
    return {
        "status": "online",
        "modelos_disponiveis": list(MODELOS.keys()),
        "versao": "1.0.0",
    }


@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "modelos": len(MODELOS)}


@app.get("/fundos", tags=["Dados"], response_model=list[InfoFundo])
def listar_fundos():
    """
    Retorna a lista de fundos no dataset com dados mais recentes.
    Útil para popular o formulário no frontend.
    """
    if FUNDOS_DF is None:
        return []

    col_sigla = META.get("col_sigla", "Sigla")
    col_dy    = META.get("col_dy",    "Dividendos_Yield")

    resultado = []
    for _, row in FUNDOS_DF.iterrows():
        resultado.append(InfoFundo(
            sigla         = str(row.get(col_sigla, "")),
            dy_recente    = _safe_float(row.get(col_dy)),
            p_vp          = _safe_float(row.get("P_VP")),
            vacancia      = _safe_float(row.get("Vacancia")),
            segmento      = _safe_str(row.get("Segmento")),
            tipo_do_fundo = _safe_str(row.get("Tipo_do_Fundo")),
        ))
    return resultado


@app.post("/predict", tags=["Previsão"], response_model=SaidaPredicao)
def prever(entrada: EntradaPredicao):
    """
    Recebe os dados de um FII e retorna o DY previsto para o próximo mês.
    """
    if not MODELOS:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo carregado. Verifique a pasta /modelo."
        )

    nome_modelo = entrada.modelo
    if nome_modelo not in MODELOS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{nome_modelo}' não disponível. Use: {list(MODELOS.keys())}"
        )

    pipe     = MODELOS[nome_modelo]
    num_cols = META.get("num_cols", ["DY_lag1", "DY_lag2", "DY_lag3"])
    cat_cols = META.get("cat_cols", [])

    # Monta DataFrame de entrada espelhando o treino
    row = {
        "DY_lag1": entrada.dy_lag1,
        "DY_lag2": entrada.dy_lag2,
        "DY_lag3": entrada.dy_lag3,
    }
    if "P_VP" in num_cols:
        row["P_VP"] = entrada.p_vp if entrada.p_vp is not None else np.nan
    if "Vacancia" in num_cols:
        row["Vacancia"] = entrada.vacancia if entrada.vacancia is not None else np.nan
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
        dy_previsto  = round(dy_previsto, 4),
        modelo_usado = nome_modelo,
    )


# ── Utilidades ────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 4)
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
