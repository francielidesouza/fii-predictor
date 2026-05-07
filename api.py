"""
api.py v4
---------
API FastAPI para previsão de Dividend Yield de FIIs.
Modelos treinados por segmento:
    - Logistico    (Random Forest, SELIC)
    - Hibrido      (XGBoost, SELIC)
    - Escritorios  (Random Forest, SELIC)
    - Shoppings    (Random Forest, SELIC, normalizado z-score por fundo)

Segmentos excluidos e justificativa:
    - Titulos e Val. Mob.: DY indexado ao spread dos CRIs, nao capturavel
    - Hospital:            apenas 3 fundos com comportamento muito distinto
    - Varejo:              apenas 3 fundos, amostra insuficiente
    - Outros:              grupo heterogeneo sem criterio de homogeneidade

Uso local:
    uvicorn api:app --reload --port 8000
"""

import json, os
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
MODEL_DIR   = Path("modelo")

app = FastAPI(
    title="FII Predictor API v4",
    description="Previsão de DY por segmento — Logístico, Híbrido, Escritórios, Shoppings",
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Segmentos disponíveis e suas configurações ────────────────────────────────
SEGMENTOS_CONFIG = {
    "Logistico":   {"macro": "SELIC", "normalizado": False},
    "Hibrido":     {"macro": "SELIC", "normalizado": False},
    "Escritorios": {"macro": "SELIC", "normalizado": False},
    "Shoppings":   {"macro": "SELIC", "normalizado": True},
}

SEGMENTOS_EXCLUIDOS = {
    "Titulos e Val. Mob.": "DY indexado ao spread dos CRIs — nao capturavel com variaveis macroeconomicas mensais",
    "FOF":                 "DY dependente da carteira de outros FIIs e decisoes do gestor — nao capturavel com variaveis publicas",
    "Hospital":            "apenas 3 fundos com comportamento muito distinto",
    "Varejo":              "apenas 3 fundos — amostra insuficiente",
    "Outros":              "grupo heterogeneo sem criterio de homogeneidade",
}


# ── Carregamento de artefatos ─────────────────────────────────────────────────

def carregar_artefatos():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError("modelo/meta.json nao encontrado. Execute treinar_modelo_v4.py primeiro.")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Carrega modelo por segmento
    modelos_seg = {}
    for seg in SEGMENTOS_CONFIG:
        slug = seg.lower().replace(" ", "_").replace("/", "_").replace(".", "")
        pkl  = MODEL_DIR / f"modelo_{slug}.pkl"
        if pkl.exists():
            modelos_seg[seg] = joblib.load(pkl)
            print(f"[✓] {seg}: {pkl.name} ({pkl.stat().st_size/1024:.0f} KB)")
        else:
            print(f"[!] {seg}: {pkl} nao encontrado")

    # Fallback geral
    fallback = None
    fallback_path = MODEL_DIR / "random_forest.pkl"
    if fallback_path.exists():
        fallback = joblib.load(fallback_path)
        print(f"[✓] Fallback geral carregado")

    # Fundos recentes
    fundos_path = MODEL_DIR / "fundos_recentes.csv"
    fundos_df   = pd.read_csv(fundos_path) if fundos_path.exists() else None

    # Stats de normalizacao por fundo (para Shoppings)
    # Os stats sao derivados do fundos_recentes.csv
    stats_norm = None
    if fundos_df is not None and "Dividendos_Yield" in fundos_df.columns:
        # Reconstroi stats de normalizacao para Shoppings
        df_shop = fundos_df[fundos_df.get("Segmento", pd.Series()) == "Shoppings"] if "Segmento" in fundos_df.columns else pd.DataFrame()
        if not df_shop.empty:
            stats_norm = df_shop.groupby("Sigla")["Dividendos_Yield"].agg(["mean","std"]).rename(
                columns={"mean":"dy_media","std":"dy_std"}
            )
            stats_norm["dy_std"] = stats_norm["dy_std"].replace(0, 1)

    return meta, modelos_seg, fallback, fundos_df, stats_norm


try:
    META, MODELOS_SEG, FALLBACK, FUNDOS_DF, STATS_NORM = carregar_artefatos()
except Exception as e:
    print(f"[!] Erro ao carregar artefatos: {e}")
    META, MODELOS_SEG, FALLBACK, FUNDOS_DF, STATS_NORM = {}, {}, None, None, None


# ── Helpers ───────────────────────────────────────────────────────────────────

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

def _normalizar_dy(dy: float, sigla: str) -> float:
    """Aplica z-score para Shoppings usando stats do treino."""
    if STATS_NORM is not None and sigla in STATS_NORM.index:
        media = STATS_NORM.loc[sigla, "dy_media"]
        std   = STATS_NORM.loc[sigla, "dy_std"]
        return (dy - media) / std
    # fallback: normaliza pela media geral de Shoppings
    return dy

def _desnormalizar_dy(dy_norm: float, sigla: str) -> float:
    """Reverte z-score para Shoppings."""
    if STATS_NORM is not None and sigla in STATS_NORM.index:
        media = STATS_NORM.loc[sigla, "dy_media"]
        std   = STATS_NORM.loc[sigla, "dy_std"]
        return dy_norm * std + media
    return dy_norm

def _get_selic_atual() -> float:
    """Retorna a SELIC mais recente disponivel no fundos_recentes.csv."""
    if FUNDOS_DF is not None and "SELIC" in FUNDOS_DF.columns:
        val = FUNDOS_DF["SELIC"].dropna().iloc[-1] if not FUNDOS_DF["SELIC"].dropna().empty else None
        if val:
            return float(val)
    return 0.0104  # fallback dez/2024

def _get_meta_segmento(seg: str) -> dict:
    """Retorna metricas do segmento no meta.json."""
    mps = META.get("modelos_por_segmento", {}) if META else {}
    return mps.get(seg, {})

def _get_num_cat_cols(seg: str) -> tuple:
    """Retorna num_cols e cat_cols do segmento treinado."""
    info = _get_meta_segmento(seg)
    num_cols = info.get("num_cols", ["DY_lag1","DY_lag2","DY_lag3","PVP_lag1","SELIC"])
    cat_cols = info.get("cat_cols", [])
    return num_cols, cat_cols


# ── Schemas ───────────────────────────────────────────────────────────────────

class EntradaPredicao(BaseModel):
    sigla:          str            = Field(...,  example="HGLG11")
    segmento:       str            = Field(...,  example="Logistico",
                                          description="Segmento do FII: Logistico, Hibrido, Escritorios, Shoppings")
    dy_lag1:        float          = Field(...,  example=0.0082,  description="DY mes anterior (decimal)")
    dy_lag2:        float          = Field(...,  example=0.0079,  description="DY 2 meses atras")
    dy_lag3:        float          = Field(...,  example=0.0081,  description="DY 3 meses atras")
    pvp:            Optional[float]= Field(None, example=0.91,    description="P/VP atual")
    tipo_do_fundo:  Optional[str]  = Field(None, example="Tijolo")
    data_referencia:Optional[str]  = Field(None, example="2025-01-01")
    excluir_pandemia: bool         = Field(False, description="Usar modelo treinado sem pandemia")


class SaidaPredicao(BaseModel):
    sigla:           str
    segmento:        str
    dy_previsto:     float
    dy_previsto_pct: float
    dy_previsto_aa:  float
    modelo_usado:    str
    macro_usado:     str
    normalizado:     bool
    data_referencia: str
    r2:              Optional[float] = None
    mape:            Optional[float] = None
    cv_r2:           Optional[float] = None


class InfoFundo(BaseModel):
    sigla:         str
    dy_recente:    Optional[float] = None
    dy_lag1:       Optional[float] = None
    dy_lag2:       Optional[float] = None
    dy_lag3:       Optional[float] = None
    pvp:           Optional[float] = None
    selic:         Optional[float] = None
    segmento:      Optional[str]   = None
    tipo_do_fundo: Optional[str]   = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def raiz():
    return {
        "status":              "online",
        "versao":              "4.0.0",
        "segmentos_disponiveis": list(MODELOS_SEG.keys()),
        "segmentos_excluidos": SEGMENTOS_EXCLUIDOS,
        "modelos_carregados":  len(MODELOS_SEG),
    }


@app.get("/health", tags=["Status"])
def health():
    mps = META.get("modelos_por_segmento", {}) if META else {}
    resumo = {}
    for seg, info in mps.items():
        melhor = info.get("melhor", "")
        met    = info.get("metricas", {}).get(melhor, {})
        resumo[seg] = {
            "melhor_modelo": melhor,
            "r2":    met.get("r2"),
            "mae":   met.get("mae"),
            "mape":  met.get("mape"),
            "cv_r2": met.get("cv_r2"),
            "n_fundos": info.get("n_fundos"),
            "macro": SEGMENTOS_CONFIG.get(seg, {}).get("macro"),
            "normalizado": SEGMENTOS_CONFIG.get(seg, {}).get("normalizado", False),
        }
    return {
        "status":          "ok",
        "versao_modelo":   META.get("versao") if META else None,
        "n_fundos_total":  META.get("n_fundos_total") if META else None,
        "r2_medio":        META.get("r2_medio_com_pandemia") if META else None,
        "segmentos":       resumo,
        "melhor_modelo":   "por segmento",
        "n_fundos":        META.get("n_fundos_total") if META else None,
    }


@app.get("/fundos", tags=["Dados"], response_model=list[InfoFundo])
def listar_fundos(segmento: Optional[str] = None):
    """Lista fundos disponíveis. Filtrar por segmento opcional."""
    if FUNDOS_DF is None:
        return []
    df = FUNDOS_DF.copy()
    if segmento and "Segmento" in df.columns:
        df = df[df["Segmento"] == segmento]
    resultado = []
    for _, row in df.iterrows():
        resultado.append(InfoFundo(
            sigla         = str(row.get("Sigla", "")),
            dy_recente    = _safe_float(row.get("Dividendos_Yield")),
            dy_lag1       = _safe_float(row.get("DY_lag1")),
            dy_lag2       = _safe_float(row.get("DY_lag2")),
            dy_lag3       = _safe_float(row.get("DY_lag3")),
            pvp           = _safe_float(row.get("PVP_lag1")),
            selic         = _safe_float(row.get("SELIC")),
            segmento      = _safe_str(row.get("Segmento")),
            tipo_do_fundo = _safe_str(row.get("Tipo_do_Fundo")),
        ))
    return resultado


@app.get("/segmentos", tags=["Dados"])
def listar_segmentos():
    """Lista segmentos disponíveis, excluídos e suas configurações."""
    mps = META.get("modelos_por_segmento", {}) if META else {}
    disponiveis = []
    for seg, cfg in SEGMENTOS_CONFIG.items():
        info = mps.get(seg, {})
        melhor = info.get("melhor", "")
        met    = info.get("metricas", {}).get(melhor, {})
        disponiveis.append({
            "segmento":    seg,
            "macro":       cfg["macro"],
            "normalizado": cfg["normalizado"],
            "n_fundos":    info.get("n_fundos"),
            "r2":          met.get("r2"),
            "mape":        met.get("mape"),
        })
    return {
        "disponiveis": disponiveis,
        "excluidos":   [{"segmento": s, "motivo": m} for s, m in SEGMENTOS_EXCLUIDOS.items()],
    }


@app.post("/predict", tags=["Previsão"], response_model=SaidaPredicao)
def prever(entrada: EntradaPredicao):
    """
    Previsão de DY para o próximo mês.
    Seleciona automaticamente o modelo correto para o segmento informado.
    Para Shoppings aplica normalização z-score automaticamente.
    """
    seg = entrada.segmento

    # Verifica se segmento está disponível
    if seg in SEGMENTOS_EXCLUIDOS:
        raise HTTPException(400, detail=(
            f"Segmento '{seg}' excluido do modelo: {SEGMENTOS_EXCLUIDOS[seg]}"
        ))

    if seg not in MODELOS_SEG and FALLBACK is None:
        raise HTTPException(503, detail="Nenhum modelo disponível.")

    # Seleciona modelo
    pipe = MODELOS_SEG.get(seg, FALLBACK)
    cfg  = SEGMENTOS_CONFIG.get(seg, {"macro": "SELIC", "normalizado": False})

    # Sufixo sem pandemia
    if entrada.excluir_pandemia:
        slug     = seg.lower().replace(" ","_").replace("/","_").replace(".","")
        pkl_sp   = MODEL_DIR / f"modelo_{slug}_sem_pandemia.pkl"
        if pkl_sp.exists():
            pipe = joblib.load(pkl_sp)

    # Normaliza DY para Shoppings
    dy1, dy2, dy3 = entrada.dy_lag1, entrada.dy_lag2, entrada.dy_lag3
    normalizado = cfg["normalizado"]
    if normalizado:
        dy1 = _normalizar_dy(dy1, entrada.sigla)
        dy2 = _normalizar_dy(dy2, entrada.sigla)
        dy3 = _normalizar_dy(dy3, entrada.sigla)

    # SELIC atual
    selic = _get_selic_atual()

    # Monta features
    num_cols, cat_cols = _get_num_cat_cols(seg)

    row: dict = {
        "DY_lag1": dy1,
        "DY_lag2": dy2,
        "DY_lag3": dy3,
    }

    if "PVP_lag1" in num_cols:
        row["PVP_lag1"] = entrada.pvp if entrada.pvp is not None else np.nan

    if "SELIC" in num_cols:
        row["SELIC"] = selic

    if "CDI" in num_cols:
        row["CDI"] = selic  # CDI ~ SELIC para previsao futura

    if "Tipo_do_Fundo" in cat_cols:
        row["Tipo_do_Fundo"] = entrada.tipo_do_fundo or "Tijolo"

    X = pd.DataFrame([row])[num_cols + cat_cols]

    try:
        dy_pred = float(pipe.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, detail=f"Erro na predicao: {str(e)}")

    # Desnormaliza para Shoppings
    if normalizado:
        dy_pred = _desnormalizar_dy(dy_pred, entrada.sigla)

    # Metricas do segmento
    info   = _get_meta_segmento(seg)
    melhor = info.get("melhor", "")
    met    = info.get("metricas", {}).get(melhor, {})

    data_ref = entrada.data_referencia or datetime.now().strftime("%Y-%m-01")

    return SaidaPredicao(
        sigla           = entrada.sigla.upper(),
        segmento        = seg,
        dy_previsto     = round(dy_pred, 6),
        dy_previsto_pct = round(dy_pred * 100, 4),
        dy_previsto_aa  = round(dy_pred * 12 * 100, 2),
        modelo_usado    = melhor or "Random Forest",
        macro_usado     = cfg["macro"],
        normalizado     = normalizado,
        data_referencia = data_ref,
        r2              = met.get("r2"),
        mape            = met.get("mape"),
        cv_r2           = met.get("cv_r2"),
    )


@app.post("/predict/comparar", tags=["Previsão"])
def comparar_pandemia(entrada: EntradaPredicao):
    """
    Compara previsao com pandemia vs sem pandemia para o segmento informado.
    Util para visualizacao no TCC.
    """
    seg = entrada.segmento
    if seg in SEGMENTOS_EXCLUIDOS:
        raise HTTPException(400, detail=f"Segmento '{seg}' excluido: {SEGMENTOS_EXCLUIDOS[seg]}")

    cfg          = SEGMENTOS_CONFIG.get(seg, {"macro": "SELIC", "normalizado": False})
    normalizado  = cfg["normalizado"]
    selic        = _get_selic_atual()
    num_cols, cat_cols = _get_num_cat_cols(seg)
    resultados   = []

    for com_pandemia in [True, False]:
        # Seleciona pipe
        slug = seg.lower().replace(" ","_").replace("/","_").replace(".","")
        if com_pandemia:
            pkl = MODEL_DIR / f"modelo_{slug}.pkl"
        else:
            pkl = MODEL_DIR / f"modelo_{slug}_sem_pandemia.pkl"

        if not pkl.exists():
            continue

        pipe = joblib.load(pkl)

        dy1 = _normalizar_dy(entrada.dy_lag1, entrada.sigla) if normalizado else entrada.dy_lag1
        dy2 = _normalizar_dy(entrada.dy_lag2, entrada.sigla) if normalizado else entrada.dy_lag2
        dy3 = _normalizar_dy(entrada.dy_lag3, entrada.sigla) if normalizado else entrada.dy_lag3

        row = {"DY_lag1": dy1, "DY_lag2": dy2, "DY_lag3": dy3}
        if "PVP_lag1" in num_cols:
            row["PVP_lag1"] = entrada.pvp if entrada.pvp is not None else np.nan
        if "SELIC" in num_cols:
            row["SELIC"] = selic
        if "CDI" in num_cols:
            row["CDI"] = selic
        if "Tipo_do_Fundo" in cat_cols:
            row["Tipo_do_Fundo"] = entrada.tipo_do_fundo or "Tijolo"

        X = pd.DataFrame([row])[num_cols + cat_cols]

        try:
            dy_pred = float(pipe.predict(X)[0])
            if normalizado:
                dy_pred = _desnormalizar_dy(dy_pred, entrada.sigla)

            # Metricas
            info   = _get_meta_segmento(seg)
            if not com_pandemia:
                info = info.get("sem_pandemia", info)
            melhor = info.get("melhor", "")
            met    = info.get("metricas", {}).get(melhor, {})

            resultados.append({
                "versao":        "com_pandemia" if com_pandemia else "sem_pandemia",
                "dy_previsto":   round(dy_pred, 6),
                "dy_pct_am":     round(dy_pred * 100, 4),
                "dy_pct_aa":     round(dy_pred * 12 * 100, 2),
                "modelo":        melhor,
                "r2":            met.get("r2"),
                "mape":          met.get("mape"),
                "cv_r2":         met.get("cv_r2"),
            })
        except Exception as e:
            resultados.append({"versao": "com_pandemia" if com_pandemia else "sem_pandemia", "erro": str(e)})

    return {
        "sigla":    entrada.sigla.upper(),
        "segmento": seg,
        "macro":    cfg["macro"],
        "comparacao_pandemia": resultados,
    }


@app.get("/fii/{sigla}", tags=["brapi.dev"])
async def get_fii_brapi(sigla: str):
    """Dados atuais do FII via brapi.dev. Token protegido no servidor."""
    if not BRAPI_TOKEN:
        raise HTTPException(503, detail="BRAPI_TOKEN nao configurado")
    url = f"{BRAPI_BASE}/v2/fii/indicators?symbols={sigla.upper()}"
    async with httpx.AsyncClient(timeout=15, headers={"Authorization": f"Bearer {BRAPI_TOKEN}"}) as client:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(r.status_code, detail=f"brapi.dev: {r.status_code}")
            fiis = r.json().get("fiis", [])
            if not fiis:
                raise HTTPException(404, detail=f"{sigla.upper()} nao encontrado")
            f = fiis[0]
            return {
                "sigla":           sigla.upper(),
                "nome":            f.get("name", ""),
                "segmento":        f.get("segmentoAtuacao", ""),
                "tipo":            f.get("segmentType", ""),
                "dy_12m":          f.get("dividendYield12m"),
                "dy_1m":           f.get("dividendYield1m"),
                "pvp":             f.get("priceToNav"),
                "preco":           f.get("price"),
                "patrimonio":      f.get("equity"),
                "total_cotistas":  f.get("totalInvestors"),
                "fonte":           "brapi.dev Pro",
            }
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Timeout na brapi.dev")


@app.get("/fii/{sigla}/historico2025", tags=["brapi.dev"])
async def get_historico_2025(sigla: str):
    """
    Retorna o DY mensal real de jan/2025 a dez/2025 para o fundo.
    Usado pelo grafico Real 2025 vs Previsto no frontend.
    Token protegido no servidor — nao expoe credencial ao usuario.
    """
    if not BRAPI_TOKEN:
        raise HTTPException(503, detail="BRAPI_TOKEN nao configurado")
    url = f"{BRAPI_BASE}/v2/fii/indicators/history"
    params = {
        "symbols":   sigla.upper(),
        "startDate": "2025-01-01",
        "endDate":   "2025-12-31",
        "sortOrder": "asc",
    }
    async with httpx.AsyncClient(
        timeout=30,
        headers={"Authorization": f"Bearer {BRAPI_TOKEN}"}
    ) as client:
        try:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                raise HTTPException(r.status_code, detail=f"brapi.dev: {r.status_code}")

            history = r.json().get("history", [])
            dados = []
            for item in history:
                mes = item.get("referenceDate", "")[:7]
                dy  = item.get("dividendYield1m")
                pvp = item.get("priceToNav")
                if mes and dy is not None:
                    dados.append({
                        "mes": mes,
                        "dy":  round(float(dy), 6),
                        "pvp": round(float(pvp), 4) if pvp else None,
                    })

            if not dados:
                raise HTTPException(404, detail=f"Sem dados de 2025 para {sigla.upper()}")

            return {
                "sigla":   sigla.upper(),
                "ano":     2025,
                "meses":   len(dados),
                "historico": dados,
                "fonte":   "brapi.dev Pro",
            }
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Timeout na brapi.dev")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
