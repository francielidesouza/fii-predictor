"""
treinar_modelo.py
-----------------
Retreina os modelos do Orange (Random Forest + Gradient Boosting)
usando scikit-learn e salva os artefatos prontos para deploy.

Pré-requisitos:
    pip install pandas scikit-learn openpyxl joblib

Uso:
    python treinar_modelo.py --arquivo seus_dados.xlsx
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Configuração ──────────────────────────────────────────────────────────────

# Colunas que correspondem às do seu script Orange
COL_SIGLA    = "Sigla"          # coluna com identificador do fundo (ex: HGLG11)
COL_DY       = "Dividendos_Yield"
COL_PVP      = "P_VP"          # opcional
COL_VACANCIA = "Vacancia"      # opcional
COL_SEGMENTO = "Segmento"      # opcional — será one-hot encoded
COL_TIPO     = "Tipo_do_Fundo" # opcional — será one-hot encoded

SAIDA_DIR    = Path("modelo")  # pasta onde os artefatos serão salvos


# ── Helpers ───────────────────────────────────────────────────────────────────

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Lê XLSX ou CSV automaticamente."""
    p = Path(caminho)
    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    print(f"[✓] Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas")
    return df


def construir_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica a lógica do script Python do Orange:
    - DY_lag1, DY_lag2, DY_lag3 por fundo
    - DY_target = DY do próximo mês
    """
    df = df.copy()
    df["DY_lag1"]  = df.groupby(COL_SIGLA)[COL_DY].shift(1)
    df["DY_lag2"]  = df.groupby(COL_SIGLA)[COL_DY].shift(2)
    df["DY_lag3"]  = df.groupby(COL_SIGLA)[COL_DY].shift(3)
    df["DY_target"] = df.groupby(COL_SIGLA)[COL_DY].shift(-1)
    df = df.dropna(subset=["DY_lag1", "DY_lag2", "DY_lag3", "DY_target"])
    return df.reset_index(drop=True)


def selecionar_features(df: pd.DataFrame):
    """
    Monta listas de features numéricas e categóricas disponíveis no dataset.
    Retorna (X, y, num_cols, cat_cols).
    """
    num_cols = ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = []

    for col in [COL_PVP, COL_VACANCIA]:
        if col in df.columns:
            num_cols.append(col)

    for col in [COL_SEGMENTO, COL_TIPO]:
        if col in df.columns:
            cat_cols.append(col)

    X = df[num_cols + cat_cols]
    y = df["DY_target"]
    return X, y, num_cols, cat_cols


# ── Treinamento ───────────────────────────────────────────────────────────────

def construir_pipeline(num_cols, cat_cols, modelo):
    """Monta Pipeline com pré-processamento + modelo."""
    transformers = [("num", "passthrough", num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ))
    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("pre", preprocessor), ("modelo", modelo)])


def avaliar(nome, pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"  {nome:25s} → MAE: {mae:.4f}  |  R²: {r2:.4f}")
    return mae, r2


def treinar(caminho_arquivo: str):
    SAIDA_DIR.mkdir(exist_ok=True)

    # 1. Carregar e preparar dados
    df = carregar_dados(caminho_arquivo)
    df = construir_lags(df)
    X, y, num_cols, cat_cols = selecionar_features(df)

    print(f"[✓] Features numéricas : {num_cols}")
    print(f"[✓] Features categóricas: {cat_cols}")
    print(f"[✓] Total de amostras  : {len(X)}")

    # 2. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Modelos (mesmos que o Orange)
    modelos = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    print("\n📊 Avaliação no conjunto de teste:")
    melhores = {}
    for nome, modelo in modelos.items():
        pipe = construir_pipeline(num_cols, cat_cols, modelo)
        pipe.fit(X_train, y_train)
        mae, r2 = avaliar(nome, pipe, X_test, y_test)
        melhores[nome] = {"pipe": pipe, "mae": mae, "r2": r2}

    # 4. Salvar artefatos
    for nome, info in melhores.items():
        slug = nome.lower().replace(" ", "_")
        caminho = SAIDA_DIR / f"{slug}.pkl"
        joblib.dump(info["pipe"], caminho)
        print(f"[✓] Salvo: {caminho}")

    # 5. Salvar metadados (features usadas pela API)
    import json
    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "col_sigla": COL_SIGLA,
        "col_dy": COL_DY,
        "modelos": list(modelos.keys()),
    }
    with open(SAIDA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[✓] Salvo: {SAIDA_DIR / 'meta.json'}")

    # 6. Salvar lista de FIIs únicos para o formulário do frontend
    fundos = df[[COL_SIGLA, COL_DY, "DY_lag1", "DY_lag2", "DY_lag3"]
                + ([COL_PVP] if COL_PVP in df.columns else [])
                + ([COL_VACANCIA] if COL_VACANCIA in df.columns else [])
                + ([COL_SEGMENTO] if COL_SEGMENTO in df.columns else [])
                + ([COL_TIPO] if COL_TIPO in df.columns else [])
               ].copy()
    # Último registro por fundo (dados mais recentes)
    ultimo = fundos.sort_values(COL_SIGLA).groupby(COL_SIGLA).last().reset_index()
    ultimo.to_csv(SAIDA_DIR / "fundos_recentes.csv", index=False)
    print(f"[✓] Salvo: {SAIDA_DIR / 'fundos_recentes.csv'} ({len(ultimo)} fundos)")

    print("\n✅ Treinamento concluído! Pasta gerada: ./modelo/")
    print("   Conteúdo:")
    for f in sorted(SAIDA_DIR.iterdir()):
        print(f"   - {f.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retreina modelos FII com scikit-learn")
    parser.add_argument(
        "--arquivo", "-a",
        default="dados_fii.xlsx",
        help="Caminho para o arquivo .xlsx ou .csv com os dados históricos"
    )
    args = parser.parse_args()
    treinar(args.arquivo)
