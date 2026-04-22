"""
treinar_modelo.py
-----------------
Retreina Random Forest + Gradient Boosting sem vacância.
Features: DY_lag1, DY_lag2, DY_lag3, P_VP (opcional),
          SELIC (opcional), IFIX (opcional),
          Segmento (one-hot), Tipo_do_Fundo (one-hot)

Uso:
    python treinar_modelo.py --arquivo dataset_fiis_2019_2024.xlsx
"""

import argparse
import json
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

COL_SIGLA    = "Sigla"
COL_DY       = "Dividendos_Yield"
COL_PVP      = "P_VP"
COL_SELIC    = "SELIC"
COL_IFIX     = "IFIX"
COL_SEGMENTO = "Segmento"
COL_TIPO     = "Tipo_do_Fundo"

SAIDA_DIR = Path("modelo")


def carregar_dados(caminho: str) -> pd.DataFrame:
    p = Path(caminho)
    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    print(f"[✓] Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas")
    return df


def construir_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DY_lag1"]   = df.groupby(COL_SIGLA)[COL_DY].shift(1)
    df["DY_lag2"]   = df.groupby(COL_SIGLA)[COL_DY].shift(2)
    df["DY_lag3"]   = df.groupby(COL_SIGLA)[COL_DY].shift(3)
    df["DY_target"] = df.groupby(COL_SIGLA)[COL_DY].shift(-1)
    df = df.dropna(subset=["DY_lag1", "DY_lag2", "DY_lag3", "DY_target"])
    return df.reset_index(drop=True)


def selecionar_features(df: pd.DataFrame):
    num_cols = ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = []

    # Features opcionais numéricas (sem Vacancia)
    for col in [COL_PVP, COL_SELIC, COL_IFIX]:
        if col in df.columns and df[col].notna().sum() > 0:
            num_cols.append(col)

    # Features categóricas
    for col in [COL_SEGMENTO, COL_TIPO]:
        if col in df.columns:
            cat_cols.append(col)

    X = df[num_cols + cat_cols].copy()
    # Preenche NaN em numéricas com mediana
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y = df["DY_target"]
    return X, y, num_cols, cat_cols


def construir_pipeline(num_cols, cat_cols, modelo):
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

    df = carregar_dados(caminho_arquivo)
    df = construir_lags(df)
    X, y, num_cols, cat_cols = selecionar_features(df)

    print(f"[✓] Features numéricas : {num_cols}")
    print(f"[✓] Features categóricas: {cat_cols}")
    print(f"[✓] Total de amostras  : {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelos = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    print("\n📊 Avaliação no conjunto de teste:")
    melhores = {}
    for nome, modelo in modelos.items():
        pipe = construir_pipeline(num_cols, cat_cols, modelo)
        pipe.fit(X_train, y_train)
        mae, r2 = avaliar(nome, pipe, X_test, y_test)
        melhores[nome] = {"pipe": pipe, "mae": mae, "r2": r2}

    for nome, info in melhores.items():
        slug = nome.lower().replace(" ", "_")
        caminho = SAIDA_DIR / f"{slug}.pkl"
        joblib.dump(info["pipe"], caminho)
        print(f"[✓] Salvo: {caminho}")

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

    cols_recentes = [COL_SIGLA, COL_DY, "DY_lag1", "DY_lag2", "DY_lag3"]
    for col in [COL_PVP, COL_SELIC, COL_IFIX, COL_SEGMENTO, COL_TIPO]:
        if col in df.columns:
            cols_recentes.append(col)

    ultimo = df[cols_recentes].sort_values(COL_SIGLA).groupby(COL_SIGLA).last().reset_index()
    ultimo.to_csv(SAIDA_DIR / "fundos_recentes.csv", index=False)
    print(f"[✓] Salvo: {SAIDA_DIR / 'fundos_recentes.csv'} ({len(ultimo)} fundos)")

    print("\n✅ Treinamento concluído!")
    for f in sorted(SAIDA_DIR.iterdir()):
        print(f"   - {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", "-a", default="dados_fii.xlsx")
    args = parser.parse_args()
    treinar(args.arquivo)
