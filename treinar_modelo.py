"""
treinar_modelo.py
-----------------
Treina Random Forest + Gradient Boosting com os dados históricos de FIIs.

Colunas esperadas no arquivo de entrada:
    Data, Sigla, Segmento, Tipo_do_Fundo,
    Dividendos_Yield, P_VP, Vacancia, SELIC, IFIX

Uso:
    python treinar_modelo.py --arquivo DatasetUsuario.xlsx

    # Para usar o dataset completo gerado pelo montar_dataset.py:
    python treinar_modelo.py --arquivo dataset_fiis_2019_2024.xlsx
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Configuração ──────────────────────────────────────────────────────────────

COL_SIGLA    = "Sigla"
COL_DATA     = "Data"
COL_DY       = "Dividendos_Yield"
COL_PVP      = "P_VP"
COL_VACANCIA = "Vacancia"
COL_SELIC    = "SELIC"
COL_IFIX     = "IFIX"
COL_SEGMENTO = "Segmento"
COL_TIPO     = "Tipo_do_Fundo"

SAIDA_DIR = Path("modelo")
MINIMO_MESES_POR_FUNDO = 4  # precisa de lag1+lag2+lag3+target


# ── Helpers ───────────────────────────────────────────────────────────────────

def carregar_dados(caminho: str) -> pd.DataFrame:
    p = Path(caminho)
    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    # Garante que Data é datetime e ordena
    df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df = df.sort_values([COL_SIGLA, COL_DATA]).reset_index(drop=True)

    print(f"[✓] Arquivo carregado: {len(df)} linhas · {df[COL_SIGLA].nunique()} fundos")
    print(f"    Período: {df[COL_DATA].min().strftime('%Y-%m')} → {df[COL_DATA].max().strftime('%Y-%m')}")
    print(f"    Colunas: {list(df.columns)}")
    return df


def construir_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de lag por fundo (groupby Sigla).
    Replica exatamente a lógica do Python Script do Orange.
    """
    df = df.copy()
    df["DY_lag1"]   = df.groupby(COL_SIGLA)[COL_DY].shift(1)
    df["DY_lag2"]   = df.groupby(COL_SIGLA)[COL_DY].shift(2)
    df["DY_lag3"]   = df.groupby(COL_SIGLA)[COL_DY].shift(3)
    df["DY_target"] = df.groupby(COL_SIGLA)[COL_DY].shift(-1)

    antes = len(df)
    df = df.dropna(subset=["DY_lag1", "DY_lag2", "DY_lag3", "DY_target"])
    df = df.reset_index(drop=True)
    removidas = antes - len(df)
    print(f"[✓] Lags criados: {len(df)} amostras ({removidas} linhas removidas por NaN)")
    return df


def selecionar_features(df: pd.DataFrame):
    """
    Detecta automaticamente quais colunas estão disponíveis.
    Retorna (X, y, num_cols, cat_cols).
    """
    num_cols = ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = []

    # Colunas numéricas opcionais
    for col in [COL_PVP, COL_VACANCIA, COL_SELIC, COL_IFIX]:
        if col in df.columns and df[col].notna().sum() > 0:
            num_cols.append(col)
            print(f"    + Feature numérica incluída: {col}")

    # Colunas categóricas opcionais
    for col in [COL_SEGMENTO, COL_TIPO]:
        if col in df.columns and df[col].notna().sum() > 0:
            cat_cols.append(col)
            print(f"    + Feature categórica incluída: {col}")

    X = df[num_cols + cat_cols].copy()
    y = df["DY_target"]
    return X, y, num_cols, cat_cols


# ── Pipeline ──────────────────────────────────────────────────────────────────

def construir_pipeline(num_cols, cat_cols, estimador):
    """
    Monta Pipeline scikit-learn com:
    - Imputação de NaN por média (numéricas)
    - One-hot encoding (categóricas)
    - Modelo de regressão
    """
    transformers = [(
        "num",
        SimpleImputer(strategy="mean"),
        num_cols
    )]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ))
    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("pre", preprocessor), ("modelo", estimador)])


# ── Avaliação ─────────────────────────────────────────────────────────────────

def avaliar_modelo(nome, pipe, X, y, X_test, y_test, n_amostras):
    """Avalia com cross-validation se tiver dados suficientes, senão usa hold-out."""
    preds = pipe.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    # MAPE com proteção contra divisão por zero
    mask = y_test != 0
    mape = mean_absolute_percentage_error(y_test[mask], preds[mask]) * 100 if mask.sum() > 0 else float("nan")

    cv_r2 = None
    if n_amostras >= 30:
        cv_folds = min(5, n_amostras // 5)
        try:
            scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring="r2")
            cv_r2 = scores.mean()
            print(f"  {nome:25s} → MAE: {mae:.4f} | R²: {r2:.4f} | MAPE: {mape:.1f}% | CV R²: {cv_r2:.4f} ({cv_folds} folds)")
        except Exception:
            print(f"  {nome:25s} → MAE: {mae:.4f} | R²: {r2:.4f} | MAPE: {mape:.1f}%")
    else:
        print(f"  {nome:25s} → MAE: {mae:.4f} | R²: {r2:.4f} | MAPE: {mape:.1f}% (hold-out, N={n_amostras})")

    return {"mae": mae, "r2": r2, "mape": mape, "cv_r2": cv_r2}


# ── Treinamento principal ─────────────────────────────────────────────────────

def treinar(caminho_arquivo: str):
    SAIDA_DIR.mkdir(exist_ok=True)

    # 1. Carregar dados
    df = carregar_dados(caminho_arquivo)

    # Aviso se dataset pequeno
    n_fundos = df[COL_SIGLA].nunique()
    n_meses  = df.groupby(COL_SIGLA).size().max()
    if len(df) < 50:
        print(f"\n⚠ Dataset pequeno ({len(df)} linhas). Métricas podem ser imprecisas.")
        print(f"  Recomendado: usar dataset_fiis_2019_2024.xlsx gerado pelo montar_dataset.py")
        print(f"  O modelo será treinado mesmo assim para uso em previsão.\n")

    # 2. Criar lags
    df = construir_lags(df)

    if len(df) < MINIMO_MESES_POR_FUNDO:
        print(f"\n❌ Dados insuficientes após criar lags ({len(df)} amostras).")
        print(f"   Cada fundo precisa de pelo menos {MINIMO_MESES_POR_FUNDO} meses de histórico.")
        return

    # 3. Features
    print("\n[→] Selecionando features...")
    X, y, num_cols, cat_cols = selecionar_features(df)

    print(f"\n[✓] Features numéricas : {num_cols}")
    print(f"[✓] Features categóricas: {cat_cols}")
    print(f"[✓] Total de amostras  : {len(X)}")

    # 4. Split treino/teste
    test_size = 0.2 if len(X) >= 10 else 1  # se muito pequeno, testa no próprio treino
    if len(X) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # 5. Modelos
    modelos = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=None if len(X) > 50 else 4,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8 if len(X) > 50 else 1.0,
            random_state=42
        ),
    }

    print("\n📊 Avaliação no conjunto de teste:")
    resultados = {}
    for nome, estimador in modelos.items():
        pipe = construir_pipeline(num_cols, cat_cols, estimador)
        pipe.fit(X_train, y_train)
        metricas = avaliar_modelo(nome, pipe, X_train, y_train, X_test, y_test, len(X))
        resultados[nome] = {"pipe": pipe, **metricas}

    # Melhor modelo
    melhor = max(resultados, key=lambda n: resultados[n]["r2"])
    print(f"\n🏆 Melhor modelo: {melhor} (R²={resultados[melhor]['r2']:.4f})")

    # 6. Salvar modelos
    print("\n[→] Salvando artefatos...")
    for nome, info in resultados.items():
        slug = nome.lower().replace(" ", "_")
        caminho = SAIDA_DIR / f"{slug}.pkl"
        joblib.dump(info["pipe"], caminho)
        print(f"  [✓] {caminho}")

    # 7. Meta.json — contém tudo que a api.py precisa saber
    meta = {
        "num_cols":      num_cols,
        "cat_cols":      cat_cols,
        "col_sigla":     COL_SIGLA,
        "col_dy":        COL_DY,
        "modelos":       list(modelos.keys()),
        "melhor_modelo": melhor,
        "metricas": {
            nome: {
                "mae":    round(info["mae"], 6),
                "r2":     round(info["r2"],  4),
                "mape":   round(info["mape"], 2) if not np.isnan(info["mape"]) else None,
                "cv_r2":  round(info["cv_r2"], 4) if info["cv_r2"] is not None else None,
            }
            for nome, info in resultados.items()
        },
        "n_amostras_treino": len(X_train),
        "n_fundos":          n_fundos,
        "dataset":           Path(caminho_arquivo).name,
    }
    meta_path = SAIDA_DIR / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [✓] {meta_path}")

    # 8. Fundos recentes — último mês de cada fundo para o frontend
    cols_fundo = [COL_SIGLA, COL_DY, "DY_lag1", "DY_lag2", "DY_lag3"]
    for col in [COL_PVP, COL_VACANCIA, COL_SELIC, COL_IFIX, COL_SEGMENTO, COL_TIPO]:
        if col in df.columns:
            cols_fundo.append(col)

    ultimo = (
        df[cols_fundo]
        .sort_values(COL_SIGLA)
        .groupby(COL_SIGLA)
        .last()
        .reset_index()
    )
    fundos_path = SAIDA_DIR / "fundos_recentes.csv"
    ultimo.to_csv(fundos_path, index=False)
    print(f"  [✓] {fundos_path} ({len(ultimo)} fundos)")

    print(f"\n✅ Treinamento concluído! Pasta: ./{SAIDA_DIR}/")
    print("   Conteúdo:")
    for f in sorted(SAIDA_DIR.iterdir()):
        size = f.stat().st_size
        print(f"   - {f.name:<35} {size/1024:.1f} KB")

    print(f"\n💡 Próximo passo: uvicorn api:app --reload")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelos FII com scikit-learn")
    parser.add_argument(
        "--arquivo", "-a",
        default="DatasetUsuario.xlsx",
        help="Caminho para o .xlsx com os dados históricos (padrão: DatasetUsuario.xlsx)"
    )
    args = parser.parse_args()
    treinar(args.arquivo)
