"""
treinar_modelo.py v3
--------------------
Treina Random Forest + Gradient Boosting + XGBoost com features
de calendário (mês, trimestre, semestre) para capturar sazonalidade.
Prophet incluído opcionalmente para comparação.

Features:
    DY_lag1, DY_lag2, DY_lag3    — autocorrelação temporal
    mes, trimestre, semestre      — sazonalidade (NOVO v3)
    SELIC, IFIX                   — indicadores macroeconômicos (se disponíveis)
    Segmento, Tipo_do_Fundo       — categorias (one-hot)

Nota: P_VP e Vacancia foram excluídos por indisponibilidade de dados
históricos gratuitos e verificáveis para o período 2019-2024.

Uso:
    python treinar_modelo.py --arquivo DatasetUsuario.xlsx
    python treinar_modelo.py --arquivo dataset_fiis_2019_2024.xlsx
    python treinar_modelo.py --arquivo dataset_fiis_2019_2024.xlsx --prophet
"""

import argparse
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

COL_SIGLA    = "Sigla"
COL_DATA     = "Data"
COL_DY       = "Dividendos_Yield"
COL_SELIC    = "SELIC"
COL_IFIX     = "IFIX"
COL_SEGMENTO = "Segmento"
COL_TIPO     = "Tipo_do_Fundo"

SAIDA_DIR = Path("modelo")


def carregar_dados(caminho: str) -> pd.DataFrame:
    p = Path(caminho)
    df = pd.read_excel(p) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p)
    df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df = df.sort_values([COL_SIGLA, COL_DATA]).reset_index(drop=True)
    print(f"[✓] {len(df)} linhas · {df[COL_SIGLA].nunique()} fundos · "
          f"{df[COL_DATA].min().strftime('%Y-%m')} → {df[COL_DATA].max().strftime('%Y-%m')}")
    return df


def adicionar_features_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de calendário para capturar sazonalidade de FIIs:
    - Shoppings distribuem mais em nov/dez (alta temporada de varejo)
    - Fundos de papel variam com ciclo de juros (trimestral/semestral)
    - Ciclos de revisão de contratos costumam ser anuais
    """
    df = df.copy()
    dt = pd.to_datetime(df[COL_DATA], errors="coerce")
    df["mes"]       = dt.dt.month.astype(float)
    df["trimestre"] = dt.dt.quarter.astype(float)
    df["semestre"]  = (dt.dt.month > 6).astype(float)
    print("[✓] Features de calendário: mes, trimestre, semestre")
    return df


def construir_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DY_lag1"]   = df.groupby(COL_SIGLA)[COL_DY].shift(1)
    df["DY_lag2"]   = df.groupby(COL_SIGLA)[COL_DY].shift(2)
    df["DY_lag3"]   = df.groupby(COL_SIGLA)[COL_DY].shift(3)
    df["DY_target"] = df.groupby(COL_SIGLA)[COL_DY].shift(-1)
    antes = len(df)
    df = df.dropna(subset=["DY_lag1", "DY_lag2", "DY_lag3", "DY_target"]).reset_index(drop=True)
    print(f"[✓] Lags criados: {len(df)} amostras ({antes - len(df)} removidas por NaN)")
    return df


def selecionar_features(df: pd.DataFrame):
    num_cols = ["DY_lag1", "DY_lag2", "DY_lag3", "mes", "trimestre", "semestre"]
    cat_cols = []

    for col in [COL_SELIC, COL_IFIX]:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.3:
            num_cols.append(col)
            print(f"    + {col} ({df[col].notna().mean():.0%} preenchido)")

    for col in [COL_SEGMENTO, COL_TIPO]:
        if col in df.columns and df[col].notna().sum() > 0:
            cat_cols.append(col)
            print(f"    + {col} (one-hot)")

    return df[num_cols + cat_cols].copy(), df["DY_target"], num_cols, cat_cols


def construir_pipeline(num_cols, cat_cols, estimador):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ))
    return Pipeline([("pre", ColumnTransformer(transformers)), ("modelo", estimador)])


def avaliar_modelo(nome, pipe, X, y, X_test, y_test):
    preds = pipe.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mask  = y_test != 0
    mape  = mean_absolute_percentage_error(y_test[mask], preds[mask]) * 100 if mask.sum() > 0 else float("nan")

    # TimeSeriesSplit — respeita a ordem cronológica dos dados
    # Cada fold treina no passado e testa no futuro imediato
    cv_r2 = None
    if len(X) >= 50:
        try:
            tscv   = TimeSeriesSplit(n_splits=5)
            scores = []
            for idx_tr, idx_te in tscv.split(X):
                X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
                y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]
                pipe.fit(X_tr, y_tr)
                scores.append(r2_score(y_te, pipe.predict(X_te)))
            cv_r2 = float(np.mean(scores))
            # Retreina no conjunto completo de treino após CV
            pipe.fit(X[y.index.isin(X_test.index) == False], y[y.index.isin(X_test.index) == False])
        except Exception:
            pass

    cv_str = f" | CV R²: {cv_r2:.4f}" if cv_r2 is not None else ""
    print(f"  {nome:30s} → R²: {r2:.4f} | MAE: {mae:.5f} | MAPE: {mape:.1f}%{cv_str}")
    return {"mae": mae, "r2": r2, "mape": mape, "cv_r2": cv_r2}


def avaliar_prophet(df: pd.DataFrame):
    """Avalia Prophet univariado por fundo e retorna métricas médias."""
    try:
        from prophet import Prophet
    except ImportError:
        print("  [!] Prophet não instalado: pip install prophet")
        return None

    maes, r2s = [], []
    for sigla, grupo in df.groupby(COL_SIGLA):
        sub = grupo[[COL_DATA, COL_DY]].rename(columns={COL_DATA: "ds", COL_DY: "y"}).dropna()
        if len(sub) < 8:
            continue
        split  = int(len(sub) * 0.8)
        treino = sub.iloc[:split]
        teste  = sub.iloc[split:]
        try:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, interval_width=0.95)
            m.fit(treino)
            fc    = m.predict(m.make_future_dataframe(periods=len(teste), freq="MS"))
            preds = fc.tail(len(teste))["yhat"].values
            maes.append(mean_absolute_error(teste["y"].values, preds))
            r2s.append(r2_score(teste["y"].values, preds))
        except Exception:
            continue

    if not maes:
        print("  Prophet: dados insuficientes")
        return None

    r2_med  = float(np.mean(r2s))
    mae_med = float(np.mean(maes))
    print(f"  {'Prophet (univariado)':30s} → R²: {r2_med:.4f} | MAE: {mae_med:.5f} | (média por fundo)")
    return {"mae": mae_med, "r2": r2_med, "mape": None, "cv_r2": None}


def treinar(caminho_arquivo: str, incluir_prophet: bool = False):
    SAIDA_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  FII Predictor — Treinamento v3 (com sazonalidade)")
    print(f"{'='*58}\n")

    df = carregar_dados(caminho_arquivo)

    if len(df) < 8:
        print(f"\n⚠  Dataset pequeno ({len(df)} linhas).")
        print(f"   Recomendado: dataset_fiis_2019_2024.xlsx (gerado pelo montar_dataset.py)")
        print(f"   Continuando mesmo assim...\n")

    # Features de calendário antes dos lags (precisam da coluna Data)
    df = adicionar_features_calendario(df)
    df_lags = construir_lags(df)

    if len(df_lags) < 4:
        print(f"\n❌ Amostras insuficientes após lags ({len(df_lags)}).")
        print(f"   Cada fundo precisa de pelo menos 5 meses de histórico.")
        return

    print("\n[→] Selecionando features...")
    X, y, num_cols, cat_cols = selecionar_features(df_lags)
    print(f"\n    Numéricas : {num_cols}")
    print(f"    Categóricas: {cat_cols}")
    print(f"    Amostras   : {len(X)}")

    # Split temporal — últimos 20% dos meses como teste
    # Evita data leakage: treino sempre anterior ao teste
    if len(X) >= 10 and COL_DATA in df_lags.columns:
        data_corte  = pd.to_datetime(df_lags[COL_DATA]).quantile(0.8)
        mask_treino = pd.to_datetime(df_lags[COL_DATA]) < data_corte
        X_train, y_train = X[mask_treino],  y[mask_treino]
        X_test,  y_test  = X[~mask_treino], y[~mask_treino]
        print(f"\n[✓] Split temporal: treino até {data_corte.strftime('%Y-%m')} "
              f"({mask_treino.sum()} amostras) | "
              f"teste após ({(~mask_treino).sum()} amostras)")
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        print("\n[!] Dataset pequeno — treinando e testando no mesmo conjunto")

    # Modelos — hiperparâmetros conservadores para evitar overfitting
    modelos_sk: dict = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42
        ),
    }

    try:
        from xgboost import XGBRegressor
        modelos_sk["XGBoost"] = XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, random_state=42, verbosity=0
        )
        print("\n[✓] XGBoost disponível — incluído")
    except ImportError:
        print("\n[!] XGBoost não encontrado (pip install xgboost) — pulando")

    print(f"\n{'─'*58}")
    print("📊 Resultados no conjunto de teste:")
    print(f"{'─'*58}")

    resultados: dict = {}
    for nome, estimador in modelos_sk.items():
        pipe = construir_pipeline(num_cols, cat_cols, estimador)
        pipe.fit(X_train, y_train)
        metricas = avaliar_modelo(nome, pipe, X_train, y_train, X_test, y_test)
        resultados[nome] = {"pipe": pipe, **metricas}

    if incluir_prophet:
        print("\n[→] Prophet (comparação como baseline estatístico)...")
        pm = avaliar_prophet(df)
        if pm:
            resultados["Prophet"] = pm

    # Melhor modelo
    melhor = max(
        (n for n in resultados if "pipe" in resultados[n]),
        key=lambda n: resultados[n]["r2"]
    )
    print(f"\n{'─'*58}")
    print(f"🏆 Melhor: {melhor} (R²={resultados[melhor]['r2']:.4f})")

    # Salvar modelos
    print(f"\n[→] Salvando...")
    for nome, info in resultados.items():
        if "pipe" not in info:
            continue
        slug = nome.lower().replace(" ", "_")
        path = SAIDA_DIR / f"{slug}.pkl"
        joblib.dump(info["pipe"], path)
        print(f"  [✓] {path} ({path.stat().st_size/1024:.0f} KB)")

    # meta.json
    meta = {
        "num_cols":            num_cols,
        "cat_cols":            cat_cols,
        "col_sigla":           COL_SIGLA,
        "col_dy":              COL_DY,
        "modelos":             [n for n in resultados if "pipe" in resultados[n]],
        "melhor_modelo":       melhor,
        "features_calendario": ["mes", "trimestre", "semestre"],
        "metricas": {
            n: {
                "r2":    round(info["r2"], 4),
                "mae":   round(info["mae"], 6),
                "mape":  round(info["mape"], 2) if info["mape"] and not np.isnan(info["mape"]) else None,
                "cv_r2": round(info["cv_r2"], 4) if info["cv_r2"] is not None else None,
            }
            for n, info in resultados.items()
        },
        "n_amostras_treino": len(X_train),
        "n_fundos":          int(df[COL_SIGLA].nunique()),
        "dataset":           Path(caminho_arquivo).name,
        "versao":            "3.0",
    }
    with open(SAIDA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [✓] modelo/meta.json")

    # Fundos recentes
    cols_rec = [COL_SIGLA, COL_DY, "DY_lag1", "DY_lag2", "DY_lag3"]
    for col in [COL_SELIC, COL_IFIX, COL_SEGMENTO, COL_TIPO]:
        if col in df_lags.columns:
            cols_rec.append(col)
    ultimo = df_lags[cols_rec].groupby(COL_SIGLA).last().reset_index()
    ultimo.to_csv(SAIDA_DIR / "fundos_recentes.csv", index=False)
    print(f"  [✓] modelo/fundos_recentes.csv ({len(ultimo)} fundos)")

    print(f"\n{'='*58}")
    print(f"✅ Pronto! Execute: uvicorn api:app --reload --port 8000")
    print(f"   Docs:            http://localhost:8000/docs\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", "-a", default="DatasetUsuario.xlsx")
    parser.add_argument("--prophet", action="store_true",
                        help="Incluir Prophet na comparação (pip install prophet)")
    args = parser.parse_args()
    treinar(args.arquivo, args.prophet)
