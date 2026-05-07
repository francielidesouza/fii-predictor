"""
treinar_modelo_v4.py
--------------------
Treina um modelo por SEGMENTO dentro de Tijolo e Papel.

Segmentos excluidos:
    Outros   — grupo heterogeneo sem criterio de homogeneidade
    Hospital — apenas 3 fundos com comportamento muito distinto
    Varejo   — apenas 3 fundos, amostra insuficiente

Features:
    DY_lag1, DY_lag2, DY_lag3  — autocorrelacao temporal
    PVP_lag1                    — P/VP do mes anterior
    CDI                         — apenas para Titulos e Val. Mob. (BCB SGS 4391)
    SELIC                       — demais segmentos (BCB SGS 4390)
    Tipo_do_Fundo               — one-hot quando > 1 tipo no segmento

Treina duas versoes por segmento: com pandemia e sem pandemia (2020-03 a 2020-12)

Uso:
    python treinar_modelo_v4.py --arquivo dataset_fiis_2019_2024_brapi_v2.xlsx
"""

import argparse, json, warnings, joblib, requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

COL_SIGLA    = "Sigla"
COL_DATA     = "Data"
COL_DY       = "Dividendos_Yield"
COL_PVP      = "P_VP"
COL_SELIC    = "SELIC"
COL_CDI      = "CDI"
COL_SEGMENTO = "Segmento"
COL_TIPO     = "Tipo_do_Fundo"

SAIDA_DIR = Path("modelo")

SEGS_EXCLUIR = {
    "Outros":   "grupo heterogeneo sem criterio de homogeneidade",
    "Hospital": "apenas 3 fundos com comportamento muito distinto",
    "Varejo":   "apenas 3 fundos — amostra insuficiente para generalizacao",
    "Titulos e Val. Mob.": "DY indexado ao spread dos CRIs — nao capturavel com variaveis macroeconomicas mensais"
}

PANDEMIA_INI = "2020-03"
PANDEMIA_FIM = "2020-12"


def carregar_dados(caminho: str) -> pd.DataFrame:
    p = Path(caminho)
    df = pd.read_excel(p) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p)
    df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df = df.sort_values([COL_SIGLA, COL_DATA]).reset_index(drop=True)
    print(f"[✓] {len(df)} linhas · {df[COL_SIGLA].nunique()} fundos · "
          f"{df[COL_DATA].min().strftime('%Y-%m')} → {df[COL_DATA].max().strftime('%Y-%m')}")
    return df


def buscar_indicadores() -> tuple:
    """Busca SELIC (4390) e CDI (4391) do BCB SGS. Retorna (selic_dict, cdi_dict)."""
    selic, cdi = {}, {}
    for codigo, nome, destino in [(4390, "SELIC", selic), (4391, "CDI", cdi)]:
        print(f"[→] Buscando {nome} do BCB SGS {codigo}...", end=" ")
        try:
            r = requests.get(
                f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados",
                params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"},
                timeout=20
            )
            if r.status_code == 200:
                for item in r.json():
                    mes = pd.to_datetime(item["data"], format="%d/%m/%Y").strftime("%Y-%m")
                    destino[mes] = round(float(item["valor"]) / 100, 6)
                print(f"✓ {len(destino)} meses")
            else:
                print(f"⚠ HTTP {r.status_code}")
        except Exception as e:
            print(f"⚠ falhou ({e})")
    return selic, cdi


def adicionar_indicador(df: pd.DataFrame, indicador: dict, col_nome: str) -> pd.DataFrame:
    mes_col = pd.to_datetime(df[COL_DATA]).dt.strftime("%Y-%m")
    df[col_nome] = mes_col.map(indicador)
    return df


def normalizar_dy_por_fundo(df: pd.DataFrame) -> tuple:
    """
    Normaliza DY por fundo (z-score) para remover efeito de escala.
    Retorna df normalizado e stats para desnormalizar depois.
    """
    stats = df.groupby(COL_SIGLA)[COL_DY].agg(["mean","std"]).rename(
        columns={"mean":"dy_media","std":"dy_std"}
    )
    stats["dy_std"] = stats["dy_std"].replace(0, 1)
    df = df.copy()
    df[COL_DY] = df.apply(
        lambda r: (r[COL_DY] - stats.loc[r[COL_SIGLA],"dy_media"]) / stats.loc[r[COL_SIGLA],"dy_std"],
        axis=1
    )
    return df, stats


def construir_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DY_lag1"]   = df.groupby(COL_SIGLA)[COL_DY].shift(1)
    df["DY_lag2"]   = df.groupby(COL_SIGLA)[COL_DY].shift(2)
    df["DY_lag3"]   = df.groupby(COL_SIGLA)[COL_DY].shift(3)
    df["DY_target"] = df.groupby(COL_SIGLA)[COL_DY].shift(-1)
    if COL_PVP in df.columns:
        df["PVP_lag1"] = df.groupby(COL_SIGLA)[COL_PVP].shift(1)
    antes = len(df)
    df = df.dropna(subset=["DY_lag1", "DY_lag2", "DY_lag3", "DY_target"]).reset_index(drop=True)
    print(f"  [✓] Lags: {len(df)} amostras ({antes - len(df)} removidas por NaN)")
    return df


def construir_pipeline(num_cols, cat_cols, estimador):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ))
    return Pipeline([("pre", ColumnTransformer(transformers)), ("modelo", estimador)])


def avaliar_modelo(nome, pipe, X_train, y_train, X_test, y_test):
    preds = pipe.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mask  = y_test != 0
    mape  = mean_absolute_percentage_error(y_test[mask], preds[mask]) * 100 if mask.sum() > 0 else float("nan")
    cv_r2 = None
    if len(X_train) >= 50:
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for idx_tr, idx_te in tscv.split(X_train):
                pipe.fit(X_train.iloc[idx_tr], y_train.iloc[idx_tr])
                scores.append(r2_score(y_train.iloc[idx_te], pipe.predict(X_train.iloc[idx_te])))
            cv_r2 = float(np.mean(scores))
            pipe.fit(X_train, y_train)
        except Exception:
            pass
    cv_str = f" | CV R2: {cv_r2:.4f}" if cv_r2 is not None else ""
    print(f"    {nome:30s} → R2: {r2:.4f} | MAE: {mae:.6f} | MAPE: {mape:.1f}%{cv_str}")
    return {"mae": mae, "r2": r2, "mape": mape, "cv_r2": cv_r2}


def treinar_segmento(seg, df_seg, selic, cdi, excluir_pandemia=False):
    """Treina modelos para um segmento. Retorna (melhor_pipe, melhor_nome, meta_dict)."""

    label = f"{seg}{' (sem pandemia)' if excluir_pandemia else ''}"
    print(f"\n{'─'*58}")
    print(f"  Segmento: {label}")
    print(f"{'─'*58}")

    df_work = df_seg.copy()

    # Shoppings: normaliza DY por fundo para remover dispersao de escala
    stats_norm = None
    if seg == "Shoppings":
        df_work, stats_norm = normalizar_dy_por_fundo(df_work)
        print(f"  + DY normalizado por fundo (z-score) — {df_work[COL_SIGLA].nunique()} fundos")

    # Remove pandemia se solicitado
    if excluir_pandemia:
        mes = pd.to_datetime(df_work[COL_DATA]).dt.strftime("%Y-%m")
        df_work = df_work[~((mes >= PANDEMIA_INI) & (mes <= PANDEMIA_FIM))]
        print(f"  ⚠ Pandemia excluida ({PANDEMIA_INI} a {PANDEMIA_FIM})")

    # Titulos e Val. Mob. usa CDI — indexador real dos CRIs
    # Demais segmentos usam SELIC como variavel macroeconomica
    col_macro = None
    if seg == "Titulos e Val. Mob." and cdi:
        df_work = adicionar_indicador(df_work, cdi, COL_CDI)
        col_macro = COL_CDI
        print(f"  + CDI mensal (BCB SGS 4391) — indexador dos CRIs")
    elif selic:
        df_work = adicionar_indicador(df_work, selic, COL_SELIC)
        col_macro = COL_SELIC
        print(f"  + SELIC mensal (BCB SGS 4390)")

    df_lags = construir_lags(df_work)
    if len(df_lags) < 20:
        print(f"  Amostras insuficientes ({len(df_lags)}) — pulando")
        return None, None, None

    num_cols = ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = []

    if "PVP_lag1" in df_lags.columns and df_lags["PVP_lag1"].notna().sum() > len(df_lags) * 0.3:
        num_cols.append("PVP_lag1")
        print(f"  + PVP_lag1 ({df_lags['PVP_lag1'].notna().mean():.0%} preenchido)")

    if col_macro and col_macro in df_lags.columns and df_lags[col_macro].notna().mean() > 0.3:
        num_cols.append(col_macro)
        print(f"  + {col_macro} ({df_lags[col_macro].notna().mean():.0%} preenchido)")

    if COL_TIPO in df_lags.columns and df_lags[COL_TIPO].nunique() > 1:
        cat_cols.append(COL_TIPO)
        print(f"  + Tipo_do_Fundo ({df_lags[COL_TIPO].nunique()} tipos)")

    X = df_lags[num_cols + cat_cols].copy()
    y = df_lags["DY_target"]

    if len(X) >= 10 and COL_DATA in df_lags.columns:
        data_corte  = pd.to_datetime(df_lags[COL_DATA]).quantile(0.8)
        mask_treino = pd.to_datetime(df_lags[COL_DATA]) < data_corte
        X_train, y_train = X[mask_treino],  y[mask_treino]
        X_test,  y_test  = X[~mask_treino], y[~mask_treino]
        print(f"  Split: treino ate {data_corte.strftime('%Y-%m')} "
              f"({mask_treino.sum()}) | teste ({(~mask_treino).sum()})")
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    modelos_sk = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            max_features=0.7, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=3, random_state=42
        ),
    }
    try:
        from xgboost import XGBRegressor
        modelos_sk["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=3, random_state=42, verbosity=0
        )
    except ImportError:
        pass

    melhor_r2    = -999
    melhor_nome  = None
    melhor_pipe  = None
    metricas_seg = {}

    for nome, estimador in modelos_sk.items():
        pipe = construir_pipeline(num_cols, cat_cols, estimador)
        pipe.fit(X_train, y_train)
        met = avaliar_modelo(nome, pipe, X_train, y_train, X_test, y_test)
        metricas_seg[nome] = met
        if met["r2"] > melhor_r2:
            melhor_r2   = met["r2"]
            melhor_nome = nome
            melhor_pipe = pipe

    print(f"  🏆 Melhor: {melhor_nome} (R2={melhor_r2:.4f})")
    return melhor_pipe, melhor_nome, {
        "melhor":           melhor_nome,
        "num_cols":         num_cols,
        "cat_cols":         cat_cols,
        "indicador_macro":  col_macro,
        "excluiu_pandemia": excluir_pandemia,
        "stats_norm": stats_norm is not None,
        "metricas": {
            n: {
                "r2":    round(m["r2"], 4),
                "mae":   round(m["mae"], 6),
                "mape":  round(m["mape"], 2) if m["mape"] and not np.isnan(m["mape"]) else None,
                "cv_r2": round(m["cv_r2"], 4) if m["cv_r2"] is not None else None,
            }
            for n, m in metricas_seg.items()
        }
    }


def treinar(caminho_arquivo: str):
    SAIDA_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  FII Predictor — Treinamento v4 (por segmento)")
    print(f"  CDI para Titulos | SELIC para demais | com/sem pandemia")
    print(f"{'='*58}\n")

    df = carregar_dados(caminho_arquivo)

    # Busca SELIC e CDI do BCB
    selic, cdi = buscar_indicadores()

    # Filtra apenas Tijolo e Papel
    df = df[df[COL_TIPO].isin({"Tijolo", "Papel"})]

    # Remove segmentos excluidos
    for seg, motivo in SEGS_EXCLUIR.items():
        df = df[df[COL_SEGMENTO] != seg]
        print(f"[✓] '{seg}' excluido: {motivo}")

    # Remove segmentos com menos de 3 fundos
    contagem = df.groupby(COL_SEGMENTO)[COL_SIGLA].nunique()
    segs_validos = contagem[contagem >= 3].index
    df = df[df[COL_SEGMENTO].isin(segs_validos)]

    print(f"\n[✓] {df[COL_SIGLA].nunique()} fundos | {df[COL_SEGMENTO].nunique()} segmentos")
    for seg in sorted(segs_validos):
        n    = df[df[COL_SEGMENTO] == seg][COL_SIGLA].nunique()
        tipo = df[df[COL_SEGMENTO] == seg][COL_TIPO].unique().tolist()
        macro = "CDI" if seg == "Titulos e Val. Mob." else "SELIC"
        print(f"    {seg:25s}: {n:3d} fundos | {tipo} | macro: {macro}")

    segmentos      = sorted(df[COL_SEGMENTO].dropna().unique())
    modelos_meta   = {}
    resultados_r2  = {}
    resultados_r2_sp = {}

    for seg in segmentos:
        df_seg = df[df[COL_SEGMENTO] == seg].copy()

        # Com pandemia
        pipe, nome, meta = treinar_segmento(seg, df_seg, selic, cdi, excluir_pandemia=False)
        if pipe is not None:
            slug = seg.lower().replace(" ", "_").replace("/", "_").replace(".", "")
            path = SAIDA_DIR / f"modelo_{slug}.pkl"
            joblib.dump(pipe, path)
            print(f"  [✓] {path} ({path.stat().st_size/1024:.0f} KB)")
            meta["arquivo"]  = f"modelo_{slug}.pkl"
            meta["n_fundos"] = int(df_seg[COL_SIGLA].nunique())
            resultados_r2[seg] = meta["metricas"][nome]["r2"]

        # Sem pandemia
        pipe_sp, nome_sp, meta_sp = treinar_segmento(seg, df_seg, selic, cdi, excluir_pandemia=True)
        if pipe_sp is not None:
            slug = seg.lower().replace(" ", "_").replace("/", "_").replace(".", "")
            path_sp = SAIDA_DIR / f"modelo_{slug}_sem_pandemia.pkl"
            joblib.dump(pipe_sp, path_sp)
            print(f"  [✓] {path_sp} ({path_sp.stat().st_size/1024:.0f} KB)")
            resultados_r2_sp[seg] = meta_sp["metricas"][nome_sp]["r2"]
            if meta:
                meta["sem_pandemia"] = meta_sp

        if meta:
            modelos_meta[seg] = meta

    # Modelo geral fallback
    print(f"\n{'─'*58}")
    print("  Modelo geral (fallback)...")
    df_lags_all = construir_lags(df)
    num_all = ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_all = []
    if "PVP_lag1" in df_lags_all.columns:
        num_all.append("PVP_lag1")
    for col in [COL_SEGMENTO, COL_TIPO]:
        if col in df_lags_all.columns:
            cat_all.append(col)
    pipe_geral = construir_pipeline(
        num_all, cat_all,
        RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5,
                              max_features=0.7, random_state=42, n_jobs=-1)
    )
    pipe_geral.fit(df_lags_all[num_all + cat_all], df_lags_all["DY_target"])
    joblib.dump(pipe_geral, SAIDA_DIR / "random_forest.pkl")
    print(f"  [✓] modelo/random_forest.pkl (fallback geral)")

    # meta.json
    r2_vals    = [v for v in resultados_r2.values()    if v > -999]
    r2_vals_sp = [v for v in resultados_r2_sp.values() if v > -999]
    meta_final = {
        "versao":                    "4.0-por-segmento-cdi",
        "estrategia":                "Modelo por segmento | CDI para Titulos | SELIC para demais",
        "features":                  "DY_lag1, DY_lag2, DY_lag3, PVP_lag1, CDI/SELIC, Tipo_do_Fundo",
        "pandemia_excluida":         f"{PANDEMIA_INI} a {PANDEMIA_FIM}",
        "segmentos_excluidos":       SEGS_EXCLUIR,
        "modelos_por_segmento":      modelos_meta,
        "r2_medio_com_pandemia":     round(float(np.mean(r2_vals)), 4)    if r2_vals    else None,
        "r2_medio_sem_pandemia":     round(float(np.mean(r2_vals_sp)), 4) if r2_vals_sp else None,
        "melhor_modelo":             "por segmento",
        "modelos":                   ["Random Forest", "Gradient Boosting", "XGBoost"],
        "n_fundos_total":            int(df[COL_SIGLA].nunique()),
        "dataset":                   Path(caminho_arquivo).name,
    }
    with open(SAIDA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_final, f, ensure_ascii=False, indent=2)
    print(f"  [✓] modelo/meta.json")

    # fundos_recentes.csv
    cols_rec = [COL_SIGLA, COL_DY, "DY_lag1", "DY_lag2", "DY_lag3", COL_SEGMENTO, COL_TIPO]
    if "PVP_lag1" in df_lags_all.columns:
        cols_rec.insert(3, "PVP_lag1")
    cols_rec = [c for c in cols_rec if c in df_lags_all.columns]
    ultimo = df_lags_all[cols_rec].groupby(COL_SIGLA).last().reset_index()
    ultimo.to_csv(SAIDA_DIR / "fundos_recentes.csv", index=False)
    print(f"  [✓] modelo/fundos_recentes.csv ({len(ultimo)} fundos)")

    # Resumo final
    print(f"\n{'='*58}")
    print(f"  {'Segmento':25s} {'Macro':6s} {'R² com pandemia':>16} {'R² sem pandemia':>16}")
    print(f"{'─'*58}")
    for seg in sorted(set(list(resultados_r2.keys()) + list(resultados_r2_sp.keys()))):
        r2c  = resultados_r2.get(seg, float("nan"))
        r2sp = resultados_r2_sp.get(seg, float("nan"))
        diff = r2sp - r2c if not (np.isnan(r2c) or np.isnan(r2sp)) else float("nan")
        seta = f"↑ +{diff:.4f}" if diff > 0 else f"↓ {diff:.4f}" if not np.isnan(diff) else ""
        macro = "CDI" if seg == "Titulos e Val. Mob." else "SELIC"
        print(f"  {seg:25s} {macro:6s} {r2c:>14.4f}    {r2sp:>14.4f}  {seta}")
    print(f"{'─'*58}")
    print(f"  {'Media':25s} {'':6s} {np.mean(r2_vals):>14.4f}    {np.mean(r2_vals_sp):>14.4f}")
    print(f"{'='*58}")
    print(f"  Pronto! Execute: uvicorn api:app --reload --port 8000")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", "-a", default="dataset_fiis_2019_2024_brapi_v2.xlsx")
    args = parser.parse_args()
    treinar(args.arquivo)