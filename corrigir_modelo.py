import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_excel("dataset_fiis_2019_2024.xlsx")
df["Data"] = pd.to_datetime(df["Data"])
df = df.sort_values(["Sigla","Data"]).reset_index(drop=True)

# Lags
df["DY_lag1"] = df.groupby("Sigla")["Dividendos_Yield"].shift(1)
df["DY_lag2"] = df.groupby("Sigla")["Dividendos_Yield"].shift(2)
df["DY_lag3"] = df.groupby("Sigla")["Dividendos_Yield"].shift(3)
df["DY_target"] = df.groupby("Sigla")["Dividendos_Yield"].shift(-1)
df["mes"] = df["Data"].dt.month.astype(float)
df["trimestre"] = df["Data"].dt.quarter.astype(float)
df["semestre"] = (df["Data"].dt.month > 6).astype(float)
df = df.dropna(subset=["DY_lag1","DY_lag2","DY_lag3","DY_target"]).reset_index(drop=True)

num_cols = ["DY_lag1","DY_lag2","DY_lag3","mes","trimestre","semestre"]
for col in ["SELIC","IFIX"]:
    if col in df.columns and df[col].notna().sum() > len(df)*0.3:
        num_cols.append(col)
cat_cols = ["Segmento","Tipo_do_Fundo"]

X = df[num_cols + cat_cols]
y = df["DY_target"]

# Split temporal
corte = df["Data"].quantile(0.8)
mask = df["Data"] < corte
X_train, y_train = X[mask], y[mask]
X_test, y_test = X[~mask], y[~mask]
print(f"Treino: {mask.sum()} | Teste: {(~mask).sum()}")

def pipe(estimador):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    return Pipeline([("pre", pre), ("modelo", estimador)])

modelos = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=5,
        min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, min_samples_leaf=5,
        subsample=0.8, random_state=42),
}

MODEL_DIR = Path("modelo")
meta = json.load(open(MODEL_DIR/"meta.json"))
resultados = {}

for nome, est in modelos.items():
    p = pipe(est)
    p.fit(X_train, y_train)
    preds = p.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mask2 = y_test != 0
    mape = mean_absolute_percentage_error(y_test[mask2], preds[mask2])*100
    cv = cross_val_score(p, X, y, cv=5, scoring="r2").mean()
    print(f"{nome}: R²={r2:.4f} | MAE={mae:.5f} | MAPE={mape:.1f}% | CV R²={cv:.4f}")
    slug = nome.lower().replace(" ","_")
    joblib.dump(p, MODEL_DIR/f"{slug}.pkl")
    resultados[nome] = {"r2":round(r2,4),"mae":round(mae,6),
                        "mape":round(mape,2),"cv_r2":round(cv,4)}

meta["metricas"] = resultados
meta["num_cols"] = num_cols
meta["cat_cols"] = cat_cols
meta["melhor_modelo"] = max(resultados, key=lambda n: resultados[n]["r2"])
json.dump(meta, open(MODEL_DIR/"meta.json","w"), ensure_ascii=False, indent=2)
print(f"\nMelhor: {meta['melhor_modelo']}")
print("Modelos salvos!")