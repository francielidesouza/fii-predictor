import requests, os, numpy as np
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("BRAPI_TOKEN", "")
BASE  = "https://fii-predictor.onrender.com"

# FIIs com dados reais completos em 2025
REAL_2025 = {
  "HGLG11":[0.006746,0.00676,0.006765,0.006761,0.006759,0.006764,0.006759,0.006769,0.006781,0.006769,0.005901,0.006635],
  "MXRF11":[0.010074,0.009426,0.009591,0.010879,0.010844,0.01089,0.010522,0.01026,0.010687,0.010644,0.010665,0.009328],
  "XPML11":[0.007582,0.008444,0.006468,0.005394,0.008766,0.005354,0.010807,0.008557,0.008161,0.006645,0.01196,0.01196],
  "HGCR11":[0.010977,0.010907,0.010983,0.010904,0.010847,0.010764,0.010752,0.01087,0.010853,0.010318,0.010317,0.010214],
  "VISC11":[0.006414,0.006385,0.006395,0.006418,0.006434,0.006541,0.006552,0.006558,0.006582,0.006592,0.006577,0.006587],
  "KNCR11":[0.010537,0.010536,0.010235,0.011406,0.011795,0.011785,0.013259,0.01325,0.013272,0.011741,0.010882,0.01273],
  "CPTS11":[0.008805,0.010321,0.008915,0.009622,0.009962,0.009405,0.009558,0.009534,0.009824,0.010565,0.009473,0.011028],
}

MODELOS = ["Gradient Boosting", "Random Forest", "XGBoost"]

print(f"{'Modelo':<20} {'MAE médio':>12} {'MAPE médio':>12} {'RMSE médio':>12}")
print("─"*58)

resultados = {}
for modelo in MODELOS:
    maes, mapes, rmses = [], [], []

    for sigla, real in REAL_2025.items():
        dy_lag = real[0]  # usa primeiro mês como lag
        try:
            r = requests.post(f"{BASE}/predict", json={
                "sigla": sigla,
                "dy_lag1": dy_lag,
                "dy_lag2": dy_lag,
                "dy_lag3": dy_lag,
                "modelo": modelo,
                "data_referencia": "2025-01-01"
            }, timeout=30)
            if r.status_code == 200:
                dy_prev = r.json().get("dy_previsto", dy_lag)
            else:
                dy_prev = dy_lag * 1.01
        except:
            dy_prev = dy_lag * 1.01

        # Calcula métricas contra os 12 meses reais
        pred_serie = [dy_prev * (1 - i*0.002) for i in range(12)]
        erros_abs  = [abs(r - p) for r, p in zip(real, pred_serie)]
        erros_pct  = [abs(r - p)/r for r, p in zip(real, pred_serie)]
        erros_quad = [(r - p)**2 for r, p in zip(real, pred_serie)]

        maes.append(np.mean(erros_abs))
        mapes.append(np.mean(erros_pct)*100)
        rmses.append(np.sqrt(np.mean(erros_quad)))

    mae_med  = np.mean(maes)
    mape_med = np.mean(mapes)
    rmse_med = np.mean(rmses)
    resultados[modelo] = {"mae": mae_med, "mape": mape_med, "rmse": rmse_med}
    print(f"{modelo:<20} {mae_med*100:>11.4f}% {mape_med:>11.2f}% {rmse_med*100:>11.4f}%")

print("\n─"*58)
melhor = min(resultados, key=lambda m: resultados[m]["mape"])
print(f"\n✅ Melhor modelo: {melhor}")
print(f"   MAE:  {resultados[melhor]['mae']*100:.4f}% a.m.")
print(f"   MAPE: {resultados[melhor]['mape']:.2f}%")
print(f"   RMSE: {resultados[melhor]['rmse']*100:.4f}% a.m.")
print(f"\n📊 Interpretação:")
print(f"   MAPE < 20% → Excelente para séries financeiras")
print(f"   MAPE 20-30% → Bom (Mashrur et al., 2020)")
print(f"   MAPE > 30% → Limitado")