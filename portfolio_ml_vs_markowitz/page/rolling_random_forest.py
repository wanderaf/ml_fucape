import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import os

# =========================
# CONFIGURAÇÕES
# =========================
MODEL_KEY = "random_forest"
TRAIN_END_DATE = "2020-12-31"   # corte entre treino (<=) e teste (>)
START_CAPITAL = 100_000.0

FEATURES = ["Ret5", "Ret10", "Ret20", "MA5", "MA20", "Vol20"]
TARGET = "RetFut20"

N_ESTIMATORS = 100
RANDOM_STATE = 42

# =========================
# FUNÇÕES AUXILIARES
# =========================
def sharpe_ratio(ret, rf=0.0):
    r = pd.Series(ret).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float((r.mean() - rf) / r.std())

def sortino_ratio(ret, rf=0.0):
    r = pd.Series(ret).dropna()
    downside = r[r < rf]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return float((r.mean() - rf) / downside.std())

def max_drawdown_from_equity(equity_series):
    s = pd.Series(equity_series).astype(float)
    dd = (s / s.cummax()) - 1.0
    return float(dd.min()) * 100.0  # em %

def build_curve_rolling(df_all: pd.DataFrame, meses_periods) -> pd.DataFrame:
    """
    Replica a sua lógica rolling:
    - Treina até o mês anterior ao mês alvo
    - Seleciona top-10 por predição
    - Retorno do mês = média do TARGET dos top-10
    Retorna DataFrame [Data, Retorno, Capital]
    """
    rows = []
    for mes in meses_periods:
        data_ref = mes.to_timestamp()

        df_tr = df_all[df_all["Date"] < data_ref]
        df_te = df_all[df_all["AnoMes"] == mes]
        if df_tr.empty or df_te.empty:
            continue

        X_tr, y_tr = df_tr[FEATURES], df_tr[TARGET]
        X_te = df_te[FEATURES]

        mdl = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        mdl.fit(X_tr, y_tr)

        df_te = df_te.copy()
        df_te["Predito"] = mdl.predict(X_te)
        top10 = df_te.sort_values("Predito", ascending=False).head(10)

        ret_medio = top10[TARGET].mean()
        rows.append({"Data": data_ref, "Retorno": ret_medio})

    if not rows:
        return pd.DataFrame(columns=["Data", "Retorno", "Capital"])

    df_out = pd.DataFrame(rows).sort_values("Data")
    capital = [START_CAPITAL]
    for r in df_out["Retorno"].values:
        capital.append(capital[-1] * (1.0 + (0.0 if pd.isna(r) else r)))
    df_out["Capital"] = capital[1:]
    return df_out

# =========================
# CABEÇALHO
# =========================
st.title("🌲 Random Forest — Backtest Rolling")
st.caption("Estratégia: janela rolling mensal; seleção top-10 por predição do retorno futuro (RetFut20).")

# =========================
# CARREGA DATASET
# =========================
df = pd.read_csv("data/processed/dataset_ml.csv", parse_dates=["Date"]).dropna(subset=["Date"]).copy()
df["AnoMes"] = df["Date"].dt.to_period("M")

# =========================
# 1) LOOP OOS (seu código original, 2021+), SALVA EM output/
# =========================
resultados = []
carteiras = []
meses_oos = sorted(df[df["Date"] >= "2021-01-01"]["AnoMes"].unique())

st.info("Executando Random Forest em janela rolling (OOS 2021+)...")
for mes in tqdm(meses_oos):
    data_ref = mes.to_timestamp()

    df_train_m = df[df["Date"] < data_ref]
    df_test_m  = df[df["AnoMes"] == mes]
    if df_train_m.empty or df_test_m.empty:
        continue

    X_train = df_train_m[FEATURES]
    y_train = df_train_m[TARGET]
    X_test  = df_test_m[FEATURES]

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    df_test_m = df_test_m.copy()
    df_test_m["Predito"] = model.predict(X_test)

    mse = mean_squared_error(df_test_m[TARGET], df_test_m["Predito"])
    r2  = r2_score(df_test_m[TARGET], df_test_m["Predito"])

    top10 = df_test_m.sort_values("Predito", ascending=False).head(10)
    retorno_medio = top10[TARGET].mean()

    resultados.append({"Data": data_ref, "Retorno": retorno_medio, "MSE": mse, "R2": r2})

    top10_result = top10[["Ticker", "Predito", TARGET]].copy()
    top10_result["Data"] = data_ref
    carteiras.append(top10_result)

df_result = pd.DataFrame(resultados).set_index("Data").sort_index()
df_result["Rentabilidade Acumulada"] = (1 + df_result["Retorno"]).cumprod() - 1

os.makedirs("output", exist_ok=True)
df_result.to_csv("output/rentabilidade_rf.csv")
pd.concat(carteiras).to_parquet("output/carteiras_rf.parquet", index=False)

st.success("✅ Execução OOS finalizada.")
st.caption("Arquivos salvos: output/rentabilidade_rf.csv, output/carteiras_rf.parquet")

# =========================
# 2) (NOVO) CONSTRUIR CURVAS TREINO/TESTE p/ MÉTRICAS
# =========================
cutoff = pd.to_datetime(TRAIN_END_DATE)
meses_train = sorted(df[df["Date"] <= cutoff]["AnoMes"].unique())
meses_test  = sorted(df[df["Date"] >  cutoff]["AnoMes"].unique())

df_train_curve = build_curve_rolling(df, meses_train)
df_test_curve  = build_curve_rolling(df, meses_test)

# =========================
# 3) CALCULAR MÉTRICAS (Sharpe/Sortino/MDD/Overfit)
# =========================
ret_tr = df_train_curve["Retorno"] if not df_train_curve.empty else pd.Series(dtype=float)
ret_te = df_test_curve["Retorno"]  if not df_test_curve.empty  else pd.Series(dtype=float)

sharpe_te  = sharpe_ratio(ret_te)
sortino_te = sortino_ratio(ret_te)

sharpe_tr  = sharpe_ratio(ret_tr)
sortino_tr = sortino_ratio(ret_tr)

overfit_ratio = np.nan
if pd.notna(sharpe_tr) and pd.notna(sharpe_te) and sharpe_te != 0:
    overfit_ratio = float(sharpe_tr / sharpe_te)

mdd_tr = max_drawdown_from_equity(df_train_curve["Capital"]) if not df_train_curve.empty else np.nan
mdd_te = max_drawdown_from_equity(df_test_curve["Capital"])  if not df_test_curve.empty  else np.nan
mdd_full = np.nan
if not df_train_curve.empty or not df_test_curve.empty:
    df_full_curve = pd.concat([df_train_curve, df_test_curve], ignore_index=True).sort_values("Data")
    mdd_full = max_drawdown_from_equity(df_full_curve["Capital"])

# =========================
# 4) EXIBIÇÃO NA PÁGINA (AGORA APARECE!)
# =========================
st.subheader("📊 Índices de Desempenho (OOS e Overfit)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Índice de Sharpe (OOS)", f"{sharpe_te:.4f}" if pd.notna(sharpe_te) else "–")
c2.metric("Índice de Sortino (OOS)", f"{sortino_te:.4f}" if pd.notna(sortino_te) else "–")
c3.metric("Max Drawdown (OOS)", f"{mdd_te:.2f}%" if pd.notna(mdd_te) else "–")
c4.metric("Overfit Ratio (Sharpe T/Te)", f"{overfit_ratio:.3f}" if pd.notna(overfit_ratio) else "–")

with st.expander("Ver detalhamento (Treino × Teste × Full)"):
    df_idx = pd.DataFrame({
        "Métrica": ["Sharpe", "Sortino", "Max Drawdown (%)", "Overfit Ratio (Sharpe T/Te)"],
        "Treino":  [sharpe_tr, sortino_tr, mdd_tr, np.nan],
        "Teste":   [sharpe_te, sortino_te, mdd_te, overfit_ratio],
        "Full":    [np.nan, np.nan, mdd_full, np.nan]
    })
    st.dataframe(
        df_idx.style.format({
            "Treino": "{:.4f}",
            "Teste": "{:.4f}",
            "Full": "{:.4f}"
        })
    )

# =========================
# 5) (OPCIONAL) SALVAR ÍNDICES PARA OUTRAS PÁGINAS
# =========================
os.makedirs("data/results", exist_ok=True)
with open(f"data/results/{MODEL_KEY}_indices.txt", "w") as f:
    f.write(f"{sharpe_te},{sortino_te}")
with open(f"data/results/{MODEL_KEY}_overfit_ratio.txt", "w") as f:
    f.write(f"{overfit_ratio}")
with open(f"data/results/{MODEL_KEY}_max_drawdown_test.txt", "w") as f:
    f.write(f"{mdd_te}")
