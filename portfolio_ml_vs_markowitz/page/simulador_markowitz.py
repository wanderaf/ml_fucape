import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

def run():
    st.title("📊 Visualização – Simulação Markowitz (Rolling Window Corrigida)")
    st.info("Esta versão calcula a rentabilidade mensal corretamente com base no capital acumulado e adiciona Overfit e Max Drawdown.")

    # ========= utilitários de métricas =========
    def _sharpe(ret, rf=0.0):
        r = pd.Series(ret).dropna()
        if len(r) < 2 or r.std() == 0:
            return np.nan
        return float((r.mean() - rf) / r.std())

    def _sortino(ret, rf=0.0):
        r = pd.Series(ret).dropna()
        downside = r[r < rf]
        if len(downside) < 2 or downside.std() == 0:
            return np.nan
        return float((r.mean() - rf) / downside.std())

    def _max_drawdown_from_equity(equity_series):
        s = pd.Series(equity_series).astype(float)
        dd = (s / s.cummax()) - 1.0
        return float(dd.min()) * 100.0  # %

    # ========= função que replica a lógica Markowitz para uma lista de meses =========
    def _build_curve_markowitz(df_all, meses_periods, start_capital=100_000.0):
        rows = []
        capital = start_capital
        capital_prev = capital

        for mes in meses_periods:
            data_ref = mes.to_timestamp()

            # treino: tudo antes do mês alvo
            df_train = df_all[df_all["Date"] < data_ref]
            df_test  = df_all[df_all["AnoMes"] == mes]

            # matriz de retornos (linhas = datas, colunas = tickers)
            retornos = df_train.pivot_table(index="Date", columns="Ticker", values="RetFut20").dropna(axis=1)
            if retornos.shape[1] < 2:
                rows.append({"Data": data_ref, "Retorno": 0.0, "Capital": capital})
                continue

            mu  = retornos.mean()
            cov = retornos.cov()
            tickers = mu.index.tolist()

            def portfolio_variance(weights):
                return float(np.dot(weights.T, np.dot(cov, weights)))

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in tickers]
            initial_weights = np.ones(len(tickers)) / len(tickers)

            result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints)
            if not result.success:
                rows.append({"Data": data_ref, "Retorno": 0.0, "Capital": capital})
                continue

            weights = pd.Series(result.x, index=tickers)
            weights = weights[weights > 0.01]
            weights /= weights.sum()

            df_test_month = df_test[df_test["Ticker"].isin(weights.index)].copy().drop_duplicates("Ticker")
            if df_test_month.empty:
                rows.append({"Data": data_ref, "Retorno": 0.0, "Capital": capital})
                continue

            df_test_month["Peso"] = df_test_month["Ticker"].map(weights)
            df_test_month["Investimento"] = capital * df_test_month["Peso"]
            df_test_month["Retorno"] = df_test_month["RetFut20"]
            df_test_month["Valor Final"] = df_test_month["Investimento"] * (1 + df_test_month["Retorno"])

            capital = float(df_test_month["Valor Final"].sum())
            retorno_mensal = (capital / capital_prev) - 1.0
            capital_prev = capital

            rows.append({"Data": data_ref, "Retorno": retorno_mensal, "Capital": capital})

        if not rows:
            return pd.DataFrame(columns=["Data", "Retorno", "Capital"])
        return pd.DataFrame(rows).sort_values("Data")

    with st.spinner("🔄 Executando otimização..."):
        # ========= dados =========
        df = pd.read_csv("data/processed/dataset_ml.csv", parse_dates=["Date"])
        df = df.dropna(subset=["Date", "Ticker", "RetFut20"]).copy()
        df["AnoMes"] = df["Date"].dt.to_period("M")

        meses_rolling = sorted(df[df["Date"] >= "2021-01-01"]["AnoMes"].unique())

        resultados = []
        carteiras = []
        capital_inicial = 100000.0
        capital = capital_inicial
        capital_anterior = capital_inicial

        progresso = st.progress(0)
        total = len(meses_rolling)

        # ========= loop OOS (página original) =========
        for i, mes in enumerate(meses_rolling):
            data_ref = mes.to_timestamp()

            df_train = df[df["Date"] < data_ref]
            df_test = df[df["AnoMes"] == mes]

            retornos = df_train.pivot_table(index="Date", columns="Ticker", values="RetFut20").dropna(axis=1)
            if retornos.shape[1] < 2:
                progresso.progress((i + 1) / total)
                continue

            mu = retornos.mean()
            cov = retornos.cov()
            tickers = mu.index.tolist()

            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov, weights))

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in tickers]
            initial_weights = np.ones(len(tickers)) / len(tickers)

            result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints)

            if not result.success:
                progresso.progress((i + 1) / total)
                continue

            weights = pd.Series(result.x, index=tickers)
            weights = weights[weights > 0.01]
            weights /= weights.sum()

            df_test_month = df_test[df_test["Ticker"].isin(weights.index)].copy()
            df_test_month = df_test_month.drop_duplicates("Ticker")
            df_test_month["Peso"] = df_test_month["Ticker"].map(weights)
            df_test_month["Investimento"] = capital * df_test_month["Peso"]
            df_test_month["Retorno"] = df_test_month["RetFut20"]
            df_test_month["Valor Final"] = df_test_month["Investimento"] * (1 + df_test_month["Retorno"])
            df_test_month["Data"] = data_ref

            capital = float(df_test_month["Valor Final"].sum())
            retorno_mensal = (capital / capital_anterior) - 1
            capital_anterior = capital

            resultados.append({
                "Data": data_ref,
                "Retorno": retorno_mensal,
                "Capital": capital
            })

            carteiras.append(df_test_month[["Data", "Ticker", "Peso", "Investimento", "Valor Final", "Retorno"]])
            progresso.progress((i + 1) / total)

        df_result = pd.DataFrame(resultados).set_index("Data")
        df_result["Rentabilidade Acumulada"] = (1 + df_result["Retorno"]).cumprod()

        # ========= Índices OOS + NOVOS (MDD/Overfit) =========
        st.subheader("📊 Índices de Desempenho")

        retorno_medio = df_result["Retorno"].mean()
        desvio_padrao = df_result["Retorno"].std()
        sharpe_ratio_oos = retorno_medio / desvio_padrao if desvio_padrao != 0 else np.nan

        retornos_negativos = df_result["Retorno"][df_result["Retorno"] < 0]
        desvio_negativo = retornos_negativos.std()
        sortino_ratio_oos = retorno_medio / desvio_negativo if desvio_negativo != 0 else np.nan

        # ===== curvas treino/teste para Overfit e MDD =====
        cutoff = pd.to_datetime("2020-12-31")
        meses_train = sorted(df[df["Date"] <= cutoff]["AnoMes"].unique())
        meses_test  = sorted(df[df["Date"] >  cutoff]["AnoMes"].unique())

        df_train_curve = _build_curve_markowitz(df, meses_train, start_capital=100_000.0)
        df_test_curve  = _build_curve_markowitz(df, meses_test,  start_capital=100_000.0)

        sharpe_tr  = _sharpe(df_train_curve["Retorno"]) if not df_train_curve.empty else np.nan
        sortino_tr = _sortino(df_train_curve["Retorno"]) if not df_train_curve.empty else np.nan
        sharpe_te  = _sharpe(df_test_curve["Retorno"])  if not df_test_curve.empty  else np.nan
        sortino_te = _sortino(df_test_curve["Retorno"]) if not df_test_curve.empty else np.nan

        overfit_ratio = np.nan
        if pd.notna(sharpe_tr) and pd.notna(sharpe_te) and sharpe_te != 0:
            overfit_ratio = float(sharpe_tr / sharpe_te)

        mdd_tr = _max_drawdown_from_equity(df_train_curve["Capital"]) if not df_train_curve.empty else np.nan
        mdd_te = _max_drawdown_from_equity(df_test_curve["Capital"])  if not df_test_curve.empty  else np.nan

        # ===== exibição lado a lado =====
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔹 Sharpe (OOS)", f"{sharpe_ratio_oos:.4f}" if pd.notna(sharpe_ratio_oos) else "–")
        c2.metric("🔻 Sortino (OOS)", f"{sortino_ratio_oos:.4f}" if pd.notna(sortino_ratio_oos) else "–")
        c3.metric("📉 Max Drawdown (OOS)", f"{mdd_te:.2f}%" if pd.notna(mdd_te) else "–")
        c4.metric("🧪 Overfit Ratio (Sharpe T/Te)", f"{overfit_ratio:.3f}" if pd.notna(overfit_ratio) else "–")

        # ========= composição de carteiras (para tabela) =========
        if carteiras:
            df_carteiras = pd.concat(carteiras).reset_index(drop=True)
            df_carteiras["Data"] = pd.to_datetime(df_carteiras["Data"])
        else:
            st.warning("⚠️ Nenhuma carteira foi formada.")
            return

        # ========= salvar índices principais para comparativo =========
        os.makedirs("data/results", exist_ok=True)
        with open("data/results/markowitz_indices.txt", "w") as f:
            f.write(f"{sharpe_te if pd.notna(sharpe_te) else np.nan},{sortino_te if pd.notna(sortino_te) else np.nan}")

        # complementares
        with open("data/results/markowitz_indices_train.txt", "w") as f:
            f.write(f"{sharpe_tr if pd.notna(sharpe_tr) else np.nan},{sortino_tr if pd.notna(sortino_tr) else np.nan}")
        with open("data/results/markowitz_indices_test.txt", "w") as f:
            f.write(f"{sharpe_te if pd.notna(sharpe_te) else np.nan},{sortino_te if pd.notna(sortino_te) else np.nan}")
        with open("data/results/markowitz_overfit_ratio.txt", "w") as f:
            f.write(f"{overfit_ratio}")
        with open("data/results/markowitz_max_drawdown_test.txt", "w") as f:
            f.write(f"{mdd_te}")

    st.success("✅ Otimização concluída com sucesso!")

    # ========= gráficos =========
    st.subheader("📈 Rentabilidade Acumulada")
    st.line_chart(df_result["Rentabilidade Acumulada"])

    st.subheader("📊 Capital Total por Mês")
    st.line_chart(df_result["Capital"])

    st.subheader("📋 Composição da Carteira por Mês")
    meses_disponiveis = sorted(df_carteiras["Data"].dt.strftime("%Y-%m").unique())
    mes_selecionado = st.selectbox("Selecione o mês:", meses_disponiveis)
    data_ref = pd.to_datetime(mes_selecionado + "-01")

    carteira_mes = df_carteiras[df_carteiras["Data"] == data_ref].copy()
    carteira_mes.drop(columns="Data", inplace=True)
    st.dataframe(carteira_mes.reset_index(drop=True))

    # ========= salvamentos para página comparativa =========
    df_result.reset_index()[["Data", "Capital"]].to_csv("data/results/markowitz_capital.csv", index=False)
    # também salva curvas treino/teste (para cálculo de overfit na página de comparação, se necessário)
    if 'df_train_curve' in locals() and not df_train_curve.empty:
        df_train_curve[["Data", "Capital"]].to_csv("data/results/markowitz_capital_train.csv", index=False)
    if 'df_test_curve' in locals() and not df_test_curve.empty:
        df_test_curve[["Data", "Capital"]].to_csv("data/results/markowitz_capital_test.csv", index=False)
