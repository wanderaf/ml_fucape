import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def run():
    st.title("📊 Visualização – Simulação Markowitz (Rolling Window Corrigida)")

    st.info("Esta versão calcula a rentabilidade mensal corretamente com base no capital acumulado.")

    with st.spinner("🔄 Executando otimização..."):

        df = pd.read_csv("data/processed/dataset_ml.csv", parse_dates=["Date"])
        df = df.dropna(subset=["Date", "Ticker", "RetFut20"])
        df["AnoMes"] = df["Date"].dt.to_period("M")

        meses_rolling = sorted(df[df["Date"] >= "2021-01-01"]["AnoMes"].unique())

        resultados = []
        carteiras = []
        capital_inicial = 100000
        capital = capital_inicial
        capital_anterior = capital_inicial

        progresso = st.progress(0)
        total = len(meses_rolling)

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

            capital = df_test_month["Valor Final"].sum()
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

        # === Cálculo de Índices de Desempenho ===
        st.subheader("📊 Índices de Desempenho")

        retorno_medio = df_result["Retorno"].mean()
        desvio_padrao = df_result["Retorno"].std()
        sharpe_ratio = retorno_medio / desvio_padrao if desvio_padrao != 0 else np.nan

        retornos_negativos = df_result["Retorno"][df_result["Retorno"] < 0]
        desvio_negativo = retornos_negativos.std()
        sortino_ratio = retorno_medio / desvio_negativo if desvio_negativo != 0 else np.nan

        col1, col2 = st.columns(2)
        col1.metric("🔹 Índice de Sharpe", f"{sharpe_ratio:.4f}")
        col2.metric("🔻 Índice de Sortino", f"{sortino_ratio:.4f}")

        df_carteiras = pd.concat(carteiras).reset_index(drop=True)
        df_carteiras["Data"] = pd.to_datetime(df_carteiras["Data"])

        with open("data/results/markowitz_indices.txt", "w") as f:
            f.write(f"{sharpe_ratio:.6f},{sortino_ratio:.6f}")

    st.success("✅ Otimização concluída com sucesso!")

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

    df_result[["Capital"]].to_csv(f"data/results/markowitz_capital.csv")
