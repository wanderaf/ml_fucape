import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def run():
    st.title("📊 Comparativo Final entre Modelos")

    st.markdown("""
    Esta página apresenta a comparação do desempenho acumulado entre os modelos de portfólio: 
    Markowitz (tradicional) e algoritmos de Aprendizado de Máquina (Random Forest, XGBoost, MLP e SVM).
    Os resultados são gerados a partir dos dados reais salvos em `data/results`.
    """)

    modelos = {
        "markowitz": "Markowitz",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "mlp": "MLP",
        "svm": "SVM"
    }

    df_final = pd.DataFrame()
    df_stats = {}

    for arquivo, nome_modelo in modelos.items():
        caminho_capital = f"data/results/{arquivo}_capital.csv"
        caminho_indices = f"data/results/{arquivo}_indices.txt"

        if os.path.exists(caminho_capital):
            df = pd.read_csv(caminho_capital, parse_dates=["Data"])
            df = df[["Data", "Capital"]].sort_values("Data")
            df["Retorno"] = df["Capital"].pct_change()

            retorno_acumulado = (df["Capital"].iloc[-1] / df["Capital"].iloc[0] - 1) * 100
            retorno_medio = df["Retorno"].mean() * 100
            desvio_padrao = df["Retorno"].std() * 100
            capital_final = df["Capital"].iloc[-1]

            # Lê os índices Sharpe e Sortino salvos
            sharpe, sortino = np.nan, np.nan
            if os.path.exists(caminho_indices):
                try:
                    with open(caminho_indices, "r") as f:
                        sharpe, sortino = map(float, f.read().strip().split(","))
                except:
                    pass  # se falhar, mantêm-se como NaN

            df_stats[nome_modelo] = {
                "Retorno Acumulado (%)": retorno_acumulado,
                "Retorno Médio Mensal (%)": retorno_medio,
                "Desvio Padrão (%)": desvio_padrao,
                "Índice de Sharpe": sharpe,
                "Índice de Sortino": sortino,
                "Capital Final (R$)": capital_final
            }

            df_modelo = df[["Data", "Capital"]].rename(columns={"Capital": nome_modelo})
            df_final = df_modelo if df_final.empty else pd.merge(df_final, df_modelo, on="Data", how="outer")

    if df_final.empty:
        st.error("❌ Nenhum resultado encontrado em data/results. Execute os modelos antes de visualizar o comparativo.")
        return

    # Gráfico de evolução
    df_melt = df_final.melt(id_vars="Data", var_name="Modelo", value_name="Capital")
    st.subheader("📈 Evolução do Capital ao Longo do Tempo")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melt, x="Data", y="Capital", hue="Modelo", marker="o")
    plt.title("Comparação de Capital Acumulado entre Modelos")
    plt.xlabel("Data")
    plt.ylabel("Capital (R$)")
    plt.grid(True)
    st.pyplot(plt.gcf())

    # Tabela de capital mensal
    st.subheader("📋 Tabela de Capital Mensal")
    st.dataframe(df_final.set_index("Data").style.format("R$ {:,.2f}"))

    # Tabela de indicadores
    st.subheader("📊 Indicadores Comparativos entre Modelos")
    df_stats_final = pd.DataFrame(df_stats).T
    df_stats_final = df_stats_final[
        [
            "Retorno Acumulado (%)",
            "Retorno Médio Mensal (%)",
            "Desvio Padrão (%)",
            "Índice de Sharpe",
            "Índice de Sortino",
            "Capital Final (R$)"
        ]
    ]
    st.dataframe(df_stats_final.style.format({
        "Retorno Acumulado (%)": "{:.2f}%",
        "Retorno Médio Mensal (%)": "{:.2f}%",
        "Desvio Padrão (%)": "{:.2f}%",
        "Índice de Sharpe": "{:.4f}",
        "Índice de Sortino": "{:.4f}",
        "Capital Final (R$)": "R$ {:,.2f}"
    }))
