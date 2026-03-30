import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ---------- utils ----------
def max_drawdown(series: pd.Series) -> float:
    """Retorna o Max Drawdown (%) de uma série de capital/nivel."""
    s = series.astype(float).copy()
    running_max = s.cummax()
    dd = (s / running_max) - 1.0
    return float(dd.min()) * 100.0  # em %

def sharpe_from_returns(ret: pd.Series, rf: float = 0.0) -> float:
    """Sharpe simples com rf=0 por padrão. Assume periodicidade da série (ex.: mensal)."""
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    # anualização opcional poderia ser adicionada se você desejar
    return float((r.mean() - rf) / r.std())

def load_capital_file(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Data"])
        if {"Data", "Capital"}.issubset(df.columns):
            df = df[["Data", "Capital"]].sort_values("Data")
            return df
    return None

def try_overfit_ratio(model_key: str) -> float | None:
    """
    Tenta calcular o overfit ratio do modelo:
    - procura arquivos train/test de capital e calcula Sharpe em cada um.
    - overfit_ratio = sharpe_train / sharpe_test
    Retorna NaN se não conseguir.
    """
    candidates_train = [
        f"data/results/{model_key}_capital_train.csv",
        f"data/results/{model_key}_capital_in_sample.csv",
        f"data/results/{model_key}_train_capital.csv",
    ]
    candidates_test = [
        f"data/results/{model_key}_capital_test.csv",
        f"data/results/{model_key}_capital_out_of_sample.csv",
        f"data/results/{model_key}_test_capital.csv",
    ]
    train_path = next((p for p in candidates_train if os.path.exists(p)), None)
    test_path  = next((p for p in candidates_test  if os.path.exists(p)), None)

    if not train_path or not test_path:
        return np.nan

    df_tr = load_capital_file(train_path)
    df_te = load_capital_file(test_path)
    if df_tr is None or df_te is None:
        return np.nan

    r_tr = df_tr["Capital"].pct_change()
    r_te = df_te["Capital"].pct_change()
    sharpe_tr = sharpe_from_returns(r_tr)
    sharpe_te = sharpe_from_returns(r_te)
    if np.isnan(sharpe_tr) or np.isnan(sharpe_te) or sharpe_te == 0:
        return np.nan

    return float(sharpe_tr / sharpe_te)

# ---------- página ----------
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

    # ---------- 1) Carrega resultados dos modelos ----------
    for arquivo, nome_modelo in modelos.items():
        caminho_capital = f"data/results/{arquivo}_capital.csv"
        caminho_indices = f"data/results/{arquivo}_indices.txt"

        if os.path.exists(caminho_capital):
            df = pd.read_csv(caminho_capital, parse_dates=["Data"])
            df = df[["Data", "Capital"]].sort_values("Data")

            # retorno periódico
            df["Retorno"] = df["Capital"].pct_change()

            # métricas
            retorno_acumulado = (df["Capital"].iloc[-1] / df["Capital"].iloc[0] - 1) * 100
            retorno_medio = df["Retorno"].mean() * 100
            desvio_padrao = df["Retorno"].std() * 100
            capital_final = df["Capital"].iloc[-1]
            mdd = max_drawdown(df["Capital"])

            # índices Sharpe/Sortino salvos (se existirem)
            sharpe, sortino = np.nan, np.nan
            if os.path.exists(caminho_indices):
                try:
                    with open(caminho_indices, "r") as f:
                        sharpe, sortino = map(float, f.read().strip().split(","))
                except:
                    pass

            # overfit ratio (opcional se existirem arquivos train/test)
            overfit_ratio = try_overfit_ratio(arquivo)

            df_stats[nome_modelo] = {
                "Retorno Acumulado (%)": retorno_acumulado,
                "Retorno Médio Mensal (%)": retorno_medio,
                "Desvio Padrão (%)": desvio_padrao,
                "Índice de Sharpe": sharpe,
                "Índice de Sortino": sortino,
                "Max Drawdown (%)": mdd,
                "Overfit Ratio (Sharpe T/Te)": overfit_ratio,
                "Capital Final (R$)": capital_final
            }

            # agrega capital ao df_final
            df_modelo = df[["Data", "Capital"]].rename(columns={"Capital": nome_modelo})
            df_final = df_modelo if df_final.empty else pd.merge(df_final, df_modelo, on="Data", how="outer")

    if df_final.empty:
        st.error("❌ Nenhum resultado encontrado em data/results. Execute os modelos antes de visualizar o comparativo.")
        return

    # ---------- 2) Carregar IBOV (^BVSP) ----------
    caminho_ibov = "data/prices/^BVSP.csv"
    df_ibov_raw = None
    df_ibov_norm = None
    ibov_stats = None

    if os.path.exists(caminho_ibov):
        bruto_ibov = pd.read_csv(caminho_ibov, header=[0, 1])

        # achata colunas MultiIndex: ('Close','^BVSP') -> 'Close_^BVSP'
        bruto_ibov.columns = [
            "_".join([str(x) for x in col if str(x) != "nan"]).strip("_")
            for col in bruto_ibov.columns
        ]

        # primeira coluna contém datas ('Date', '2014-01-02', ...)
        primeira_coluna = bruto_ibov.columns[0]
        bruto_ibov = bruto_ibov.rename(columns={primeira_coluna: "Data"})

        # identifica coluna de preço preferencial ("Close_*") ou fallback "Price_*"
        col_preco = next((c for c in bruto_ibov.columns if c.lower().startswith("close")), None)
        if col_preco is None:
            col_preco = next((c for c in bruto_ibov.columns if c.lower().startswith("price")), None)

        if col_preco is not None:
            df_ibov_raw = bruto_ibov[["Data", col_preco]].copy()
            df_ibov_raw.columns = ["Data", "IBOV"]
            df_ibov_raw["Data"] = pd.to_datetime(df_ibov_raw["Data"], errors="coerce")
            df_ibov_raw = df_ibov_raw.dropna(subset=["Data"]).sort_values("Data")

            # alinhar ao intervalo dos modelos
            data_min = df_final["Data"].min()
            data_max = df_final["Data"].max()
            df_ibov_raw = df_ibov_raw[(df_ibov_raw["Data"] >= data_min) & (df_ibov_raw["Data"] <= data_max)]

            if not df_ibov_raw.empty:
                # métricas IBOV
                serie_preco = df_ibov_raw["IBOV"].astype(float).reset_index(drop=True)
                retornos = serie_preco.pct_change()

                ibov_stats = {
                    "Retorno Acumulado (%)": (serie_preco.iloc[-1] / serie_preco.iloc[0] - 1.0) * 100.0,
                    "Retorno Médio Mensal (%)": retornos.mean() * 100.0,
                    "Desvio Padrão (%)": retornos.std() * 100.0,
                    "Índice de Sharpe": np.nan,    # não calculado com RF aqui
                    "Índice de Sortino": np.nan,   # idem
                    "Max Drawdown (%)": max_drawdown(serie_preco),
                    "Overfit Ratio (Sharpe T/Te)": np.nan,  # não aplicável
                    "Capital Final (R$)": serie_preco.iloc[-1]
                }

                # série normalizada do IBOV para gráfico 2
                df_ibov_norm = df_ibov_raw.copy()
                primeiro_ibov = df_ibov_norm["IBOV"].iloc[0]
                df_ibov_norm["IBOV"] = (df_ibov_norm["IBOV"] / primeiro_ibov - 1.0) * 100.0

                # incluir IBOV no df_final apenas se ainda não existir
                if "IBOV" not in df_final.columns:
                    df_final = pd.merge(df_final, df_ibov_raw, on="Data", how="left")
        else:
            st.warning("⚠️ Não encontrei coluna de preço ('Close_*' ou 'Price_*') no arquivo do IBOV.")
    else:
        st.warning("⚠️ Arquivo do IBOV não encontrado em data/prices/^BVSP.csv.")

    # Remover possíveis colunas duplicadas
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # ---------- 3) Gráfico 1: Evolução do Capital/Nível (inclui IBOV) ----------
    st.subheader("📈 Evolução do Capital ao Longo do Tempo")
    df_melt = df_final.melt(id_vars="Data", var_name="Modelo", value_name="Capital")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melt, x="Data", y="Capital", hue="Modelo", marker="o")
    plt.title("Comparação de Capital/Nível ao Longo do Tempo (Modelos e IBOV)")
    plt.xlabel("Data")
    plt.ylabel("Capital / Nível (R$ ou pontos)")
    plt.grid(True)
    st.pyplot(plt.gcf())

    # ---------- 4) Gráfico 2: Retorno Acumulado (%) vs IBOV ----------
    st.subheader("📈 Retorno Acumulado (%) vs IBOV")
    cols_modelos = [c for c in df_final.columns if c not in ("Data", "IBOV")]
    df_retacum_models = df_final[["Data"] + cols_modelos].copy()
    for col in cols_modelos:
        primeira = df_retacum_models[col].dropna().iloc[0]
        df_retacum_models[col] = (df_retacum_models[col] / primeira - 1.0) * 100.0

    # juntar IBOV normalizado (se disponível)
    if df_ibov_norm is not None and not df_ibov_norm.empty:
        df_plot = pd.merge(df_retacum_models, df_ibov_norm, on="Data", how="left")
    else:
        df_plot = df_retacum_models.copy()

    df_plot_melt = df_plot.melt(id_vars="Data", var_name="Série", value_name="Retorno Acumulado (%)")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot_melt, x="Data", y="Retorno Acumulado (%)", hue="Série", marker="o")
    plt.title("Retorno Acumulado (%) Normalizado vs IBOV")
    plt.xlabel("Data")
    plt.ylabel("Retorno Acumulado (%) desde o início")
    plt.grid(True)
    st.pyplot(plt.gcf())

    # ---------- 5) Gráfico 3: Drawdown (%) ao longo do tempo ----------
    st.subheader("📉 Drawdown (%) ao longo do tempo")
    plt.figure(figsize=(12, 6))
    for col in df_final.columns:
        if col == "Data": 
            continue
        series = df_final[col].astype(float)
        dd = (series / series.cummax() - 1.0) * 100.0
        plt.plot(df_final["Data"], dd, label=col)
    plt.title("Drawdown (%) - Modelos e IBOV")
    plt.xlabel("Data")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

    # ---------- 6) Tabela: Capital Mensal (inclui IBOV) ----------
    st.subheader("📋 Tabela de Capital Mensal")
    st.dataframe(df_final.set_index("Data").style.format("R$ {:,.2f}"))

    # ---------- 7) Tabela: Indicadores Comparativos (inclui IBOV e MDD/Overfit) ----------
    st.subheader("📊 Indicadores Comparativos entre Modelos")

    if ibov_stats is not None:
        df_stats["IBOV"] = ibov_stats

    df_stats_final = pd.DataFrame(df_stats).T
    # ordena colunas na exibição
    cols_order = [
        "Retorno Acumulado (%)",
        "Retorno Médio Mensal (%)",
        "Desvio Padrão (%)",
        "Índice de Sharpe",
        "Índice de Sortino",
        "Max Drawdown (%)",
        "Overfit Ratio (Sharpe T/Te)",
        "Capital Final (R$)"
    ]
    df_stats_final = df_stats_final.reindex(columns=cols_order)

    st.dataframe(
        df_stats_final.style.format({
            "Retorno Acumulado (%)": "{:.2f}%",
            "Retorno Médio Mensal (%)": "{:.2f}%",
            "Desvio Padrão (%)": "{:.2f}%",
            "Índice de Sharpe": "{:.4f}",
            "Índice de Sortino": "{:.4f}",
            "Max Drawdown (%)": "{:.2f}%",
            "Overfit Ratio (Sharpe T/Te)": "{:.3f}",
            "Capital Final (R$)": "R$ {:,.2f}"
        })
    )
