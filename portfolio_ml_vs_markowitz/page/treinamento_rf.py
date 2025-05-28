import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import shap
import matplotlib.pyplot as plt

def run():
    st.title("🌳 Random Forest – Classificação Binária com Rolling Window de 12 Meses")
    st.info("Modelo classificador para prever retornos positivos, com rebalanceamento mensal e explicabilidade SHAP.")

    with st.spinner("🔄 Executando simulação..."):

        df = pd.read_csv("data/processed/dataset_ml.csv", parse_dates=["Date"])
        df = df.dropna(subset=["Date"])
        df["AnoMes"] = df["Date"].dt.to_period("M")
        df["Momentum"] = df["MA5"] - df["MA20"]
        df["Classe"] = (df["RetFut20"] > 0).astype(int)

        features = ["Ret5", "Ret10", "Ret20", "MA5", "MA20", "Vol20", "Momentum"]
        target = "Classe"

        meses_rolling = sorted(df[df["Date"] >= "2021-01-01"]["AnoMes"].unique())

        resultados = []
        carteiras = []
        capital = 100000
        capital_anterior = capital

        progresso = st.progress(0)
        total = len(meses_rolling)

        for i, mes in enumerate(meses_rolling):
            data_ref = mes.to_timestamp()
            data_inicio = data_ref - pd.DateOffset(months=12)

            df_train = df[(df["Date"] >= data_inicio) & (df["Date"] < data_ref)]
            df_test = df[df["AnoMes"] == mes].copy()
            df_test = df_test[df_test["Volume"] > 500000]

            if df_train.empty or df_test.empty:
                progresso.progress((i + 1) / total)
                continue

            if len(df_train) > 10000:
                df_train = df_train.sample(10000, random_state=42)

            X_train = df_train[features]
            y_train = df_train[target]
            X_test = df_test[features]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            modelo = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            modelo.fit(X_train_scaled, y_train)

            probs = modelo.predict_proba(X_test_scaled)
            df_test["Proba_Classe1"] = probs[:, 1]
            df_test = df_test[df_test["Proba_Classe1"] > 0.55]

            if df_test.empty:
                resultados.append({
                    "Data": data_ref,
                    "Retorno": 0,
                    "Acuracia": np.nan,
                    "Capital": capital
                })
                progresso.progress((i + 1) / total)
                continue

            top_ativos = df_test.sort_values("Proba_Classe1", ascending=False).drop_duplicates("Ticker").copy()
            top_ativos["Peso"] = 1 / len(top_ativos)
            top_ativos["Investimento"] = capital * top_ativos["Peso"]
            top_ativos["Valor Final"] = top_ativos["Investimento"] * (1 + top_ativos["RetFut20"])

            capital = top_ativos["Valor Final"].sum()
            retorno_mensal = (capital / capital_anterior) - 1
            capital_anterior = capital

            acc = accuracy_score(y_train, modelo.predict(X_train_scaled))

            resultados.append({
                "Data": data_ref,
                "Retorno": retorno_mensal,
                "Acuracia": acc,
                "Capital": capital
            })

            carteira_mes = top_ativos[["Ticker", "Proba_Classe1", "RetFut20", "Peso", "Investimento", "Valor Final"]].head(10).copy()
            carteira_mes["Data"] = data_ref
            carteiras.append(carteira_mes)

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

        if carteiras:
            df_carteiras = pd.concat(carteiras).reset_index(drop=True)
            df_carteiras["AnoMes"] = df_carteiras["Data"].dt.to_period("M").astype(str)
        else:
            st.warning("⚠️ Nenhuma carteira foi formada.")
            return

    st.success("✅ Simulação concluída!")

    st.subheader("📈 Rentabilidade Acumulada")
    st.line_chart(df_result["Rentabilidade Acumulada"])

    st.subheader("💰 Evolução do Capital")
    st.line_chart(df_result["Capital"])

    st.subheader("🎯 Acurácia do Modelo ao Longo do Tempo")
    st.line_chart(df_result["Acuracia"])

    st.subheader("📋 Composição da Carteira por Mês")
    meses_disponiveis = sorted(df_carteiras["AnoMes"].unique())
    mes_selecionado = st.selectbox("Selecione o mês:", meses_disponiveis)
    carteira_mes = df_carteiras[df_carteiras["AnoMes"] == mes_selecionado]
    st.dataframe(carteira_mes.drop(columns=["AnoMes", "Data"]).reset_index(drop=True))

    df_result[["Capital"]].to_csv(f"data/results/random_forest_capital.csv")

    with open("data/results/random_forest_indices.txt", "w") as f:
        f.write(f"{sharpe_ratio:.6f},{sortino_ratio:.6f}")

    # === SHAP Dinâmico ===
    st.subheader("🧠 SHAP – Explicabilidade para o mês selecionado")

    data_ref_shap = pd.to_datetime(mes_selecionado + "-01")
    data_inicio_shap = data_ref_shap - pd.DateOffset(months=12)

    df_treino_shap = df[(df["Date"] >= data_inicio_shap) & (df["Date"] < data_ref_shap)]
    df_treino_shap["Momentum"] = df_treino_shap["MA5"] - df_treino_shap["MA20"]
    df_treino_shap = df_treino_shap[df_treino_shap["Volume"] > 500000]

    X_train_shap = df_treino_shap[features]
    y_train_shap = df_treino_shap["Classe"]

    scaler_shap = StandardScaler()
    X_train_scaled_shap = scaler_shap.fit_transform(X_train_shap)

    modelo_shap = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    modelo_shap.fit(X_train_scaled_shap, y_train_shap)

    ativos_carteira = carteira_mes["Ticker"].unique()
    df_explicacao_mes = df[(df["Ticker"].isin(ativos_carteira)) & (df["AnoMes"] == mes_selecionado)].copy()
    df_explicacao_mes["Momentum"] = df_explicacao_mes["MA5"] - df_explicacao_mes["MA20"]

    X_exp_mes = df_explicacao_mes[features]
    X_exp_scaled_mes = scaler_shap.transform(X_exp_mes)
    X_exp_df = pd.DataFrame(X_exp_scaled_mes, columns=features)

    explainer_mes = shap.TreeExplainer(modelo_shap)
    shap_values_full = explainer_mes.shap_values(X_exp_scaled_mes)

    # Extrai classe positiva corretamente (3D output)
    if len(shap_values_full.shape) == 3:
        shap_values_mes = shap_values_full[:, :, 1]
    elif isinstance(shap_values_full, list):
        shap_values_mes = shap_values_full[1]
    else:
        shap_values_mes = shap_values_full

    # SHAP GLOBAL
    st.markdown("**📌 Importância Global das Variáveis**")
    assert shap_values_mes.shape == X_exp_df.shape, f"Incompatível: {shap_values_mes.shape} vs {X_exp_df.shape}"
    fig_global, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_mes, X_exp_df, show=False)
    st.pyplot(fig_global)

    # SHAP LOCAL
    try:
        idx_local_label = carteira_mes["Proba_Classe1"].idxmax()
        ticker_destaque = carteira_mes.loc[idx_local_label, "Ticker"]
        exemplo_local = df_explicacao_mes[df_explicacao_mes["Ticker"] == ticker_destaque].iloc[0]
        idx_pos_local = df_explicacao_mes[df_explicacao_mes["Ticker"] == ticker_destaque].index.get_loc(exemplo_local.name)

        st.markdown(f"**🔍 Explicação Local – {ticker_destaque}**")
        fig_local, ax = plt.subplots(figsize=(10, 5))
        expected_value = explainer_mes.expected_value
        if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1]  # classe positiva

        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values_mes[idx_pos_local],
            features=X_exp_df.iloc[idx_pos_local],
            feature_names=features
        )
        st.pyplot(fig_local)

    except Exception as e:
        st.warning(f"❌ Erro ao gerar explicação local: {e}")
