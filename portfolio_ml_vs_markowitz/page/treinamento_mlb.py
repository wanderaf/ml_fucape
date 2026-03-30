import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import os

def run():
    st.title("🧠 MLP – Classificação Binária com Rolling Window de 12 Meses")
    st.info("O modelo prevê se o retorno será positivo (classe 1) ou não (classe 0), usando apenas os últimos 12 meses para treinar.")

    # ---------- utilitários métricas ----------
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

    # reproduz a mesma lógica do MLP para qualquer lista de meses (treino/teste)
    def _build_curve_mlp(df_all, meses_periods, features, prob_thresh=0.55, start_capital=100000.0):
        rows = []
        capital = start_capital
        capital_prev = capital

        for mes in meses_periods:
            data_ref = mes.to_timestamp()
            data_inicio = data_ref - pd.DateOffset(months=12)

            df_train = df_all[(df_all["Date"] >= data_inicio) & (df_all["Date"] < data_ref)]
            df_test  = df_all[df_all["AnoMes"] == mes].copy()
            df_test  = df_test[df_test["Volume"] > 500000]

            if df_train.empty or df_test.empty:
                rows.append({"Data": data_ref, "Retorno": 0.0, "Capital": capital})
                continue

            if len(df_train) > 10000:
                df_train = df_train.sample(10000, random_state=42)

            X_train = df_train[features]
            y_train = df_train["Classe"]
            X_test  = df_test[features]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            modelo = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
                early_stopping=True
            )
            modelo.fit(X_train_scaled, y_train)

            probs = modelo.predict_proba(X_test_scaled)
            df_test["Proba_Classe1"] = probs[:, 1]

            df_test = df_test[df_test["Proba_Classe1"] > prob_thresh]

            if df_test.empty:
                retorno_mensal = 0.0
            else:
                top_ativos = df_test.sort_values("Proba_Classe1", ascending=False)\
                                    .drop_duplicates("Ticker").head(10).copy()
                top_ativos["Peso"] = 1 / len(top_ativos)
                top_ativos["Investimento"] = capital * top_ativos["Peso"]
                top_ativos["Valor Final"] = top_ativos["Investimento"] * (1 + top_ativos["RetFut20"])
                capital = top_ativos["Valor Final"].sum()
                retorno_mensal = (capital / capital_prev) - 1

            capital_prev = capital
            rows.append({"Data": data_ref, "Retorno": retorno_mensal, "Capital": capital})

        if not rows:
            return pd.DataFrame(columns=["Data", "Retorno", "Capital"])
        return pd.DataFrame(rows).sort_values("Data")

    with st.spinner("🔄 Executando simulação..."):

        # ---------- dados e features ----------
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

        # ---------- loop OOS (original) ----------
        for i, mes in enumerate(meses_rolling):
            data_ref = mes.to_timestamp()
            data_inicio = data_ref - pd.DateOffset(months=12)

            df_train = df[(df["Date"] >= data_inicio) & (df["Date"] < data_ref)]
            df_test  = df[df["AnoMes"] == mes].copy()
            df_test  = df_test[df_test["Volume"] > 500000]

            if df_train.empty or df_test.empty:
                progresso.progress((i + 1) / total)
                continue

            if len(df_train) > 10000:
                df_train = df_train.sample(10000, random_state=42)

            X_train = df_train[features]
            y_train = df_train[target]
            X_test  = df_test[features]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            modelo = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
                early_stopping=True
            )
            modelo.fit(X_train_scaled, y_train)

            probs = modelo.predict_proba(X_test_scaled)
            df_test["Proba_Classe1"] = probs[:, 1]

            # filtro de confiança
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

            top_ativos = df_test.sort_values("Proba_Classe1", ascending=False)\
                                .drop_duplicates("Ticker").head(10).copy()
            top_ativos["Peso"] = 1 / len(top_ativos)
            top_ativos["Investimento"] = capital * top_ativos["Peso"]
            top_ativos["Valor Final"] = top_ativos["Investimento"] * (1 + top_ativos["RetFut20"])

            capital = top_ativos["Valor Final"].sum()
            retorno_mensal = (capital / capital_anterior) - 1
            capital_anterior = capital

            y_pred_train = modelo.predict(X_train_scaled)
            acc = accuracy_score(y_train, y_pred_train)

            resultados.append({
                "Data": data_ref,
                "Retorno": retorno_mensal,
                "Acuracia": acc,
                "Capital": capital
            })

            carteira_mes = top_ativos[["Ticker", "Proba_Classe1", "RetFut20", "Peso", "Investimento", "Valor Final"]].copy()
            carteira_mes["Data"] = data_ref
            carteiras.append(carteira_mes)

            progresso.progress((i + 1) / total)

        df_result = pd.DataFrame(resultados).set_index("Data")
        df_result["Rentabilidade Acumulada"] = (1 + df_result["Retorno"]).cumprod()

        # ---------- Índices OOS + NOVOS (MDD/Overfit) ----------
        st.subheader("📊 Índices de Desempenho")

        retorno_medio = df_result["Retorno"].mean()
        desvio_padrao = df_result["Retorno"].std()
        sharpe_ratio_oos = retorno_medio / desvio_padrao if desvio_padrao != 0 else np.nan

        retornos_negativos = df_result["Retorno"][df_result["Retorno"] < 0]
        desvio_negativo = retornos_negativos.std()
        sortino_ratio_oos = retorno_medio / desvio_negativo if desvio_negativo != 0 else np.nan

        # construir curvas treino/teste para Overfit e Drawdown
        cutoff = pd.to_datetime("2020-12-31")
        meses_train = sorted(df[df["Date"] <= cutoff]["AnoMes"].unique())
        meses_test  = sorted(df[df["Date"] >  cutoff]["AnoMes"].unique())

        df_train_curve = _build_curve_mlp(df, meses_train, features)
        df_test_curve  = _build_curve_mlp(df, meses_test,  features)

        sharpe_tr  = _sharpe(df_train_curve["Retorno"]) if not df_train_curve.empty else np.nan
        sortino_tr = _sortino(df_train_curve["Retorno"]) if not df_train_curve.empty else np.nan
        sharpe_te  = _sharpe(df_test_curve["Retorno"])  if not df_test_curve.empty  else np.nan
        sortino_te = _sortino(df_test_curve["Retorno"]) if not df_test_curve.empty else np.nan

        overfit_ratio = np.nan
        if pd.notna(sharpe_tr) and pd.notna(sharpe_te) and sharpe_te != 0:
            overfit_ratio = float(sharpe_tr / sharpe_te)

        mdd_tr = _max_drawdown_from_equity(df_train_curve["Capital"]) if not df_train_curve.empty else np.nan
        mdd_te = _max_drawdown_from_equity(df_test_curve["Capital"])  if not df_test_curve.empty  else np.nan

        # quatro indicadores lado a lado
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔹 Sharpe (OOS)", f"{sharpe_ratio_oos:.4f}" if pd.notna(sharpe_ratio_oos) else "–")
        c2.metric("🔻 Sortino (OOS)", f"{sortino_ratio_oos:.4f}" if pd.notna(sortino_ratio_oos) else "–")
        c3.metric("📉 Max Drawdown (OOS)", f"{mdd_te:.2f}%" if pd.notna(mdd_te) else "–")
        c4.metric("🧪 Overfit Ratio (Sharpe T/Te)", f"{overfit_ratio:.3f}" if pd.notna(overfit_ratio) else "–")

        # preparar carteiras para as visualizações
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

    # ---------- SALVAMENTO PARA PÁGINA COMPARATIVA ----------
    os.makedirs("data/results", exist_ok=True)

    # capital OOS
    df_result.reset_index()[["Data", "Capital"]].to_csv("data/results/mlp_capital.csv", index=False)

    # capital treino/teste p/ overfit
    if 'df_train_curve' in locals() and not df_train_curve.empty:
        df_train_curve[["Data", "Capital"]].to_csv("data/results/mlp_capital_train.csv", index=False)
    if 'df_test_curve' in locals() and not df_test_curve.empty:
        df_test_curve[["Data", "Capital"]].to_csv("data/results/mlp_capital_test.csv", index=False)

    # índices (usa TESTE/OOS para o principal)
    with open("data/results/mlp_indices.txt", "w") as f:
        f.write(f"{sharpe_te if pd.notna(sharpe_te) else np.nan},{sortino_te if pd.notna(sortino_te) else np.nan}")
    with open("data/results/mlp_indices_train.txt", "w") as f:
        f.write(f"{sharpe_tr if pd.notna(sharpe_tr) else np.nan},{sortino_tr if pd.notna(sortino_tr) else np.nan}")
    with open("data/results/mlp_indices_test.txt", "w") as f:
        f.write(f"{sharpe_te if pd.notna(sharpe_te) else np.nan},{sortino_te if pd.notna(sortino_te) else np.nan}")

    with open("data/results/mlp_overfit_ratio.txt", "w") as f:
        f.write(f"{overfit_ratio}")
    with open("data/results/mlp_max_drawdown_test.txt", "w") as f:
        f.write(f"{mdd_te}")

    # ---------- SHAP Dinâmico ----------
    st.subheader("🧠 SHAP – Explicabilidade para o mês selecionado")

    data_ref_shap = pd.to_datetime(mes_selecionado + "-01")
    data_inicio_shap = data_ref_shap - pd.DateOffset(months=12)

    df_treino_shap = df[(df["Date"] >= data_inicio_shap) & (df["Date"] < data_ref_shap)].copy()
    df_treino_shap["Momentum"] = df_treino_shap["MA5"] - df_treino_shap["MA20"]
    df_treino_shap = df_treino_shap[df_treino_shap["Volume"] > 500000]

    X_train_shap = df_treino_shap[features]
    y_train_shap = df_treino_shap["Classe"]

    scaler_shap = StandardScaler()
    X_train_scaled_shap = scaler_shap.fit_transform(X_train_shap)

    modelo_shap = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=42,
        early_stopping=True
    )
    modelo_shap.fit(X_train_scaled_shap, y_train_shap)

    ativos_carteira = carteira_mes["Ticker"].unique()
    df_explicacao_mes = df[
        (df["Ticker"].isin(ativos_carteira)) & (df["AnoMes"] == mes_selecionado)
    ].copy()
    df_explicacao_mes["Momentum"] = df_explicacao_mes["MA5"] - df_explicacao_mes["MA20"]

    X_exp_mes = df_explicacao_mes[features]
    X_exp_scaled_mes = scaler_shap.transform(X_exp_mes)
    X_exp_df = pd.DataFrame(X_exp_scaled_mes, columns=features)

    # background para KernelExplainer
    background = shap.utils.sample(X_train_scaled_shap, 100, random_state=42)
    explainer_mes = shap.KernelExplainer(modelo_shap.predict_proba, background)
    shap_values_all = explainer_mes.shap_values(X_exp_scaled_mes)

    if isinstance(shap_values_all, list):
        shap_values_mes = shap_values_all[1]
        expected_value = explainer_mes.expected_value[1]
    elif hasattr(shap_values_all, "shape") and len(shap_values_all.shape) == 3:
        shap_values_mes = shap_values_all[:, :, 1]
        expected_value = explainer_mes.expected_value[1]
    else:
        shap_values_mes = shap_values_all
        expected_value = explainer_mes.expected_value

    assert shap_values_mes.shape == X_exp_df.shape, f"SHAP shape incompatível: {shap_values_mes.shape} vs {X_exp_df.shape}"

    st.markdown("**📌 Importância Global das Variáveis**")
    fig_global, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_mes, X_exp_df, feature_names=features, show=False)
    st.pyplot(fig_global)

    # SHAP LOCAL
    try:
        idx_local_label = carteira_mes["Proba_Classe1"].idxmax()
        ticker_destaque = carteira_mes.loc[idx_local_label, "Ticker"]
        exemplo_local = df_explicacao_mes[df_explicacao_mes["Ticker"] == ticker_destaque].iloc[0]
        idx_pos_local = df_explicacao_mes[df_explicacao_mes["Ticker"] == ticker_destaque].index.get_loc(exemplo_local.name)

        st.markdown(f"**🔍 Explicação Local – {ticker_destaque}**")
        fig_local, _ = plt.subplots(figsize=(10, 5))
        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values_mes[idx_pos_local],
            features=X_exp_df.iloc[idx_pos_local],
            feature_names=features
        )
        st.pyplot(fig_local)

    except Exception as e:
        st.warning(f"❌ Erro ao gerar explicação local: {e}")
