import streamlit as st
import pandas as pd
import numpy as np
import os

# fallback se precisar
try:
    import yfinance as yf
except Exception:
    yf = None

def run():
    st.title("📈 IBOV – Benchmark (Sharpe, Sortino, Drawdown e Overfit)")
    st.info("Calcula os índices de risco-retorno do IBOV e salva arquivos padronizados para a comparação com os modelos.")

    # =========================
    # Helpers
    # =========================
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
        if s.empty:
            return np.nan
        dd = (s / s.cummax()) - 1.0
        return float(dd.min()) * 100.0  # %

    def _load_ibov_prices(path="data/prices/^BVSP.csv", start="2014-01-01", end="2024-12-31"):
        """
        Tenta ler o ^BVSP.csv em diferentes formatos:
        1) CSV limpo com colunas 'Date','Close'
        2) CSV com multi-cabeçalho do Yahoo (Price/Close/...; Ticker; Date)
        Se falhar, baixa via yfinance (se disponível).
        Retorna DataFrame com colunas: ['Date','Close'].
        """
        if os.path.exists(path):
            # 1) CSV simples
            try:
                df = pd.read_csv(path)
                if "Date" in df.columns:
                    # achar a coluna Close
                    close_col = "Close" if "Close" in df.columns else None
                    if close_col is None:
                        cand = [c for c in df.columns if str(c).strip().lower() == "close"]
                        if cand:
                            close_col = cand[0]
                    if close_col is not None:
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        df = df.dropna(subset=["Date", close_col])[
                            ["Date", close_col]
                        ].sort_values("Date")
                        df = df.rename(columns={close_col: "Close"})
                        return df
            except Exception:
                pass

            # 2) Multi-cabeçalho Yahoo
            try:
                df_multi = pd.read_csv(path, header=[0, 1, 2])
                # localizar coluna Date
                date_col = None
                for col in df_multi.columns:
                    if isinstance(col, tuple) and any(str(x).strip().lower() == "date" for x in col):
                        date_col = col
                        break
                if date_col is None:
                    date_col = df_multi.columns[0]

                # localizar coluna Close
                close_col = None
                for col in df_multi.columns:
                    if isinstance(col, tuple):
                        lvls = [str(x).strip().lower() for x in col]
                        if "close" in lvls and "^bvsp" in lvls:
                            close_col = col
                            break
                if close_col is None:
                    for col in df_multi.columns:
                        if isinstance(col, tuple) and "close" in [str(x).strip().lower() for x in col]:
                            close_col = col
                            break

                df2 = df_multi[[date_col, close_col]].copy()
                df2.columns = ["Date", "Close"]
                df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
                df2 = df2.dropna(subset=["Date", "Close"]).sort_values("Date")
                return df2
            except Exception:
                pass

        # 3) Fallback: yfinance
        if yf is not None:
            try:
                st.warning("Não consegui interpretar o arquivo local de ^BVSP. Baixando via yfinance como fallback…")
                data = yf.download("^BVSP", start=start, end=end)
                data = data.reset_index()
                data = data.rename(columns={"Date": "Date", "Close": "Close"})
                data = data[["Date", "Close"]].dropna().sort_values("Date")
                return data
            except Exception as e:
                st.error(f"Falha ao baixar ^BVSP via yfinance: {e}")

        raise RuntimeError("Não foi possível carregar os preços do IBOV (^BVSP).")

    with st.spinner("🔄 Preparando dados do IBOV…"):
        # =========================
        # Leitura de preços
        # =========================
        df_ibov = _load_ibov_prices()
        if df_ibov.empty:
            st.error("Série do IBOV vazia.")
            return

        # Resample mensal (último dia útil do mês)
        s_close_m = df_ibov.set_index("Date")["Close"].resample("M").last().dropna()
        if len(s_close_m) < 3:
            st.error("Série mensal do IBOV muito curta para calcular métricas.")
            return

        df_monthly = pd.DataFrame({"Data": s_close_m.index, "Close": s_close_m.values}).reset_index(drop=True)
        df_monthly["Retorno"] = df_monthly["Close"].pct_change()

        # Curva de capital robusta (sem listas)
        start_capital = 100_000.0
        df_monthly["Capital"] = (1 + df_monthly["Retorno"].fillna(0)).cumprod() * start_capital

        # Corte treino/teste para overfit
        cutoff = pd.Timestamp("2020-12-31")
        df_train = df_monthly[df_monthly["Data"] <= cutoff].copy()
        df_test  = df_monthly[df_monthly["Data"] >  cutoff].copy()

        # Índices (treino/teste)
        sharpe_train  = _sharpe(df_train["Retorno"])  if not df_train.empty else np.nan
        sortino_train = _sortino(df_train["Retorno"]) if not df_train.empty else np.nan
        sharpe_test   = _sharpe(df_test["Retorno"])   if not df_test.empty  else np.nan
        sortino_test  = _sortino(df_test["Retorno"])  if not df_test.empty else np.nan

        overfit_ratio = np.nan
        if pd.notna(sharpe_train) and pd.notna(sharpe_test) and sharpe_test != 0:
            overfit_ratio = float(sharpe_train / sharpe_test)

        mdd_test = _max_drawdown_from_equity(df_test["Capital"]) if not df_test.empty else np.nan

        # =========================
        # Exibição – Métricas
        # =========================
        st.subheader("📊 Índices de Desempenho (IBOV)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔹 Sharpe (OOS)", f"{sharpe_test:.4f}" if pd.notna(sharpe_test) else "–")
        c2.metric("🔻 Sortino (OOS)", f"{sortino_test:.4f}" if pd.notna(sortino_test) else "–")
        c3.metric("📉 Max Drawdown (OOS)", f"{mdd_test:.2f}%" if pd.notna(mdd_test) else "–")
        c4.metric("🧪 Overfit (Sharpe T/Te)", f"{overfit_ratio:.3f}" if pd.notna(overfit_ratio) else "–")

        # =========================
        # Gráficos
        # =========================
        st.subheader("📈 Rentabilidade Acumulada (mensal)")
        rent_acum = (1 + df_monthly["Retorno"].fillna(0)).cumprod()
        st.line_chart(pd.DataFrame({"Rent. Acumulada": rent_acum.values}, index=df_monthly["Data"]))

        st.subheader("💰 Evolução do Capital")
        st.line_chart(pd.DataFrame({"Capital": df_monthly["Capital"].values}, index=df_monthly["Data"]))

        # =========================
        # Salvamento para Comparativo
        # =========================
        os.makedirs("data/results", exist_ok=True)

        # capital FULL / TRAIN / TEST
        df_monthly[["Data", "Capital"]].to_csv("data/results/ibov_capital.csv", index=False)
        if not df_train.empty:
            df_train[["Data", "Capital"]].to_csv("data/results/ibov_capital_train.csv", index=False)
        if not df_test.empty:
            df_test[["Data", "Capital"]].to_csv("data/results/ibov_capital_test.csv", index=False)

        # índices
        with open("data/results/ibov_indices.txt", "w") as f:
            f.write(f"{sharpe_test},{sortino_test}")
        with open("data/results/ibov_indices_train.txt", "w") as f:
            f.write(f"{sharpe_train},{sortino_train}")
        with open("data/results/ibov_indices_test.txt", "w") as f:
            f.write(f"{sharpe_test},{sortino_test}")
        with open("data/results/ibov_overfit_ratio.txt", "w") as f:
            f.write(f"{overfit_ratio}")
        with open("data/results/ibov_max_drawdown_test.txt", "w") as f:
            f.write(f"{mdd_test}")

    st.success("✅ IBOV processado e salvo para comparação.")
