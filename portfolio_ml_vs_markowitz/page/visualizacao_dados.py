import streamlit as st
import pandas as pd
import os

def run():
    st.title("🔎 Visualização dos Dados")
    st.write("Explore os dados históricos de ações baixados previamente.")

    pasta = "data/prices"
    arquivos = [f for f in os.listdir(pasta) if f.endswith(".csv")]

    arquivo_selecionado = st.selectbox("Selecione o arquivo para visualizar:", arquivos)

    if arquivo_selecionado:
        caminho = os.path.join(pasta, arquivo_selecionado)
        df = pd.read_csv(caminho, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)

        # Formatando os valores
        df_formatado = df.copy()
        df_formatado.index = df_formatado.index.strftime("%d/%m/%Y")
        for col in df_formatado.columns:
            df_formatado[col] = df_formatado[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

        st.subheader("📄 Todos os dados do arquivo")
        st.dataframe(df_formatado, use_container_width=True)

        st.subheader("📈 Evolução de preços (Close)")
        st.line_chart(df["Close"])
