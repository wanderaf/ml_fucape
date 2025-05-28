import streamlit as st
import yfinance as yf
import pandas as pd
import os

def run():
    st.title("📥 Download dos Dados do IBOV")
    st.write("Esta página permite baixar os dados com base na composição do IBOV.")

    arquivo = st.file_uploader("Envie o arquivo composicao_ibov.xlsx", type=["xlsx"])

    if arquivo:
        df = pd.read_excel(arquivo)
        df.columns = ['Codigo', 'Quantidade', 'Participacao', 'Referencia', 'Ano', 'Mes']
        df['Referencia'] = pd.to_datetime(df['Referencia'], errors='coerce')
        df = df.dropna(subset=['Referencia'])

        tickers_unicos = df[['Codigo', 'Referencia']].drop_duplicates()
        tickers_unicos['Ticker_YF'] = tickers_unicos['Codigo'].apply(lambda x: x + ".SA")

        st.write(f"Total de ativos únicos identificados: {tickers_unicos.shape[0]}")

        pasta_precos = "data/prices"
        os.makedirs(pasta_precos, exist_ok=True)

        for _, row in tickers_unicos.iterrows():
            ticker = row['Ticker_YF']
            entrada = row['Referencia']
            caminho_arquivo = f"{pasta_precos}/{row['Codigo']}.csv"
            if not os.path.exists(caminho_arquivo):
                df_dados = yf.download(ticker, start="2014-01-01", end="2024-12-31")
                if not df_dados.empty:
                    df_dados.to_csv(caminho_arquivo)
        st.success("Download dos preços realizado com sucesso!")

        # Salvar mapa de entrada de ativos
        os.makedirs("data/mapeamentos", exist_ok=True)
        tickers_unicos.to_csv("data/mapeamentos/entrada_ativos.csv", index=False)
        st.info("Arquivo 'entrada_ativos.csv' salvo com data de entrada de cada ativo.")
