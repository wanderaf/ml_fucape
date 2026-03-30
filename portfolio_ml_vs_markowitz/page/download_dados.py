import streamlit as st
import yfinance as yf
import pandas as pd
import os

def run():
    st.title("📥 Download dos Dados do IBOV")
    st.write("Esta página permite baixar os dados com base na composição do IBOV.")

    # Garantir que pasta existe antes de qualquer download
    pasta_precos = "data/prices"
    os.makedirs(pasta_precos, exist_ok=True)

    # 1. Baixar sempre o IBOV (^BVSP)
    st.subheader("Índice IBOV ( ^BVSP )")
    caminho_ibov = f"{pasta_precos}/^BVSP.csv"

    if not os.path.exists(caminho_ibov):
        st.write("Baixando histórico do IBOV (^BVSP)...")
        df_ibov = yf.download("^BVSP", start="2014-01-01", end="2024-12-31")
        if not df_ibov.empty:
            df_ibov.to_csv(caminho_ibov)
            st.success("Dados do IBOV salvos em data/prices/^BVSP.csv")
        else:
            st.error("Falha ao baixar os dados do IBOV.")
    else:
        st.write("✅ IBOV já existente em disco. Pulando download.")

    st.divider()

    # 2. Upload e processamento do arquivo de composição
    arquivo = st.file_uploader("Envie o arquivo composicao_ibov.xlsx", type=["xlsx"])

    if arquivo:
        df = pd.read_excel(arquivo)

        # padroniza colunas
        df.columns = ['Codigo', 'Quantidade', 'Participacao', 'Referencia', 'Ano', 'Mes']
        df['Referencia'] = pd.to_datetime(df['Referencia'], errors='coerce')
        df = df.dropna(subset=['Referencia'])

        # pega apenas código e data de entrada, únicos
        tickers_unicos = df[['Codigo', 'Referencia']].drop_duplicates()
        tickers_unicos['Ticker_YF'] = tickers_unicos['Codigo'].apply(lambda x: x + ".SA")

        st.write(f"Total de ativos únicos identificados: {tickers_unicos.shape[0]}")

        # já garantimos pasta_precos acima, então só usar

        for _, row in tickers_unicos.iterrows():
            ticker = row['Ticker_YF']
            caminho_arquivo = f"{pasta_precos}/{row['Codigo']}.csv"

            if not os.path.exists(caminho_arquivo):
                st.write(f"Baixando {ticker} ...")
                df_dados = yf.download(ticker, start="2014-01-01", end="2024-12-31")
                if not df_dados.empty:
                    df_dados.to_csv(caminho_arquivo)
                else:
                    st.warning(f"Sem dados disponíveis para {ticker}.")

        st.success("Download dos preços dos ativos realizado com sucesso!")

        # 3. Salvar mapa de entrada de ativos
        os.makedirs("data/mapeamentos", exist_ok=True)
        tickers_unicos.to_csv("data/mapeamentos/entrada_ativos.csv", index=False)
        st.info("Arquivo 'entrada_ativos.csv' salvo com data de entrada de cada ativo.")
