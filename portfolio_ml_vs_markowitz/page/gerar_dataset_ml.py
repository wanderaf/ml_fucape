import streamlit as st
import pandas as pd
import numpy as np
import os

def run():
    st.title("🛠️ Geração do Dataset para Machine Learning")

    if st.button("Gerar Dataset"):
        dataset = gerar_dataset_ml()
        st.success("✅ Dataset gerado com sucesso!")
        st.subheader("📄 Visualização do Dataset")
        st.dataframe(dataset)


def gerar_dataset_ml(pasta_precos='data/prices', janela_target=20):
    arquivos = [f for f in os.listdir(pasta_precos) if f.endswith('.csv')]
    lista_dfs = []

    for arquivo in arquivos:
        ticker = arquivo.replace('.csv', '')
        caminho = os.path.join(pasta_precos, arquivo)

        df = pd.read_csv(caminho, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close", "Volume"])
        df = df[["Date", "Close", "Volume"]]
        df.set_index("Date", inplace=True)

        df_feat = pd.DataFrame(index=df.index)
        df_feat["Ret5"] = df["Close"].pct_change(5)
        df_feat["Ret10"] = df["Close"].pct_change(10)
        df_feat["Ret20"] = df["Close"].pct_change(20)
        df_feat["MA5"] = df["Close"].rolling(5).mean()
        df_feat["MA20"] = df["Close"].rolling(20).mean()
        df_feat["Vol20"] = df["Close"].pct_change().rolling(20).std()
        df_feat["RetFut20"] = df["Close"].shift(-janela_target) / df["Close"] - 1
        df_feat["Volume"] = df["Volume"].rolling(5).mean()  # suavização da liquidez
        df_feat["Ticker"] = ticker

        df_feat = df_feat.reset_index()
        df_feat = df_feat.dropna()

        lista_dfs.append(df_feat)

    dataset_final = pd.concat(lista_dfs, ignore_index=True)

    os.makedirs('data/processed', exist_ok=True)
    dataset_final.to_csv('data/processed/dataset_ml.csv', index=False)

    return dataset_final
