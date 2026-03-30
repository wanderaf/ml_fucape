import streamlit as st
import streamlit.components.v1 as components
from page import benchmark_ibov, download_dados, visualizacao_dados, simulador_markowitz, gerar_dataset_ml, treinamento_rf, treinamento_xgb_reg, treinamento_svm_reg, treinamento_mlb, comparativo_final

st.set_page_config(page_title="Simulador de Carteira", layout="wide")
#, initial_sidebar_state="collapsed"

# Oculta o menu e o rodapé padrão do Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
components.html(hide_streamlit_style, height=0)

pagina = st.sidebar.selectbox(
    "Escolha a página:",
    [
        "Download dos dados",
        "Visualização dos dados",
        "Simulação Markowitz",
        "Benchmark - IBOV",
        "Geração Dataset ML",
        "Treinamento Random Forest",
        "Treinamento XGBoost Regressor",
        "Treinamento SVM Regressor",
        "Treinamento MLP",
        "Comparativo Final"

    ]
)

if pagina == "Download dos dados":
    download_dados.run()
elif pagina == "Visualização dos dados":
    visualizacao_dados.run()
elif pagina == "Simulação Markowitz":
    simulador_markowitz.run()
elif pagina == "Benchmark - IBOV":
    benchmark_ibov.run()
elif pagina == "Geração Dataset ML":
    gerar_dataset_ml.run()
elif pagina == "Treinamento Random Forest":
    treinamento_rf.run()
elif pagina == "Treinamento XGBoost Regressor":
    treinamento_xgb_reg.run()
elif pagina == "Treinamento SVM Regressor":
    treinamento_svm_reg.run()
elif pagina == "Treinamento MLP":
    treinamento_mlb.run()
elif pagina == "Comparativo Final":
    comparativo_final.run()
