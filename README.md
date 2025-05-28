
# 📈 Modelagem de Risco e Retorno em Portfólios de Ações com Aprendizado de Máquina

Este projeto corresponde à dissertação de mestrado apresentada à FUCAPE Business School, com o objetivo de investigar como algoritmos de aprendizado de máquina podem ser utilizados na construção de portfólios de ações, otimizando simultaneamente risco e retorno em ambientes de alta incerteza.

---

## 🧪 Objetivo do Projeto

Desenvolver e comparar estratégias de alocação de ativos baseadas em:
- Modelos tradicionais (Markowitz)
- Algoritmos de aprendizado de máquina (Random Forest, XGBoost, MLP e SVM)
- Métricas de desempenho ajustadas ao risco
- Técnicas de explicabilidade (XAI/SHAP)

---

## 📁 Estrutura do Projeto

```
portfolio_ml_vs_markowitz/
├── data/
│   ├── prices/            # Dados históricos dos ativos
│   ├── processed/         # Dataset processado para ML
│   └── results/           # Resultados das simulações
├── pages/
│   ├── treinamento_rf.py
│   ├── treinamento_xgb.py
│   ├── treinamento_mlp.py
│   └── ...
├── utils/
│   └── funcoes.py         # Funções auxiliares
├── streamlit_app.py       # Aplicação principal
└── requirements.txt       # Dependências do projeto
```

---

## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- Streamlit
- Pandas & NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

---

## ▶️ Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/wanderaf/ml_fucape.git
cd ml_fucape
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Inicie a aplicação:

```bash
streamlit run streamlit_app.py
```

---

## 📊 Resultados

Os modelos foram testados entre 2021 e 2024 utilizando janelas rolling de 12 meses e rebalanceamento mensal. As melhores performances foram obtidas com Random Forest e XGBoost. A explicabilidade foi abordada com SHAP, permitindo entender as variáveis mais influentes nas decisões de cada modelo.

---

## 📚 Referência Bibliográfica

A bibliografia completa com links DOI está descrita no corpo da dissertação.

---

## 🎓 Sobre

Este projeto faz parte da dissertação de mestrado profissional em Administração com ênfase em Finanças pela FUCAPE Business School.

Autor: Wanderson Batista  
Orientador: Prof. Walter Souto
