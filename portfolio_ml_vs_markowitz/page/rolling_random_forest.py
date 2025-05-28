import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import os

# Carregar dataset
df = pd.read_csv("data/processed/dataset_ml.csv", parse_dates=["Date"])
df = df.dropna(subset=["Date"])
df["AnoMes"] = df["Date"].dt.to_period("M")

# Definir variáveis
features = ["Ret5", "Ret10", "Ret20", "MA5", "MA20", "Vol20"]
target = "RetFut20"

# Parâmetros do modelo
n_estimators = 100
random_state = 42

# Inicializar listas
resultados = []
carteiras = []

# Meses a simular (2021 em diante)
meses = sorted(df[df["Date"] >= "2021-01-01"]["AnoMes"].unique())

print("Executando modelo Random Forest em janela rolling...")
for mes in tqdm(meses):
    data_ref = mes.to_timestamp()

    # Treino: dados até o mês anterior
    df_train = df[df["Date"] < data_ref]
    df_test = df[df["AnoMes"] == mes]

    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    df_test = df_test.copy()
    df_test["Predito"] = model.predict(X_test)

    # Avaliação
    mse = mean_squared_error(df_test[target], df_test["Predito"])
    r2 = r2_score(df_test[target], df_test["Predito"])

    # Seleção top 10 ativos
    top10 = df_test.sort_values("Predito", ascending=False).head(10)
    retorno_medio = top10[target].mean()

    resultados.append({
        "Data": data_ref,
        "Retorno": retorno_medio,
        "MSE": mse,
        "R2": r2
    })

    top10_result = top10[["Ticker", "Predito", target]].copy()
    top10_result["Data"] = data_ref
    carteiras.append(top10_result)

# Gerar dataframe final
df_result = pd.DataFrame(resultados)
df_result.set_index("Data", inplace=True)
df_result["Rentabilidade Acumulada"] = (1 + df_result["Retorno"]).cumprod() - 1

# Salvar resultados
os.makedirs("output", exist_ok=True)
df_result.to_csv("output/rentabilidade_rf.csv")
pd.concat(carteiras).to_parquet("output/carteiras_rf.parquet", index=False)

print("✅ Finalizado!")
print("Arquivos salvos em: output/rentabilidade_rf.csv e output/carteiras_rf.parquet")
