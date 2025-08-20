import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Lendo o dataset
df = pd.read_csv("ibov.csv")

# Detectando automaticamente a primeira coluna numérica
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) == 0:
    raise ValueError("⚠ Nenhuma coluna numérica encontrada no CSV!")
forecast_col = numeric_cols[0]
print(f"📊 Coluna escolhida para previsão: {forecast_col}")

# Função para preparar os dados
def prepare_data(df, forecast_col, forecast_out, test_size):
    # Criando a coluna alvo (y), deslocando as últimas linhas para prever
    y = df[forecast_col].shift(-forecast_out)

    # Criando a matriz de recursos (X)
    X = np.array(df[[forecast_col]])
    X = preprocessing.scale(X)

    # Separando a parte para previsão futura
    X_lately = X[-forecast_out:]

    # Removendo as últimas linhas para manter consistência
    X = X[:-forecast_out]
    y = y[:-forecast_out]

    # Separando treino e teste
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    return X_train, X_test, Y_train, Y_test, X_lately

# Configurações
forecast_out = 5   # número de passos à frente
test_size = 0.2    # 20% para teste

# Preparando os dados
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)

# Criando e treinando o modelo
model = LinearRegression()
model.fit(X_train, Y_train)

# Avaliando o modelo
accuracy = model.score(X_test, Y_test)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Fazendo previsões futuras
forecast = model.predict(X_lately)
print("🔮 Previsões futuras:", forecast)
