import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Carregar os datasets
hospitalizacoes = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/hospitalizations/covid-hospitalizations.csv")
vacinacoes = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/vaccinations/vaccinations.csv")

# Verificar as colunas disponíveis nos datasets
print("Colunas em hospitalizacoes:", hospitalizacoes.columns.tolist())
print("Colunas em vacinacoes:", vacinacoes.columns.tolist())

# Converter colunas de data para datetime
hospitalizacoes['date'] = pd.to_datetime(hospitalizacoes['date'])
vacinacoes['date'] = pd.to_datetime(vacinacoes['date'])

# Agrupar hospitalizações por mês (assumindo que já são dados globais)
hospitalizacoes['ano_mes'] = hospitalizacoes['date'].dt.to_period('M')
hospitalizacoes_por_mes = hospitalizacoes.groupby('ano_mes')['value'].sum().reset_index()

# Agrupar vacinações por mês (usando o total acumulado de pessoas vacinadas)
vacinacoes['ano_mes'] = vacinacoes['date'].dt.to_period('M')
vacinacoes_por_mes = vacinacoes.groupby('ano_mes')['people_vaccinated'].max().reset_index()

# Unir os dados de hospitalizações e vacinações
dados_combinados = pd.merge(hospitalizacoes_por_mes, vacinacoes_por_mes, on='ano_mes', how='inner')

# Calcular a taxa de hospitalização (hospitalizações por pessoa vacinada)
dados_combinados['taxa_hospitalizacao'] = dados_combinados['value'] / dados_combinados['people_vaccinated']

# Remover valores infinitos ou NaN
dados_combinados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados_combinados.dropna(inplace=True)

# Ajustar o modelo de Poisson
X = dados_combinados['people_vaccinated']  # Variável independente (pessoas vacinadas acumuladas)
y = dados_combinados['value']  # Variável dependente (hospitalizações)

# Adicionar uma constante ao modelo (intercepto)
X = sm.add_constant(X)

# Ajustar o modelo
modelo_poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Exibir resumo do modelo
print(modelo_poisson.summary())

# Prever probabilidades de hospitalização
dados_combinados['predito'] = modelo_poisson.predict(X)

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.scatter(dados_combinados['people_vaccinated'], dados_combinados['value'], label='Dados Observados')
plt.plot(dados_combinados['people_vaccinated'], dados_combinados['predito'], color='red', label='Modelo de Poisson')
plt.title('Modelo de Poisson: Hospitalizações vs. Pessoas Vacinadas (Acumulado)')
plt.xlabel('Pessoas Vacinadas (Acumulado)')
plt.ylabel('Hospitalizações')
plt.legend()
plt.grid(True)
plt.show()