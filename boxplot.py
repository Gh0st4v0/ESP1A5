import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do link
url = "https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/hospitalizations/covid-hospitalizations.csv"
dados = pd.read_csv(url)

# Converter a coluna 'date' para o tipo datetime
dados['date'] = pd.to_datetime(dados['date'])

# Extrair o mês e o ano da coluna 'date'
dados['month_year'] = dados['date'].dt.to_period('M')

# Agrupar os dados por mês e calcular a soma das hospitalizações
hospitalizados_por_mes = dados.groupby('month_year')['value'].sum().reset_index()

# Converter o número de hospitalizações para milhões
hospitalizados_por_mes['value_millions'] = hospitalizados_por_mes['value'] / 1_000_000

# Criar o boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(hospitalizados_por_mes['value_millions'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Distribuição do Número de Hospitalizações por COVID-19 (em milhões)')
plt.ylabel('Número de Hospitalizações (em milhões)')
plt.xticks([1], ['Hospitalizações'])  # Adiciona um rótulo no eixo x
plt.grid(True)
plt.tight_layout()
plt.show()