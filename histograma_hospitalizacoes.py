import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset de hospitalizações
hospitalizacoes = pd.read_csv("../data/covid-hospitalizations.csv")

# Converter a coluna 'date' para o tipo datetime
hospitalizacoes['date'] = pd.to_datetime(hospitalizacoes['date'])

# Extrair o ano e o mês da coluna 'date'
hospitalizacoes['ano_mes'] = hospitalizacoes['date'].dt.to_period('M')

# Agrupar os dados por mês e somar as hospitalizações
hospitalizacoes_por_mes = hospitalizacoes.groupby('ano_mes')['value'].sum().reset_index()

# Converter 'ano_mes' para string (para facilitar a visualização no gráfico)
hospitalizacoes_por_mes['ano_mes'] = hospitalizacoes_por_mes['ano_mes'].astype(str)

# Verificar se os valores são muito grandes (em milhões)
if hospitalizacoes_por_mes['value'].max() > 1_000_000:
    hospitalizacoes_por_mes['value'] = hospitalizacoes_por_mes['value'] / 1_000_000  # Converter para milhões
    unidade = 'milhões'
else:
    unidade = 'unidades'

# Criar o histograma
hospitalizacoes_por_mes.plot(kind='bar', x='ano_mes', y='value', legend=False, edgecolor='black')

# Adicionar título e labels
plt.title(f'Número de Hospitalizações por Mês')
plt.xlabel('Mês/Ano')
plt.ylabel(f'Número de Hospitalizações (em {unidade})')

# Melhorar a legibilidade do eixo x
plt.xticks(rotation=45, ha='right')

# Ajustar a escala do eixo Y
plt.ylim(0, hospitalizacoes_por_mes['value'].max() * 1.1)

# Exibir o gráfico
plt.tight_layout()
plt.show()

# Salvar o gráfico
plt.savefig('hospitalizacoes_por_mes.png')