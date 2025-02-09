import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset de vacinações
vacinacoes = pd.read_csv("../data/vaccinations.csv")

# Converter a coluna 'date' para datetime
vacinacoes['date'] = pd.to_datetime(vacinacoes['date'])

# Filtrar apenas a localização "World"
vacinacoes_world = vacinacoes[vacinacoes['location'] == 'World']

# Garantir que 'daily_people_vaccinated' não tenha valores nulos
vacinacoes_world['daily_people_vaccinated'].fillna(0, inplace=True)

# Criar uma coluna "ano_mes"
vacinacoes_world.loc[:, 'ano_mes'] = vacinacoes_world['date'].dt.to_period('M')

# Agrupar os dados por mês e somar as primeiras doses diárias
primeiras_doses_por_mes = vacinacoes_world.groupby('ano_mes')['daily_people_vaccinated'].sum().reset_index()

# Calcular o somatório acumulado
primeiras_doses_por_mes['soma_acumulada'] = primeiras_doses_por_mes['daily_people_vaccinated'].cumsum()

# Converter 'ano_mes' para string para o gráfico
primeiras_doses_por_mes['ano_mes'] = primeiras_doses_por_mes['ano_mes'].astype(str)

# Verificar o valor máximo após a soma acumulada
max_soma_acumulada = primeiras_doses_por_mes['soma_acumulada'].max()

# **Definir a unidade de medida**
if max_soma_acumulada > 1_000_000:
    primeiras_doses_por_mes['soma_acumulada'] /= 1_000_000
    unidade = 'milhões'
elif max_soma_acumulada > 1_000:
    primeiras_doses_por_mes['soma_acumulada'] /= 1_000
    unidade = 'milhares'
else:
    unidade = 'unidades'

# Criar o gráfico de barras (histograma)
plt.figure(figsize=(12, 6))
plt.bar(primeiras_doses_por_mes['ano_mes'], primeiras_doses_por_mes['soma_acumulada'], edgecolor='black')

# Adicionar título e labels
plt.title(f'Somatório Acumulado de Primeiras Doses por Mês (World, em {unidade})')
plt.xlabel('Mês/Ano')
plt.ylabel(f'Somatório Acumulado de Primeiras Doses (em {unidade})')

# Melhorar a legibilidade do eixo x
plt.xticks(rotation=45, ha='right')

# Ajustar a escala do eixo Y
plt.ylim(0, primeiras_doses_por_mes['soma_acumulada'].max() * 1.1)

# Ajustar layout
plt.tight_layout()

# Exibir o gráfico antes de salvar
plt.show()

# **Salvar o gráfico corretamente**
plt.savefig('soma_acumulada_primeiras_doses_world_por_mes.png', dpi=300, bbox_inches='tight')