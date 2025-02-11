import pandas as pd
import matplotlib.pyplot as plt

# Carregar os datasets
hospitalizacoes = pd.read_csv("../data/covid-hospitalizations.csv")
vacinacoes = pd.read_csv("../data/vaccinations.csv")

# Converter colunas de data
hospitalizacoes['date'] = pd.to_datetime(hospitalizacoes['date'])
vacinacoes['date'] = pd.to_datetime(vacinacoes['date'])

# Lista de países com dados em ambos os datasets
paises_validos = set(hospitalizacoes['entity'].unique()).intersection(set(vacinacoes['location'].unique()))

# Inicializar DataFrames para armazenar as somas
soma_hospitalizacoes = pd.DataFrame()
soma_vacinacoes = pd.DataFrame()

# Somar dados de todos os países válidos
for pais in paises_validos:
    hospitalizacoes_pais = hospitalizacoes[(hospitalizacoes['entity'] == pais) &
                                           (hospitalizacoes['indicator'] == 'Daily hospital occupancy')].copy()
    vacinacoes_pais = vacinacoes[vacinacoes['location'] == pais].copy()

    if hospitalizacoes_pais.empty or vacinacoes_pais.empty:
        continue

    # Somar hospitalizações
    if soma_hospitalizacoes.empty:
        soma_hospitalizacoes = hospitalizacoes_pais[['date', 'value']].copy()
    else:
        soma_hospitalizacoes = pd.merge(soma_hospitalizacoes,
                                        hospitalizacoes_pais[['date', 'value']],
                                        on='date', how='outer', suffixes=('', '_extra'))
        soma_hospitalizacoes['value'] = soma_hospitalizacoes.filter(like='value').sum(axis=1)
        soma_hospitalizacoes = soma_hospitalizacoes[['date', 'value']]  # Manter apenas colunas principais

    # Calcular vacinações diárias
    vacinacoes_pais['daily_vaccinations'] = vacinacoes_pais['people_vaccinated'].diff().fillna(0)

    if soma_vacinacoes.empty:
        soma_vacinacoes = vacinacoes_pais[['date', 'daily_vaccinations']].copy()
    else:
        soma_vacinacoes = pd.merge(soma_vacinacoes,
                                   vacinacoes_pais[['date', 'daily_vaccinations']],
                                   on='date', how='outer', suffixes=('', '_extra'))
        soma_vacinacoes['daily_vaccinations'] = soma_vacinacoes.filter(like='daily_vaccinations').sum(axis=1)
        soma_vacinacoes = soma_vacinacoes[['date', 'daily_vaccinations']]  # Manter apenas colunas principais

# Acumular vacinações ao longo do tempo (convertendo para milhões)
soma_vacinacoes['people_vaccinated_cumulative'] = soma_vacinacoes['daily_vaccinations'].cumsum() / 1_000_000

# Combinar as somas por data
dados_combinados = pd.merge(soma_hospitalizacoes, soma_vacinacoes[['date', 'people_vaccinated_cumulative']],
                            on='date', how='inner')

# Remover valores nulos
dados_combinados.dropna(subset=['value', 'people_vaccinated_cumulative'], inplace=True)

# Criar o gráfico
if dados_combinados.empty:
    print("Não há dados válidos para plotar o gráfico.")
else:
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Linha para hospitalizações
    ax1.plot(dados_combinados['date'], dados_combinados['value'], color='red', label='Hospitalizações')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Hospitalizações (Soma)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Linha para vacinações acumuladas em milhões
    ax2 = ax1.twinx()
    ax2.plot(dados_combinados['date'], dados_combinados['people_vaccinated_cumulative'], color='blue',
             label='Pessoas Vacinadas (1ª Dose) - Milhões')
    ax2.set_ylabel('Pessoas Vacinadas (1ª Dose) - Milhões', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Adicionar título e legendas
    plt.title('Soma de Hospitalizações e Pessoas Vacinadas (1ª Dose) - Todos os Países Válidos')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    plt.tight_layout()
    plt.show()
