# Bibliotecas
import streamlit as st                  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

#Teste Prophet
from prophet import Prophet

#Teste ARIMA:
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

#Para machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


##Importando banco de dados
#df = pd.read_csv(r"C:\Users\gih\Downloads\postechchall02\Dados Históricos - Ibovespa Fase 02.csv", sep=",")
#df = df.drop(columns=['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%'])    
#df.rename(columns={'Data': "Date", 'Último': 'Close'}, inplace=True)    

#Importando banco de dados com yfinance
import yfinance as yf
symbol = '^BVSP'
start_date = '2014-01-01'
end_date = '2024-03-07'
df = yf.download(symbol, start=start_date, end=end_date)

#Conversão da data para formato datetime
df = df.reset_index('Date')
df = df.sort_values(by="Date", ascending=True)
df['Date'] = pd.to_datetime(df['Date'], format="%d.%m.%Y") 



# PROPHET #
df = yf.download(symbol, start=start_date, end=end_date)
df = df.reset_index('Date')
df['Date'] = pd.to_datetime(df['Date'])
df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'], inplace=True)
df[['ds','y']] = df[['Date','Close']]
df.head()

train_data = df.sample(frac=0.8, random_state=0)
test_data = df.drop(train_data.index)
print(f'training data size : {train_data.shape}')
print(f'testing data size : {test_data.shape}')

modelo = Prophet(daily_seasonality=True)
modelo.fit(train_data)
futuro = modelo.make_future_dataframe(periods=30, freq='D')

forecast = modelo.predict(futuro)



##### INÍCIO DO DASHBOARD ##### 
st.set_page_config(page_title="Tech Challenge 02 - Gio Queiroz", page_icon=":chart_with_upwards_trend:", layout="wide", initial_sidebar_state="auto", menu_items=None)

# CABEÇALHO
st.markdown(
        """
    <style>
    .title-style {
        text-align: left;
        font-weight: bold;
        color: #F58a4c;
        font-size: 40px;
    }

    .subtitle-style {
        text-align: left;
        font-weight: bold;
        color: #4E41A6;
        font-size: 40px;
    }

    .about-style {
        font-weight: bold;
        color: #d7e46f;
    }

     .about2-style {
        font-weight: normal;
        color: #ffffff;
    }

    .about3-style {
        font-weight: bold;
        color: #d7e46f;
    }

     .about4-style {
        font-weight: bold;
        color: #8899ff;
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: normal;
    }

    .button:hover {
        font-size: 16px;
        font-weight: normal;
        color: red;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }

    .tab1-title {
        font-weight: bold;
        font-size: 30px;
        color: #F58a4c;
    }

    [data-testid="stMetric"] {
        background-color: #e8e8e8;
        border-radius: 10px;
        text-align: left;
        font-weight: bold;
        color: #962450;
        padding: 5% 5% 5% 5%;
        font-size: 40px;
        border-style: groove;
        border-width: 1px;
        border-color: #962450;
    }
}
    </style>
    """,
        unsafe_allow_html=True,
    )
st.markdown('<p class="title-style">PredVespa: <h n class="subtitle-style">Modelo Preditivo</h></p>', unsafe_allow_html=True)
st.write("Bem-vindo ao modelo preditivo de fechamento do IBOVESPA.")

st.markdown(':chart:<h class="about4-style"> Com a análise de dados do mercado financeiro e utilizando técnicas avançadas de machine learning, o modelo busca prever com precisão as flutuações do Ibovespa, sendo uma ferramenta poderosa para investidores e instituições financeiras.</h>', unsafe_allow_html=True)
st.markdown(':bar_chart:<h class="about4-style"> O IBOVESPA é o principal índice de desempenho das ações negociadas na Bolsa de Valores brasileira, servindo como referência essencial para o mercado financeiro nacional.', unsafe_allow_html=True)
st.markdown(':male-office-worker:<h class="about-style"> Sobre o time responsável: <h n class="about2-style">A equipe da PredVespa é especialista em Data Science e Machine Learning, proporcionando uma abordagem analítica e precisa para o projeto.</h>', unsafe_allow_html=True)

st.markdown('<p class="about-style">Propósito:</p>', unsafe_allow_html=True)
st.markdown("- Desenvolver um modelo preditivo para desvendar os padrões do fechamento diário do IBOVESPA.")
st.markdown("- Atingir uma acurácia superior a 70%, visando assegurar resultados robustos e confiáveis para nossos stakeholders.")

st.markdown('<p class="about-style">Metodologia</p>', unsafe_allow_html=True)
st.markdown("- Optamos por uma abordagem em séries temporais, considerando a complexidade dos dados e a necessidade de previsões diárias confiáveis. Utilizamos Python e suas bibliotecas especializadas, explorando métodos como média móvel, ARIMA e Prophet e a análise do melhor desempenho para garantir análises precisas dos padrões do mercado.")

st.markdown('<p class="about3-style">Fonte de Dados: <h class="about2-style"><a href="https://br.investing.com/indices/bovespa-historical-data" class="about2-style">IBOVESPA (IBOV) Historical Data - Investing</a>. <p class="about3-style">Período: 07/03/2004 a 07/03/2024.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([":bar_chart: Análise Exploratória (EDA)", ":clipboard: Métodos", ":chart: Resultados e Conclusões"])
tab1.write()
tab2.write()
tab3.write()
with tab1:
    st.markdown('<p class="tab1-title">Análise Exploratória dos Dados (EDA)</p>', unsafe_allow_html=True)
    



## ANÁLISE EXPLORATÓRIA DOS DADOS ##
#Visualizando DS Fechamento
    st.write("  Iniciamos o nosso projeto com a etapa vital da Análise Exploratória dos Dados, fundamental para a construção do PredVespa. Aqui, imergimos nos dados coletados e na história econômica do país, realizando passos essenciais que nos conduzirão a uma compreensão mais profunda do comportamento do IBOVESPA ao longo do tempo.")    



    coluna4, coluna5 = st.columns(2, gap="large")
    with coluna4:
        st.write("O gráfico de linha abaixo está sendo utilizado para exibir a flutuação diária do fechamento do IBOVESPA nos últimos 10 anos. O eixo x representa a série temporal, enquanto o eixo y indica os valores de fechamento.")
        ax = px.line(df, x="Date", y="Close", color_discrete_sequence=['#d7e46f'], width=800, height=500, range_y=[0, 150000], title="Valores Diários de Fechamento (IBOVESPA) - Série Temporal")
        ax.update_xaxes(dtick=1)
        ax.update_layout(
        yaxis=dict(
        title="Valor de Fechamento",
        titlefont=dict(size=16),
        tickfont=dict(size=14),
        showgrid=True,
        gridcolor="#c2c0c0",
        griddash="dash",
        title_standoff=30 # The higher the value, the farther away it is displayed
        ),
        xaxis = dict(
        title="Ano",
        tickmode = 'array',
        tickvals = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        titlefont=dict(size=16),
        tickfont=dict(size=14),
        tickangle=-45,
        title_standoff=20 # The higher the value, the farther away it is displayed
        ))
        ax.add_hline(y="Close", line_color="black", line_width=1, line_dash="dot", label=dict(text="                                                                  Valor Médio: US$ 7,51 MM", textposition="middle", font=dict(size=14, color="black"))
        )
        st.plotly_chart(ax, use_container_width=True)

    with coluna5:        
        st.markdown(':triangular_flag_on_post: <h class="about3-style">2014 - 2016</h>', unsafe_allow_html=True)
        st.write("- As exportações de vinho brasileiro enfrentaram o câmbio desfavorável e a crise econômica internacional iniciada em 2008. Os preços no mercado internacional caíram, tornando a competitividade da uva brasileira mais desafiadora. Além disso, novas ameaças surgiram, como a produção de uvas sem sementes em escala no Peru, onde os custos são menores.")
        st.write(" ")
        st.markdown(':chart: <h class="about3-style">2016 - 2019</h>', unsafe_allow_html=True)
        st.write("- Em 2013, as exportações de vinho tiveram um pico significativo devido ao aumento expressivo no volume exportado, impulsionado pela exportação do vinho a granel, tanto de mesa quanto fino, aproveitando o programa de escoamento de produção (PEP), que conferia vantagem competitiva no mercado internacional. No entanto, houve uma desvalorização no valor do produto nesse ano. ")
        st.write(" ")
        st.markdown(':chart_with_downwards_trend: <h class="about3-style">2020 - atualmente</h>', unsafe_allow_html=True)
        st.write("- Em contraste, em 2015, as exportações caíram drasticamente pela falta de adoção do PEP, resultando em uma diminuição acentuada no volume exportado. A valorização do produto não compensou a ausência do programa, enquanto a crise política e os escândalos envolvendo a Petrobrás que vieram à tona também contribuíram para o cenário desfavorável.")
        st.write(" ")
        st.markdown('<h class="about3-style">Fontes: <h n class="about2-style"><a href="https://www.infoteca.cnptia.embrapa.br/bitstream/doc/922116/1/PROTASpanoramavitivinicultura2010.pdf">1</a> | <a href="https://releia.ifsertao-pe.edu.br/jspui/bitstream/123456789/821/1/TCC%20-%20PANORAMA%20DA%20EXPORTAÇÃO%20E%20IMPORTAÇÃO%20DE%20VINHOS%20NO%20BRASIL.pdf">2</a>  | <a href="https://www.embrapa.br/busca-de-noticias/-/noticia/9952204/artigo-desempenho-da-vitivinicultura-brasileira-em-2015">3</a>  | <a href="https://www.poder360.com.br/brasil/venda-de-vinho-no-brasil-aumentou-37-na-pandemia/">4</a></h></h>', unsafe_allow_html=True)
