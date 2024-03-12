# Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet

#Teste ARIMA:
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

## Para deep learning
#from keras.models import Sequential
#from keras.layers import LSTM,Dense,Dropout
#from tensorflow.keras.optimizers import Adam
#from keras.models import load_model
#from keras.preprocessing.sequence import TimeseriesGenerator

# Para machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


#Importando banco de dados
import yfinance as yf

symbol = '^BVSP'
start_date = '2014-01-01'
end_date = '2024-03-07'

df = yf.download(symbol, start=start_date, end=end_date)

#Conversão da data para formato datetime
df = df.reset_index('Date')
df['Date'] = pd.to_datetime(df['Date'])


##### INÍCIO DO DASHBOARD #####

st.set_page_config(page_title="Tech Challenge 02 - Gio Queiroz", page_icon=":wine_glass:", layout="wide", initial_sidebar_state="auto", menu_items=None)

# CABEÇALHO
st.markdown(
        """
    <style>
    .title-style {
        text-align: left;
        font-weight: bold;
        color: #962450;
        font-size: 40px;
    }

    .subtitle-style {
        text-align: left;
        font-weight: bold;
        color: #737373;
        font-size: 40px;
    }

    .about-style {
        font-weight: bold;
        color: #d93474;
    }

     .about2-style {
        font-weight: normal;
        color: #000000;
    }

    .about3-style {
        font-weight: bold;
        color: #000000;
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
        color: #962450;
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
st.markdown('<p class="title-style">IBOVESPA: <h n class="subtitle-style">Modelo Preditivo</h></p>', unsafe_allow_html=True)
st.write("Bem-vindo ao modelo preditivo de fechamento do IBOVESPA.")

st.markdown(':chart:<h class="about-style"><h n class="about2-style">Com a análise de dados históricos e indicadores do mercado financeiro e utilizando técnicas avançadas de machine learning, o modelo busca prever com precisão as flutuações do Ibovespa, oferecendo uma ferramenta poderosa para investidores e instituições financeiras.</h>', unsafe_allow_html=True)
st.markdown(':male-office-worker:<h class="about-style"> Sobre o time responsável: <h n class="about2-style">Somos Experts em Data Analytics e integrantes do DataTeam da MaWine.</h>', unsafe_allow_html=True)

st.markdown('<p class="about-style">Objetivos do Relatório:</p>', unsafe_allow_html=True)
st.markdown("- Apresentar aos investidores e acionistas o montante de venda de exportação da MaWine nos últimos 15 anos, separando a análise por país e trazendo prospecções futuras e ações para uma melhoria nas exportações.")

st.markdown('<p class="about-style">Metodologia</p>', unsafe_allow_html=True)
st.markdown("- Utilizando o python e suas bibliotecas, fizemos a aquisição, a manipulação e o tratamento dos dados obtidos na fonte, assim como utilizamos gráficos para permitir a visualização e compreensão desses dados.")

st.markdown('<p class="about3-style">Fonte de Dados: <a href="http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_06" class="about2-style">Dados da Vitivinicultura - Embrapa</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([":bar_chart: Dados de Exportação", ":clipboard: Tabela (Origem x Destino)", ":chart: Análise Final"])
tab1.write()
tab2.write()
tab3.write()




#Visualizando DS Fechamento
plt.figure(figsize = (15,10))
plt.plot(df['Date'], df['Close'], label='IBOVESPA')

plt.legend(loc='best')
plt.show()