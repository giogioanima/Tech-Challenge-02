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
import statsmodels.api as sm

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
start_date = '2004-01-01'
end_date = '2024-01-01'
df = yf.download(symbol, start=start_date, end=end_date)

# Reindexandoo o DataFrame
df = df.reset_index()

# Autocorrelação
df_limpo = df
df_limpo['Date'] = pd.to_datetime(df_limpo['Date'])
df_limpo = df_limpo.set_index('Date')    

# Cálculo da média móvel
df_ma = pd.DataFrame(df[['Date', 'Close']])
df_ma.set_index('Date', inplace=True)
df_ma['MA_window_30'] = df_ma['Close'].rolling(30).mean().shift()
df_ma['MA_window_90'] = df_ma['Close'].rolling(90).mean().shift() 

# Calcular a média móvel e o desvio padrão
rolmean = df_ma.rolling(window=12).mean()
rolstd = df_ma.rolling(window=12).std()


# PROPHET #
df_pro = df.copy()  # Copiar o DataFrame original
df_pro['Date'] = pd.to_datetime(df_pro['Date'])
df_pro.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'], inplace=True)
df_pro.columns = ['ds', 'y']  # Renomear colunas para o formato esperado pelo Prophet

train_data = df_pro.sample(frac=0.8, random_state=0)
test_data = df_pro.drop(train_data.index)
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

    .about5-style {
    font-weight: bold;
    color: #f145f7;
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

st.markdown('<p class="about3-style">Fonte de Dados: <h class="about2-style">Optamos pela biblioteca Yahoo Finance para ter acesso aos dados diários. No entanto, também é possível obter os dados em: <a href="https://br.investing.com/indices/bovespa-historical-data" class="about2-style">IBOVESPA (IBOV) Historical Data - Investing</a>. <p class="about3-style">Período: 01/01/2004 a 01/01/2024.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([":bar_chart: Análise Exploratória (EDA)", ":clipboard: Modelos", ":chart: Forecasting"])
tab1.write()
tab2.write()
tab3.write()

## ANÁLISE EXPLORATÓRIA DOS DADOS ##
#Visualizando DataSeries Fechamento

with tab1:
    st.markdown('<p class="tab1-title">Análise Exploratória dos Dados (EDA)</p>', unsafe_allow_html=True)

    st.write("Iniciamos o nosso projeto com a etapa vital da Análise Exploratória dos Dados, fundamental para a construção do PredVespa. Aqui, imergimos nos dados coletados e na história econômica do país, realizando passos essenciais que nos conduzirão a uma compreensão mais profunda do comportamento do IBOVESPA ao longo do tempo.")    
    st.write("Nosso primeiro passo foi realizar o método 'pd.datetime' para transformar a primeira coluna em objetos de data e hora. É uma ferramenta essencial para lidar com dados temporais em análises de dados e projetos que envolvem séries temporais.")    
    st.write("Em seguida, transformamos a coluna data em index.")    

    coluna4, coluna5 = st.columns(2, gap="large")
    with coluna4:
        st.write("O gráfico de linha abaixo exibe a flutuação diária do fechamento do IBOVESPA nos últimos 20 anos. O eixo x representa a série temporal, enquanto o eixo y indica os valores de fechamento.")
        ax = px.line(df, x="Date", y="Close", color_discrete_sequence=['#d7e46f'], width=800, height=560, range_y=[0, 150000], title="Valores Diários de Fechamento (IBOVESPA) - Série Temporal")
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
        title="Série Temporal",
        tickmode = 'array',
        tickvals = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        titlefont=dict(size=16),
        tickfont=dict(size=14),
        tickangle=-45,
        title_standoff=20 # The higher the value, the farther away it is displayed
        ))
        ax.add_hline(y="Close", line_color="black", line_width=1, line_dash="dot", label=dict(text="                                                                  Valor Médio: US$ 7,51 MM", textposition="middle", font=dict(size=14, color="black"))
        )
        st.plotly_chart(ax, use_container_width=True)

    with coluna5:
        st.markdown(':triangular_flag_on_post: <h class="about3-style">2004 - 2007</h>', unsafe_allow_html=True)
        st.write('<h class="about2-style">* O IBOVESPA experimentou um período de crescimento impulsionado pela recuperação econômica mundial, políticas econômicas favoráveis e o boom das commodities. <a href="https://www.infomoney.com.br/mercados/bolsa-fecha-2005-como-alternativa-mais-rentavel-de-investimento/">[1]</a> <a href="https://www.nexojornal.com.br/explicado/2016/03/31/as-commodities-e-seu-impacto-na-economia-do-brasil">[2]</a></h>', unsafe_allow_html=True)
        st.write(" ")        
        st.markdown(':chart_with_downwards_trend: <h class="about3-style">2008 - 2011</h>', unsafe_allow_html=True)
        st.write('<h class="about2-style">* A crise financeira global de 2008 resultou em uma queda acentuada no IBOVESPA, impactado pela desaceleração econômica global, redução na demanda por commodities e retirada de investimentos estrangeiros. <a href="https://www.infomoney.com.br/mercados/crise-de-2008-quebra-do-lehman-brothers-virou-referencia-e-gerou-mudancas-importantes-no-mercado/">[3]</a>', unsafe_allow_html=True)
        st.write(" ")
        st.markdown(':arrow_right: <h class="about3-style">2012 - 2017</h>', unsafe_allow_html=True)
        st.write('<h class="about2-style">* Desafios econômicos e políticos, tanto internos quanto externos, resultaram em volatilidade no IBOVESPA, influenciado pela crise na Europa, instabilidade política no Brasil e desaceleração econômica global. Em janeiro de 2016 houve uma das maiores quedas dos últimos 20 anos. <a href="https://g1.globo.com/economia/mercados/noticia/bovespa-tem-primeira-valorizacao-anual-desde-2012.ghtml">[4]</a>', unsafe_allow_html=True)
        st.write(" ")
        st.markdown(':chart_with_upwards_trend: <h class="about3-style">2018 - 2020</h>', unsafe_allow_html=True)
        st.write('<h class="about2-style">* O IBOVESPA enfrentou turbulências devido à guerra comercial EUA-China, tensões geopolíticas e, posteriormente, à pandemia de COVID-19, resultando em quedas significativas no mercado. <a href="https://veja.abril.com.br/economia/guerra-comercial-entre-eua-e-china-derruba-ibovespa-e-bolsas-pelo-mundo">[5]</a> <a href="https://economia.uol.com.br/noticias/reuters/2020/04/01/ibovespa-comeca-abril-com-forte-queda-em-meio-a-persistentes-preocupacoes-sobre-covid-19.htm">[6]</a>', unsafe_allow_html=True)
        st.write(" ")
        st.markdown(':round_pushpin: <h class="about3-style">2021 - 2023</h>', unsafe_allow_html=True)
        st.write('<h class="about2-style">* O IBOVESPA foi influenciado pela recuperação econômica pós-pandemia, reformas econômicas no Brasil e aumento dos preços das commodities, apesar de preocupações com inflação e incertezas políticas. <a href="https://g1.globo.com/economia/noticia/2023/12/28/dolar-ibovespa.ghtml">[7]</a>', unsafe_allow_html=True)


## ANÁLISE EXPLORATÓRIA DOS DADOS ##
#Decomposição
    st.markdown("""---""")
    st.markdown('<p class="tab1-title">Decomposição</p>', unsafe_allow_html=True)

    st.write("A análise de séries temporais envolve a decomposição da série em componentes distintos, revelando informações latentes e estruturais. Esses componentes, como tendência, sazonalidade e ruído, permitem uma compreensão mais profunda da dinâmica dos dados ao longo do tempo. Essa técnica estatística facilita a identificação de padrões e tendências subjacentes, contribuindo para uma análise mais precisa e informada das séries temporais.")    

    st.write("Para iniciar a decomposição, retiramos todas as colunas que não são relevantes para o nosso modelo.")

    st.write("Utilizaremos o Statsmodel para desmembrar nossos dados e visualizar em gráficos, o que nos auxiliará na compreensão dos dados e na decisão sobre qual modelo preditivo adotar.")

    img_codedec = r'C:\Users\gih\Downloads\postechchall02\img\codedec.png'
    st.image(img_codedec, use_column_width=False)
    st.write(" ")
    img_path = r'C:\Users\gih\Downloads\postechchall02\img\decomp.png'
    st.image(img_path, use_column_width=True)

    st.write(" ")
    st.markdown(':memo: <h class="about3-style">Observações</h>', unsafe_allow_html=True)
    st.write("- A decomposição da série temporal nos ajuda a compreender padrões complexos. Identificamos tendências de longo prazo, sazonalidades de curto prazo e resíduos anteriormente negligenciados, aprimorando nossos modelos. E ainda, a decomposição revela se a série é aditiva ou multiplicativa, o que influencia a escolha do modelo para previsões futuras.")
    st.write("- A profusão de oscilações na sazonalidade e o considerável ruído identificado sugerem que estamos lidando com uma série não-estacionária.")
    st.write(" ")

    st.markdown("""---""")
    st.markdown('<p class="tab1-title">Teste de Dickey-Fuller Aumentado (ADF)</p>', unsafe_allow_html=True)

    st.write("O Teste de Dickey-Fuller Aumentado (ADF) é uma técnica estatística que determina se uma série temporal é estacionária ou não, analisando se há raiz unitária nos dados. Ele compara as diferenças entre os valores ao longo do tempo e, se o valor-p resultante for menor que um limite específico (geralmente 0,05), indica que a série é estacionária. Se for maior, indica que a série é não estacionária.")
    st.write("A estacionariedade em séries temporais significa que suas propriedades estatísticas, como média e variância, permanecem constantes ao longo do tempo. Isso é crucial para muitos modelos de previsão. O Teste de Dickey-Fuller Aumentado (ADF) é usado para verificar se uma série é estacionária, identificando tendências significativas que podem influenciar as análises e previsões.")
    st.write(" ")

    img_adf1 = r'C:\Users\gih\Downloads\postechchall02\img\test_adf1.png'
    st.image(img_adf1, use_column_width=False)
    st.write(" ")
    st.markdown(':memo: <h class="about3-style">Observações</h>', unsafe_allow_html=True)
    st.write("- A estatística de teste é maior do que todos os valores críticos (1%, 5% e 10%) e o valor-p de 0.796 é maior do que o nível de significância comum de 0.05, indicando que não podemos rejeitar a hipótese nula de não estacionariedade na série temporal.")
    st.write("- Em outras palavras, podemos considerar que se trata de uma série não estacionária.")


with tab2:
    st.markdown('<p class="tab1-title">Modelos Avaliados</p>', unsafe_allow_html=True)
    st.write("Neste estudo, exploraremos diferentes modelos de séries temporais com o objetivo de realizar previsões precisas em um conjunto de dados específico. Investigaremos três modelos amplamente utilizados: média móvel, ARIMA (AutoRegressive Integrated Moving Average) e Prophet. Cada um desses modelos tem suas próprias características e suposições, e compararemos seu desempenho em termos de precisão das previsões. Ao testar esses modelos, esperamos identificar o mais adequado para o conjunto de dados em questão e fornecer insights valiosos para aplicações futuras em análise e previsão de séries temporais.")
    option = st.selectbox(
    'Selecione o modelo:',
    ('Modelo de Média Móvel (MA)', 'Modelo Autorregressivo Integrado de Média Móvel (ARIMA)', 'Biblioteca META Prophet'))

    if option == "Modelo de Média Móvel (MA)":
        coluna6, coluna7 = st.columns(2, gap="large")
        with coluna6:
            st.write(" ")
            st.write(':white_medium_square: O modelo de média móvel suaviza uma série temporal calculando a média dos valores em torno de cada ponto, usando uma janela deslizante. Isso ajuda a eliminar variações aleatórias e identificar tendências de longo prazo, facilitando a compreensão das mudanças ao longo do tempo.')
            st.write(':white_medium_square: Descreve a relação entre uma observação e um erro residual obtido após aplicar médias móveis em observações passadas. Ele é útil para modelar a dependência serial entre observações em séries temporais.')

            #Grafico 1
            fig = plt.figure(figsize=(4,5.7))
            plt.grid(True)
            plt.plot(df_ma['Close'], label='Close')
            plt.plot(df_ma['MA_window_30'], label='MA window 30 days')
            plt.plot(df_ma['MA_window_90'], label='MA window 90 days')
            plt.legend(loc=2)
            plt.title('Dados Reais e Médias Móveis X Tempo', color='white', fontsize=18)
            plt.xlabel('Série Temporal', color='white', fontsize=14)
            plt.ylabel('Valor de Fechamento', color='white', fontsize=14)
            legend = plt.legend()
            plt.setp(legend.get_texts(), color='w')
            st.plotly_chart(fig, use_container_width=True)

            st.write("- Percebemos, no gráfico acima, uma grande distância entre os dados reais e os dados preditos pelo modelo de média móvel. Isso é preocupante porque indica uma baixa precisão nas previsões, o que pode resultar em decisões financeiras errôneas e perdas no mercado de ações.")

        with coluna7:
            st.write(" ")
            st.markdown(':mag_right: <h class="about3-style">Validando com Mean Absolute Percentage Error (MAPE)</h>', unsafe_allow_html=True)
            
            img_MAmape = r'C:\Users\gih\Downloads\postechchall02\img\MA_mape.png'
            st.image(img_MAmape, use_column_width=False)
            st.write(" ")

            def calculate_mape(y_true, y_pred):
                errors = np.abs(y_true - y_pred)
                ape = errors / y_true * 100
                mape = np.mean(ape)
                return mape
            y_true = df_ma['Close']
            y_pred30 = df_ma['MA_window_30']
            y_pred90 = df_ma['MA_window_90']

            # Calcular o MAPE
            st.markdown(':sparkles: <h class="about3-style">Resultados:</h>', unsafe_allow_html=True)
            mape = calculate_mape(y_true, y_pred30)
            st.write(f"MAPE - 30d: {mape:.2f}% - Um ótimo resultado, quando observado isoladamente.")
            mape = calculate_mape(y_true, y_pred90)
            st.write(f"MAPE - 90d: {mape:.2f}% - Também um ótimo resultado, mas com desempenho inferior ao de 30 dias.")
            st.write(" ")
        
            st.markdown(':x: <h class="about3-style">Conclusões:</h>', unsafe_allow_html=True)
            st.write("- Avaliação do MAPE: Embora o MAPE seja uma métrica comum para avaliar a precisão de modelos de previsão, é importante considerar outras métricas e realizar uma análise mais abrangente do desempenho do modelo. O MAPE sozinho pode não fornecer uma imagem completa do desempenho do modelo, especialmente em séries temporais complexas.")
            st.write("- O modelo de média móvel pode ser inadequado para analisar séries temporais do IBOVESPA devido à sua simplicidade, ao atraso nas previsões, à sensibilidade ao tamanho da janela e à incapacidade de capturar padrões complexos do mercado.")
            st.write("- A discrepância observada entre os dados reais e as previsões do modelo de média móvel no gráfico sugere uma imprecisão significativa nas previsões. Isso levanta preocupações quanto à confiabilidade do modelo, especialmente dada a complexidade dos dados envolvidos.")
        
    elif option == "Modelo Autorregressivo Integrado de Média Móvel (ARIMA)":
        st.write(" ")
        st.write(':white_medium_square: O ARIMA (Modelo Autorregressivo Integrado de Médias Móveis) é uma classe de modelos estatísticos usados para analisar e prever dados de séries temporais. Enquanto os modelos de suavização exponencial descrevem tendências e sazonalidades, o ARIMA se concentra nas autocorrelações dos dados. Ele combina autoregressão, médias móveis e diferenciação.')
        st.write(':white_medium_square: Os termos autoregressivos captam a influência dos valores passados, os de média móvel capturam erros de previsão anteriores, e a diferenciação ajuda a capturar tendências. Os hiperparâmetros incluem P (número de lags), D (número de diferenciações) e Q (ordem de média móvel).')
        
        coluna8, coluna9 = st.columns(2, gap="large")
        with coluna8:
            st.markdown("""---""")
            st.write(" ")
            st.markdown(':arrows_counterclockwise: <h class="about3-style">Transformação da Série em Estacionária</h>', unsafe_allow_html=True)
            st.write("O modelo ARIMA requer que os dados de séries temporais sejam estacionários para produzir previsões precisas. Se a série não for estacionária, os parâmetros estimados podem ser imprecisos, resultando em previsões inadequadas. Portanto, é crucial transformar a série em estacionária antes de aplicar o ARIMA, geralmente através de diferenciação ou outras técnicas de preparação de dados.")
            st.write("- Faremos essa transformação utilizando o método da diferenciação:")

            # Função para testar a estacionariedade da série transformada
            def test_stationarity(timeseries):

                #Determinar estatísticas contínuas
                movingAverage = timeseries.rolling(window=12).mean()
                movingSTD = timeseries.rolling(window=12).std()

                #Plot estatísticas contínuas
                orig = plt.plot(timeseries, color='blue', label='Original')
                mean = plt.plot(movingAverage, color='red', label='Média Móvel')
                std = plt.plot(movingSTD, color='black', label='Desvio Padrão')
                plt.legend(loc='best')
                plt.title('Média Móvel e Desvio Padrão')
                plt.show(block=False)

                #Performance do Dickey–Fuller:
                print('Results of Dickey Fuller Test:')
                dftest = adfuller(timeseries['Close'], autolag='AIC')
                dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
                for key,value in dftest[4].items():
                    dfoutput['Critical Value (%s)'%key] = value
                st.write(dfoutput)

            df_logscale = np.log(df_limpo)
            df_diff = df_logscale - df_logscale.shift() #diferença entre o valor anterior e o atual

            img_diff = r'C:\Users\gih\Downloads\postechchall02\img\diff.png'
            st.image(img_diff, use_column_width=False)
            st.write(" ")

            df_diff.dropna(inplace=True)
            test_stationarity(df_diff)

            st.markdown(':sparkles: <h class="about3-style">Resultado:</h>', unsafe_allow_html=True)
            st.write('- Com base nos resultados fornecidos, podemos concluir que a série temporal é estacionária após a transformação por diferenciação, pois o valor da estatística do teste é significativamente menor que os valores críticos, e o valor p é muito próximo de zero. Isso sugere que não há evidências estatísticas para rejeitar a hipótese nula de estacionariedade.')
           
            st.markdown("""---""")

        with coluna9:
            st.markdown("""---""")
            st.write(" ")
            st.markdown(':microscope: <h class="about3-style">Análise de Autocorrelação (ACF e PACF)</h>', unsafe_allow_html=True)
            st.write("Analisar a autocorrelação em séries temporais é crucial para entender padrões temporais, selecionar e validar modelos apropriados, e avaliar a estacionariedade dos dados. A presença de autocorrelação indica dependência serial nos dados, influenciando a inclusão de termos autoregressivos e de médias móveis em modelos como o ARIMA. Além disso, a autocorrelação dos resíduos é utilizada para validar a adequação do modelo.")
            
            img_acf = r'C:\Users\gih\Downloads\postechchall02\img\acf.png'
            st.image(img_acf, width=615, use_column_width=False)
            st.write(" ")

            # Calcular o MAPE
            st.markdown(':chart_with_upwards_trend: <h class="about3-style">Observações:</h>', unsafe_allow_html=True)
            st.write('- Na série temporal que foi diferenciada, observa-se uma interligação notável entre os gráficos de ACD e PACF. Isso sugere uma falta de distinção clara entre os efeitos de autocorrelação e autocorrelação parcial. Tal fenômeno pode surgir quando há uma mescla desses efeitos na série, dificultando a identificação precisa de cada um deles.')
            st.write('- No gráfico ACF, observamos que a curva intersecta a linha y=0,0 em x=2, indicando um valor de Q=2. Por outro lado, no gráfico PACF, vemos que a curva toca a linha y=0,0 em x=2, sugerindo um valor de P=2.')
            st.markdown("""---""")

        coluna10, coluna11 = st.columns(2, gap="large")
        with coluna10:
            st.write(" ")
            st.markdown(':microscope: <h class="about3-style">Aplicando o Modelo ARIMA</h>', unsafe_allow_html=True)
            st.write("Ao analisar modelos individuais AR e MA antes de aplicar o ARIMA, busca-se minimizar a Soma dos Quadrados dos Resíduos (RSS), onde valores mais baixos indicam uma melhor adequação do modelo aos dados. Idealmente, visamos alcançar um RSS o mais próximo possível de zero, indicando um ajuste ideal do modelo aos dados observados.")
            img_arima = r'C:\Users\gih\Downloads\postechchall02\img\arima.png'
            st.image(img_arima, width=400, use_column_width=False)
            st.write(" ")
            st.write("- Um RSS de 1.4508 sugere que o modelo ARIMA tem uma adequação razoável aos dados observados, indicando uma boa captura das características da série temporal. No entanto, é importante considerar outras métricas de avaliação e o contexto específico da aplicação para uma interpretação completa.")
            st.write("- Dessa forma, iremos validar o modelo com o MAPE.")
            st.write(" ")
           

        with coluna11:
            st.write(" ")
            st.markdown(':mag_right: <h class="about3-style">Validando com Mean Absolute Percentage Error (MAPE)</h>', unsafe_allow_html=True)
           
            img_mapearima = r'C:\Users\gih\Downloads\postechchall02\img\mapearima.png'
            st.image(img_mapearima, width=600, use_column_width=False)
            st.write(" ")
            st.write("MAPE: 1132.76%")
            st.write(" ")
            st.markdown(':x: <h class="about3-style">Conclusões:</h>', unsafe_allow_html=True)
            st.write("- O MAPE de 1132.76% obtido pelo modelo ARIMA indica uma falta significativa de precisão nas previsões. Isso sugere que o modelo não está capturando efetivamente a estrutura dos dados e que suas previsões estão fora por uma margem substancial.") 
            st.write("- Dessa forma, concluímos que o modelo ARIMA atual não é o mais apropriado para esses dados, sugerindo a necessidade de explorar outras abordagens de modelagem para melhorar a precisão das previsões.")
        


    elif option == "Biblioteca META Prophet":
        st.write(" ")
        st.write(':white_medium_square: Exploraremos o uso do Prophet, uma biblioteca de previsão de séries temporais desenvolvida pelo Facebook, como uma opção viável para prever o IBOVESPA. O Prophet é reconhecido por sua facilidade de uso e capacidade de lidar com características complexas dos dados financeiros, como sazonalidade e tendências não lineares. Sua adaptabilidade e capacidade de capturar padrões sazonais e tendências fazem dele uma escolha atraente para análise de séries temporais financeiras, oferecendo insights valiosos para investidores e tomadores de decisão.')
                 
        coluna12, coluna13 = st.columns(2, gap="large")
        with coluna12:
            st.markdown("""---""")
            st.write(" ")
            st.markdown('📋 <h class="about3-style">Aplicando o Prophet</h>', unsafe_allow_html=True)
            st.write('Na primeira etapa da análise com o Prophet, é essencial modelar os dados adequadamente, seguindo o padrão de utilização das variáveis ds e y. Essa padronização é fundamental para garantir a correta interpretação e funcionamento do modelo.')

            img_dsy = r'C:\Users\gih\Downloads\postechchall02\img\ds_y.png'
            st.image(img_dsy, width=450, use_column_width=False)

            st.write(" ")
            st.write('Após etapas de modelagem, como a separação da base de treino e de teste, é possível obter os seguintes resultados:')
            img_prop = r'C:\Users\gih\Downloads\postechchall02\img\prophet.png'
            st.image(img_prop, use_column_width=True)
            img_change = r'C:\Users\gih\Downloads\postechchall02\img\changepoints.png'
            st.image(img_change, use_column_width=True)

            st.write('- Nos gráficos acima, os pontos pretos representam os dados reais da série temporal, enquanto os pontos vermelhos correspondem às projeções geradas pelo modelo Prophet. Além disso, a linha azul ao redor das projeções indica a margem de confiança associada às previsões.')

            st.write('- Essencialmente, o Prophet utiliza um modelo de regressão aditiva baseado em decomposição de tendência, sazonalidade e efeitos de feriados para gerar suas projeções. Os pontos vermelhos são gerados a partir da combinação desses componentes, representando as estimativas do modelo para os valores futuros da série temporal.')
            
            st.write('- Ao observar os gráficos, podemos perceber uma simularidade na relação entre os pontos pretos (dados reais) e os pontos vermelhos (projeções do Prophet). Isso sugere que o modelo Prophet está capturando efetivamente os padrões presentes nos dados históricos e é capaz de fazer previsões que se alinham bem com o comportamento observado na série temporal. A presença da margem de confiança ao redor das projeções também nos fornece uma medida da incerteza associada às previsões, permitindo uma avaliação mais completa da precisão do modelo.')

        with coluna13:
            st.markdown("""---""")
            st.write(" ")
            st.markdown(':bar_chart: <h class="about3-style">Decompondo com o Prophet</h>', unsafe_allow_html=True)
            st.write('O Prophet possui uma capacidade única de decompor automaticamente séries temporais em seus diversos componentes. Isso inclui a identificação da tendência geral dos dados, a análise dos padrões sazonais anuais e a consideração dos efeitos de feriados, quando disponíveis. Essa funcionalidade permite uma compreensão mais abrangente e detalhada da estrutura subjacente dos dados temporais, facilitando a modelagem e previsão mais precisa.')

            st.write('<h class="about4-style">modelo.plot_components(forecast, figsize=(10,6));</h>', unsafe_allow_html=True)
            img_propdec = r'C:\Users\gih\Downloads\postechchall02\img\propdec.png'
            st.image(img_propdec, use_column_width=True)

            st.write(" ")
            st.markdown(':memo: <h class="about3-style">Observações</h>', unsafe_allow_html=True)
            st.write('- Foi observado que os índices da bolsa exibem uma tendência de flutuação positiva durante os fins de semana.', unsafe_allow_html=True)
            st.write('- O mês de fevereiro demonstra uma performance superior no IBOVESPA.', unsafe_allow_html=True)

            st.write(" ")
            st.markdown(':mag_right: <h class="about3-style">Validando com Mean Absolute Percentage Error (MAPE)</h>', unsafe_allow_html=True)
            img_propmape = r'C:\Users\gih\Downloads\postechchall02\img\propmape.png'
            st.image(img_propmape, use_column_width=True)
            st.write(" ")
            st.markdown('<h class="about5-style">* MAPE: 6.71%</h>', unsafe_allow_html=True)
            st.write(" ")
            st.markdown(':heavy_check_mark: <h class="about3-style">Conclusões:</h>', unsafe_allow_html=True)
            st.write("- Com um MAPE de 6.71%, o modelo Prophet demonstra uma precisão satisfatória em suas previsões, indicando sua capacidade de realizar previsões próximas aos valores reais. Essa baixa taxa de erro sugere que o Prophet é uma ferramenta confiável para previsões futuras do mercado financeiro.") 
            st.write("-  Sua capacidade de capturar padrões complexos nos dados, aliada à sua implementação simplificada e fácil utilização, fazem do Prophet uma escolha justificada para a análise e previsão de séries temporais no mercado financeiro, auxiliando na formulação de estratégias de investimento mais informadas e eficazes.")
with tab3:
    st.markdown('<p class="tab1-title">Forecasting - Prophet</p>', unsafe_allow_html=True)
    from prophet.plot import plot_plotly

    st.write(" ")
    # Plotar o gráfico
    fig = plot_plotly(modelo, forecast)
    fig.update_traces(marker=dict(color='green'))  # Alterar a cor dos dados para vermelho
    fig.update_layout(xaxis_title="Data", title='Previsão de Pontuação IBOVESPA X Dados Reais')  # Alterar o rótulo do eixo x
    fig.update_layout(yaxis_title="Valor de Fechamento")  # Alterar o rótulo do eixo x
    st.plotly_chart(fig, use_column_width=True)

    st.write("- Ao concluir o processo de previsão utilizando o Prophet, destaca-se sua notável precisão em fornecer projeções que se alinham de forma próxima aos dados reais. A capacidade do modelo de capturar com eficácia padrões complexos, como sazonalidade e tendências, é evidente na comparação entre os pontos verdes representando os dados atuais e as projeções geradas.")

    st.write("- Vale ressaltar que a interatividade oferecida pelo gráfico permite uma análise mais aprofundada dos resultados do modelo de machine learning. Ao explorar os detalhes do gráfico, os usuários podem identificar nuances e insights valiosos que podem não ser imediatamente aparentes em uma análise superficial.")

    st.write("- Essa capacidade de interação com o gráfico enriquece a experiência de análise e facilita a interpretação dos resultados, tornando o Prophet uma ferramenta poderosa para previsão de séries temporais em uma variedade de aplicações.")




            

