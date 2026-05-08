# Módulo Especial de Consultoria na Área de Dados com Agentes de IA
# Projeto Prático Para Consultoria na Área de Dados com Agentes de IA
# Deploy de App Para Day Trade Analytics em Tempo Real com Agentes de IA, Gemini e AWS Para Monetização

# Imports
import os
import re
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from PIL import Image
import base64

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA (ÚNICA VEZ E NO TOPO!)
# ==========================================
try:
    icone_app = Image.open("logo.png") 
    st.set_page_config(page_title="Com:D.A - Busca e Análise de Preços de Ações", page_icon=icone_app, layout="wide")
except FileNotFoundError:
    st.set_page_config(page_title="Com:D.A - Busca e Análise de Preços de Ações", layout="wide")

# ==========================================
# 2. INJEÇÃO DO PLANO DE FUNDO
# ==========================================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    css = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

try:
    set_background('background.png')
except FileNotFoundError:
    st.warning("⚠️ Imagem de fundo não encontrada. Verifique se o arquivo 'background.png' está na pasta.")

# ==========================================
# 3. VALIDAÇÃO DE CHAVES DE API E CONFIGURAÇÃO
# ==========================================
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

if not google_api_key or not alpha_vantage_key:
    st.error("🚨 ERRO CRÍTICO: Chaves da API não encontradas!")
    st.warning("Verifique se o arquivo .env contém GOOGLE_API_KEY e ALPHA_VANTAGE_API_KEY.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
genai.configure(api_key=google_api_key)

########## Analytics (Usando Alpha Vantage) ##########

@st.cache_data
def comda_extrai_dados(ticker):
    """Extrai histórico diário de preços da Alpha Vantage e formata para o Plotly."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={alpha_vantage_key}"
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        st.error(f"Erro ao buscar dados para {ticker}. Verifique o ticker ou o limite da API.")
        return pd.DataFrame()
        
    # Converter JSON para DataFrame Pandas
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df.index = pd.to_datetime(df.index)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.astype(float)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    df.sort_values('Date', inplace=True)
    
    # Retornar aproximadamente os últimos 6 meses (130 dias úteis)
    return df.tail(130).reset_index(drop=True)

def comda_plot_stock_price(hist, ticker):
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Preços das Ações (Últimos 6 Meses)", markers=True)
    st.plotly_chart(fig)

def comda_plot_candlestick(hist, ticker):
    fig = go.Figure(
        data=[go.Candlestick(x=hist['Date'], 
                             open=hist['Open'], 
                             high=hist['High'], 
                             low=hist['Low'], 
                             close=hist['Close'])]
    )
    fig.update_layout(title=f"{ticker} Candlestick Chart (Últimos 6 Meses)")
    st.plotly_chart(fig)

def comda_plot_media_movel(hist, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    
    fig = px.line(hist, 
                  x='Date', 
                  y=['Close', 'SMA_20', 'EMA_20'],
                  title=f"{ticker} Médias Móveis (Últimos 6 Meses)",
                  labels={'value': 'Price', 'Date': 'Date'})
    st.plotly_chart(fig)

def comda_plot_volume(hist, ticker):
    fig = px.bar(hist, 
                 x='Date', 
                 y='Volume', 
                 title=f"{ticker} Trading Volume (Últimos 6 Meses)")
    st.plotly_chart(fig)


########## Ferramentas Customizadas para a IA ##########

def obter_noticias_acao(ticker: str) -> str:
    """Obtém as últimas notícias e sentimento de mercado para uma ação específica usando Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'feed' not in data:
        return "Nenhuma notícia recente encontrada ou limite de API atingido."
        
    # Extrai o resumo das 5 notícias mais recentes
    noticias = []
    for item in data['feed'][:5]:
        titulo = item.get('title', 'Sem título')
        resumo = item.get('summary', 'Sem resumo')
        sentimento = item.get('overall_sentiment_label', 'Neutro')
        noticias.append(f"Título: {titulo}\nResumo: {resumo}\nSentimento Geral: {sentimento}\n")
        
    return "\n".join(noticias)

########## Agentes de IA ##########

ComDA_agente_web_search = Agent(
    name="Agente_Pesquisa_Web",
    role="Buscar informações macroeconômicas curtas e diretas",
    model=Gemini(id="gemini-2.5-flash", api_key=google_api_key),
    tools=[DuckDuckGo()],
    instructions=[
        "Seja extremamente objetivo.",
        "Traga apenas manchetes ou fatos que impactem o day trade hoje.",
        "Sempre cite a fonte de forma resumida."
    ],
    show_tool_calls=True, 
    markdown=True
)

ComDA_agente_financeiro = Agent(
    name="Agente_Analise_Financeira",
    model=Gemini(id="gemini-2.5-flash", api_key=google_api_key),
    tools=[obter_noticias_acao], 
    instructions=[
        "Analise o sentimento das notícias fornecidas pela ferramenta.",
        "Foque apenas em dados que afetam o preço da ação no curtíssimo prazo (Day Trade)."
    ],
    show_tool_calls=True, 
    markdown=True
)

multi_ai_agent = Agent(
    team=[ComDA_agente_web_search, ComDA_agente_financeiro],
    model=Gemini(id="gemini-2.5-flash", api_key=google_api_key),
    instructions=[
        "Você é um analista Sênior de Day Trade.",
        "Forneça um relatório direto, limpo e em tópicos.",
        "NÃO escreva parágrafos longos.",
        "Use tabelas para apresentar dados estruturados.",
        "Sua resposta deve focar em ser 'acionável' (o que o trader deve fazer com a informação)."
    ],
    show_tool_calls=True, 
    markdown=True
)

########## Base de Dados de Ações e Lógica de Moeda ##########

DICIONARIO_ACOES = {
    # --- MERCADO GLOBAL (Listagens Internacionais e ADRs) ---
    "BABA": {"empresa": "Alibaba Group", "nacionalidade": "CN", "moeda": "USD"},
    "TSM": {"empresa": "TSMC - Taiwan Semiconductor", "nacionalidade": "TW", "moeda": "USD"},
    "ASML": {"empresa": "ASML Holding", "nacionalidade": "NL", "moeda": "USD"},
    "NVO": {"empresa": "Novo Nordisk", "nacionalidade": "DK", "moeda": "USD"},
    "TM": {"empresa": "Toyota Motor", "nacionalidade": "JP", "moeda": "USD"},
    "SONY": {"empresa": "Sony Group", "nacionalidade": "JP", "moeda": "USD"},
    "SHEL": {"empresa": "Shell plc", "nacionalidade": "UK", "moeda": "USD"},
    "AZN": {"empresa": "AstraZeneca", "nacionalidade": "UK", "moeda": "USD"},
    "SAP": {"empresa": "SAP SE", "nacionalidade": "DE", "moeda": "USD"},
    "NVS": {"empresa": "Novartis AG", "nacionalidade": "CH", "moeda": "USD"},
    "MELI": {"empresa": "MercadoLibre", "nacionalidade": "AR", "moeda": "USD"},
    "NU": {"empresa": "Nu Holdings (Nubank)", "nacionalidade": "BR", "moeda": "USD"},
    "SHOP": {"empresa": "Shopify Inc.", "nacionalidade": "CA", "moeda": "USD"},
    "BHP": {"empresa": "BHP Group", "nacionalidade": "AU", "moeda": "USD"},
    "HDB": {"empresa": "HDFC Bank", "nacionalidade": "IN", "moeda": "USD"},
    
    # --- BOLSAS EUROPEIAS DIRETAS (Para Alpha Vantage) ---
    "BMW.DEX": {"empresa": "BMW AG", "nacionalidade": "DE", "moeda": "EUR"},
    "MC.PAR": {"empresa": "LVMH", "nacionalidade": "FR", "moeda": "EUR"},
    "HSBA.LON": {"empresa": "HSBC Holdings", "nacionalidade": "UK", "moeda": "GBP"},

    # --- MERCADO AMERICANO (EUA - USD) ---
    "AAPL": {"empresa": "Apple Inc.", "nacionalidade": "US", "moeda": "USD"},
    "MSFT": {"empresa": "Microsoft Corp.", "nacionalidade": "US", "moeda": "USD"},
    "NVDA": {"empresa": "NVIDIA Corp.", "nacionalidade": "US", "moeda": "USD"},
    "AMD": {"empresa": "Advanced Micro Devices", "nacionalidade": "US", "moeda": "USD"},
    "INTC": {"empresa": "Intel Corporation", "nacionalidade": "US", "moeda": "USD"},
    "CRM": {"empresa": "Salesforce Inc.", "nacionalidade": "US", "moeda": "USD"},
    "GOOGL": {"empresa": "Alphabet Inc. (Google)", "nacionalidade": "US", "moeda": "USD"},
    "META": {"empresa": "Meta Platforms (Facebook)", "nacionalidade": "US", "moeda": "USD"},
    "NFLX": {"empresa": "Netflix Inc.", "nacionalidade": "US", "moeda": "USD"},
    "AMZN": {"empresa": "Amazon.com Inc.", "nacionalidade": "US", "moeda": "USD"},
    "WMT": {"empresa": "Walmart Inc.", "nacionalidade": "US", "moeda": "USD"},
    "JPM": {"empresa": "JPMorgan Chase & Co.", "nacionalidade": "US", "moeda": "USD"},
    "V": {"empresa": "Visa Inc.", "nacionalidade": "US", "moeda": "USD"},
    "TSLA": {"empresa": "Tesla Inc.", "nacionalidade": "US", "moeda": "USD"},
    "F": {"empresa": "Ford Motor Co.", "nacionalidade": "US", "moeda": "USD"},
    "XOM": {"empresa": "Exxon Mobil Corp.", "nacionalidade": "US", "moeda": "USD"},
    "UNH": {"empresa": "UnitedHealth Group", "nacionalidade": "US", "moeda": "USD"},
    
    # --- MERCADO BRASILEIRO (B3 - BRL) ---
    "PETR4.SA": {"empresa": "Petrobras PN", "nacionalidade": "BR", "moeda": "BRL"},
    "VALE3.SA": {"empresa": "Vale S.A.", "nacionalidade": "BR", "moeda": "BRL"},
    "ITUB4.SA": {"empresa": "Itaú Unibanco PN", "nacionalidade": "BR", "moeda": "BRL"},
    "BBDC4.SA": {"empresa": "Bradesco PN", "nacionalidade": "BR", "moeda": "BRL"},
    "BBAS3.SA": {"empresa": "Banco do Brasil ON", "nacionalidade": "BR", "moeda": "BRL"},
    "B3SA3.SA": {"empresa": "B3 S.A.", "nacionalidade": "BR", "moeda": "BRL"},
    "ELET3.SA": {"empresa": "Eletrobras ON", "nacionalidade": "BR", "moeda": "BRL"},
    "WEGE3.SA": {"empresa": "WEG S.A.", "nacionalidade": "BR", "moeda": "BRL"},
    "ABEV3.SA": {"empresa": "Ambev S.A.", "nacionalidade": "BR", "moeda": "BRL"},
    "RENT3.SA": {"empresa": "Localiza Rent a Car", "nacionalidade": "BR", "moeda": "BRL"},
    "MGLU3.SA": {"empresa": "Magazine Luiza", "nacionalidade": "BR", "moeda": "BRL"}
}

def identificar_moeda(ticker):
    """Retorna o código da moeda e o símbolo correspondente com inteligência ampliada."""
    # Se estiver no dicionário, pega as infos diretas
    if ticker in DICIONARIO_ACOES:
        moeda = DICIONARIO_ACOES[ticker]["moeda"]
        # Mapeamento estendido de símbolos
        simbolos = {
            "BRL": "R$",
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥"
        }
        return moeda, simbolos.get(moeda, "$")
    
    # Se o usuário digitar um ticker manualmente, tenta inferir pelo sufixo (Bolsas Alpha Vantage)
    elif ticker.endswith(".SA"):
        return "BRL", "R$"
    elif ticker.endswith(".DEX") or ticker.endswith(".PAR") or ticker.endswith(".AS") or ticker.endswith(".MIL"):
        return "EUR", "€"
    elif ticker.endswith(".LON"):
        return "GBP", "£"
    elif ticker.endswith(".TOK"):
        return "JPY", "¥"
    else:
        return "USD", "$"

########## App Web (Interface) ##########

# Inicializa o session state para o text input se não existir
if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = ""

def atualizar_input():
    """Função callback para preencher o input text quando o dropdown for alterado"""
    selecao = st.session_state["selecao_dropdown"]
    if selecao != "Outro (Digitar manualmente)":
        # Extrai apenas o Ticker (tudo antes do " - ")
        st.session_state["ticker_input"] = selecao.split(" - ")[0]
    else:
        # Limpa o input para o usuário digitar livremente
        st.session_state["ticker_input"] = ""

st.sidebar.title("Configurações da Análise")

# Criando as opções para o dropdown
opcoes_dropdown = ["Outro (Digitar manualmente)"] + [f"{ticker} - {dados['empresa']} ({dados['nacionalidade']}, {dados['moeda']})" 
                                                     for ticker, dados in DICIONARIO_ACOES.items()]

st.sidebar.markdown("### Selecione um Ativo da Lista:")
st.sidebar.selectbox("Ativos Populares Globais", opcoes_dropdown, key="selecao_dropdown", on_change=atualizar_input)

st.sidebar.markdown("---")
st.sidebar.title("Instruções")
st.sidebar.markdown("""
### Como Utilizar a App:

- Selecione o ativo desejado no menu lateral **OU** digite um ticker personalizado na barra de pesquisa principal.
- Clique em **Analisar**.
""")

if st.sidebar.button("Suporte"):
    st.sidebar.write("No caso de dúvidas envie e-mail para: alexandrencrego@gmail.com")

col_img, col_titulo = st.columns([1, 11])
with col_img:
    try:
        st.image(icone_app, width=100) 
    except NameError:
        pass # Imagem não carregada, ignora
with col_titulo:
    st.title("Com:D.A - Comunidade de Dados e Automação")

st.header("Análise do Mercado de Ações em Tempo Real")

# Input text principal conectado ao session_state
ticker = st.text_input("Digite o Código (símbolo do ticker):", key="ticker_input").upper()

if st.button("Analisar"):
    if ticker:
        with st.spinner(f"Buscando os Dados de {ticker} em Tempo Real e Gerando Análise. Aguarde..."):
            
            hist = comda_extrai_dados(ticker)
            
            if not hist.empty:
                st.subheader("Análise Gerada Por IA")
                
                # Identifica Moeda Dinamicamente
                moeda_codigo, moeda_simbolo = identificar_moeda(ticker)
                
                # --- 1. CÁLCULO DE VARIAÇÕES RECENTES (PANDAS) ---
                try:
                    preco_atual = hist['Close'].iloc[-1]
                    preco_anterior = hist['Close'].iloc[-2]
                    var_diaria = ((preco_atual - preco_anterior) / preco_anterior) * 100
                    
                    # Variação 1 semana (aprox 5 dias úteis)
                    preco_1s = hist['Close'].iloc[-6] if len(hist) >= 6 else hist['Close'].iloc[0]
                    var_1s = ((preco_atual - preco_1s) / preco_1s) * 100
                    
                    # Variação 1 mês (aprox 21 dias úteis)
                    preco_1m = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                    var_1m = ((preco_atual - preco_1m) / preco_1m) * 100
                    
                    dados_tabela = f"""
| Período | Preço ({moeda_codigo}) | Variação (%) |
| :--- | :--- | :--- |
| Fechamento Atual | {moeda_simbolo}{preco_atual:.2f} | - |
| Diária (1 Dia) | - | {var_diaria:+.2f}% |
| Semanal (5 Dias) | - | {var_1s:+.2f}% |
| Mensal (21 Dias) | - | {var_1m:+.2f}% |
                    """
                except IndexError:
                    dados_tabela = "Dados insuficientes para calcular variações."

                # Recupera o nome da empresa se estiver no dicionário, senão usa só o ticker
                nome_empresa = DICIONARIO_ACOES[ticker]['empresa'] if ticker in DICIONARIO_ACOES else ticker

                # --- 2. NOVO PROMPT DIRECIONADO PARA DADOS ACIONÁVEIS ---
                prompt = f"""
                Aja como um analista quantitativo de Day Trade. Analise a ação {ticker} ({nome_empresa}).
                
                Aqui estão as variações recentes de preço já calculadas:
                {dados_tabela}
                
                Com base nos dados acima e buscando as últimas notícias do mercado, construa um relatório ESTRITAMENTE nesta estrutura:

                ### 📊 Resumo de Variações ({ticker})
                (Exiba a tabela de variações fornecida acima exatamente como está).

                ### 📰 Sentimento do Mercado (Curto Prazo)
                (Use bullet points curtos com as 3 principais notícias ou fatores macroeconômicos de HOJE e classifique o sentimento como 🟢 Positivo, 🔴 Negativo ou ⚪ Neutro).

                ### 🎯 Ação Recomendada para Day Trade
                **Viés:** (Alta, Baixa, ou Consolidação)
                **Justificativa:** (Uma frase curta explicando o porquê).
                **Nível de Risco:** (Alto, Médio, Baixo).
                """
                
                # --- 3. EXECUÇÃO DO AGENTE ---
                ai_response = multi_ai_agent.run(prompt)

                clean_response = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to_finance_ai_agent.*\n?)","", ai_response.content, flags=re.MULTILINE).strip()
                st.markdown(clean_response)

                st.markdown("---")
                st.subheader("Visualização dos Dados")
                
                comda_plot_stock_price(hist, ticker)
                comda_plot_candlestick(hist, ticker)
                comda_plot_media_movel(hist, ticker)
                comda_plot_volume(hist, ticker)
    else:
        st.error("Insira um símbolo de ação válido ou selecione um na lista.")