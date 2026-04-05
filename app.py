import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import google.generativeai as genai

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="OptionVisualizer Pro", layout="wide", initial_sidebar_state="expanded")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DE CALCUL BLACK-SCHOLES ---
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Calcule les grecques pour une option européenne."""
    if T <= 0: T = 1e-6  # Évite la division par zéro
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (pour 1% de changement de vol)
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    
    # Theta (journalier)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (term1 - term2) / 365
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365
        
    return delta, gamma, theta, vega

# --- CACHE DONNÉES FINANCIÈRES ---
@st.cache_data(ttl=3600)
def fetch_market_data(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        # Calcul de la volatilité historique (annualisée)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        return {"price": current_price, "vol": volatility, "name": tk.info.get('longName', ticker_symbol)}
    except:
        return None

# --- CONFIGURATION IA GEMINI ---
def get_ai_analysis(prompt):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur IA : Assurez-vous que GEMINI_API_KEY est configuré. {str(e)}"

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Paramètres")
    symbol = st.text_input("Ticker Yahoo Finance", value="AAPL").upper()
    data = fetch_market_data(symbol)
    
    if data:
        st.success(f"Connecté à : {data['name']}")
        s_price = st.number_input("Prix Sous-jacent ($)", value=float(data['price']))
        vol = st.slider("Volatilité Implicite (%)", 5.0, 150.0, float(data['vol']*100)) / 100
    else:
        st.error("Symbole introuvable")
        s_price = st.number_input("Prix Sous-jacent ($)", value=100.0)
        vol = st.slider("Volatilité Implicite (%)", 5.0, 150.0, 20.0) / 100

    strike = st.number_input("Prix d'Exercice (Strike)", value=s_price)
    rate = st.slider("Taux sans risque (%)", 0.0, 10.0, 4.0) / 100
    opt_type = st.selectbox("Type d'Option", ["Call", "Put"])

# --- CORPS DE L'APPLICATION ---
st.title(f"Analyse des Grecques : {symbol}")

if data:
    # Préparation des données pour les graphiques 3D
    price_range = np.linspace(s_price * 0.8, s_price * 1.2, 30)
    time_range = np.linspace(0.01, 1.0, 30) # de 4 jours à 1 an
    P, T = np.meshgrid(price_range, time_range)

    # Fonction pour générer les Z (grecques) selon le type choisi
    def get_z_data(greek_name):
        func = np.vectorize(lambda p, t: bs_greeks(p, strike, t, rate, vol, opt_type.lower())[["delta", "gamma", "theta", "vega"].index(greek_name)])
        return func(P, T)

    # Onglets
    t_delta, t_gamma, t_theta, t_vega, t_ai = st.tabs(["Δ Delta", "Γ Gamma", "Θ Theta", "ν Vega", "🤖 Analyse IA"])

    def plot_surface(Z, title, label, color="Viridis"):
        fig = go.Figure(data=[go.Surface(z=Z, x=price_range, y=time_range, colorscale=color)])
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='Prix ($)', yaxis_title='Temps (Années)', zaxis_title=label),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        return fig

    with t_delta:
        st.plotly_chart(plot_surface(get_z_data("delta"), "Surface du Delta", "Delta", "Blues"), use_container_width=True)
    
    with t_gamma:
        st.plotly_chart(plot_surface(get_z_data("gamma"), "Surface du Gamma", "Gamma", "Magma"), use_container_width=True)
        
    with t_theta:
        st.plotly_chart(plot_surface(get_z_data("theta"), "Surface du Theta (Time Decay)", "Theta", "Thermal"), use_container_width=True)
        
    with t_vega:
        st.plotly_chart(plot_surface(get_z_data("vega"), "Surface du Vega", "Vega", "Algae"), use_container_width=True)

    with t_ai:
        st.subheader("Analyse prédictive par Gemini")
        current_greeks = bs_greeks(s_price, strike, 0.5, rate, vol, opt_type.lower())
        
        user_query = st.text_area("Question spécifique pour l'IA :", 
                                 f"En tant qu'expert en produits dérivés, analyse le risque de cette option {opt_type} sur {symbol}. "
                                 f"Le Delta est de {current_greeks[0]:.2f} et le Gamma de {current_greeks[1]:.4f}. "
                                 f"Quels sont les dangers à l'approche de l'échéance ?")
        
        if st.button("Générer l'analyse"):
            with st.spinner("Analyse des surfaces en cours..."):
                result = get_ai_analysis(user_query)
                st.markdown("---")
                st.markdown(result)

else:
    st.info("Veuillez saisir un ticker valide dans la barre latérale pour commencer.")

# --- FOOTER ---
st.markdown("---")
st.caption("Données fournies par Yahoo Finance. Calculs basés sur le modèle Black-Scholes-Merton.")
