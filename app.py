import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="FinAnalytics AI", layout="wide")

# --- FONCTIONS DE CALCUL (BLACK-SCHOLES) ---
def calc_gamma(S, K, T, r, sigma):
    """Calcule le Gamma d'une option."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# --- CACHE POUR YAHOO FINANCE ---
@st.cache_data(ttl=3600)  # Cache d'une heure
def get_ticker_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1d")
    if hist.empty:
        return None, None
    current_price = hist['Close'].iloc[-1]
    # Estimation de la volatilité (simplifiée pour l'exemple)
    vola = ticker.history(period="1y")['Close'].pct_change().std() * np.sqrt(252)
    return current_price, vola

# --- INTERFACE STREAMLIT ---
st.title("📊 Analyseur d'Options Avancé")

with st.sidebar:
    st.header("Paramètres")
    ticker_input = st.text_input("Symbole (ex: AAPL, TSLA)", value="AAPL")
    risk_free_rate = st.slider("Taux sans risque (%)", 0.0, 5.0, 1.5) / 100
    
    current_price, sigma_est = get_ticker_data(ticker_input)
    
    if current_price:
        st.success(f"Prix actuel : {current_price:.2f}$")
        st.info(f"Volatilité Hist. (1an) : {sigma_est:.2%}")
    else:
        st.error("Symbole invalide")

# --- ONGLETS ---
tab_gamma, tab_analysis = st.tabs(["Surface du Gamma (3D)", "Analyse IA Gemini"])

# --- ONGLET 1 : GAMMA 3D ---
with tab_gamma:
    st.subheader("Visualisation du Gamma : f(Prix, Temps)")

    if current_price:
        # Création des axes
        prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
        times = np.linspace(0.01, 1, 50) # De 3 jours à 1 an
        P, T = np.meshgrid(prices, times)
        
        # Calcul du Gamma pour la surface
        # On fixe le Strike au prix actuel (At-the-money)
        K = current_price 
        Z = np.vectorize(calc_gamma)(P, K, T, risk_free_rate, sigma_est)

        # Création du graphique Plotly
        fig = go.Figure(data=[go.Surface(z=Z, x=P, y=T, colorscale='Viridis')])
        
        fig.update_layout(
            title=f"Surface du Gamma pour {ticker_input} (Strike @ {K:.2f})",
            scene=dict(
                xaxis_title="Prix du Sous-jacent",
                yaxis_title="Temps restant (Années)",
                zaxis_title="Gamma"
            ),
            width=900,
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""> **Note :** Le Gamma est maximal lorsque l'option est "At-the-money" (proche du prix actuel) et que l'échéance approche.""")

# --- ONGLET 2 : ANALYSE IA ---
with tab_analysis:
    st.header("Analyse prédictive via Gemini")
    prompt = st.text_area("Posez une question sur ces données :", 
                         value=f"Analyse l'impact d'une hausse de la volatilité sur le Gamma de {ticker_input} à court terme.")
    
    if st.button("Lancer l'analyse IA"):
        st.write("*(Note: Connectez votre API Key Gemini pour activer cette section)*")
        # Ici on intégrerait : model.generate_content(prompt)
