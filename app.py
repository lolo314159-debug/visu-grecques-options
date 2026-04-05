import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="OptiStrat Pro", layout="wide")

# --- FONCTIONS BLACK-SCHOLES ---
def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return np.maximum(0, S - K) if option_type == "call" else np.maximum(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0: T = 1e-6
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) / 365
    return delta, gamma, theta, vega

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        cp = tk.history(period="1d")['Close'].iloc[-1]
        vol = tk.history(period="1y")['Close'].pct_change().std() * np.sqrt(252)
        return {"price": cp, "vol": vol, "name": tk.info.get('longName', ticker)}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌐 Global")
    ticker = st.text_input("Ticker", "AAPL").upper()
    data = get_market_data(ticker)
    s_curr = st.number_input("Prix actuel", value=float(data['price']) if data else 150.0)
    vol_glob = st.slider("Volatilité (%)", 5, 150, int(data['vol']*100) if data else 25) / 100
    rate = st.slider("Taux (%)", 0.0, 10.0, 4.3) / 100

# --- ONGLET STRATÉGIES ---
tabs = st.tabs(["📉 Stratégies P&L", "🔺 Grecques 3D", "🤖 Analyse IA"])

with tabs[0]:
    st.subheader("Simulateur de Stratégies Multi-Jambes")
    
    col1, col2 = st.columns([1, 3])
    
    legs = []
    with col1:
        st.write("**Configuration des jambes (max 4)**")
        t_days_sim = st.slider("Visualiser à (jours restants)", 1, 365, 30)
        
        for i in range(4):
            with st.expander(f"Jambe {i+1}", expanded=(i==0)):
                c1, c2 = st.columns(2)
                with c1:
                    l_type = st.selectbox(f"Type", ["Call", "Put"], key=f"t{i}")
                    l_side = st.selectbox(f"Action", ["Achat", "Vente"], key=f"s{i}")
                with c2:
                    l_strike = st.number_input(f"Strike", value=float(s_curr), key=f"k{i}")
                    l_qty = st.number_input(f"Quantité", value=0, key=f"q{i}")
                
                if l_qty > 0:
                    legs.append({
                        "type": l_type.lower(),
                        "side": 1 if l_side == "Achat" else -1,
                        "strike": l_strike,
                        "qty": l_qty
                    })

    with col2:
        if not legs:
            st.info("Ajoutez une quantité à au moins une jambe pour voir le graphique.")
        else:
            # Calcul du P&L
            s_range = np.linspace(s_curr * 0.5, s_curr * 1.5, 100)
            
            pnl_at_exp = np.zeros_like(s_range)
            pnl_t_days = np.zeros_like(s_range)
            
            total_cost = 0
            for l in legs:
                # Calcul du coût initial (Premium) au prix actuel
                price_now = bs_price(s_curr, l['strike'], t_days_sim/365, rate, vol_glob, l['type'])
                total_cost += price_now * l['qty'] * l['side']
                
                # Valeur à l'échéance (T=0)
                val_exp = np.array([bs_price(s, l['strike'], 0, rate, vol_glob, l['type']) for s in s_range])
                pnl_at_exp += (val_exp * l['qty'] * l['side'])
                
                # Valeur à T jours
                val_t = np.array([bs_price(s, l['strike'], t_days_sim/365, rate, vol_glob, l['type']) for s in s_range])
                pnl_t_days += (val_t * l['qty'] * l['side'])

            # Ajustement pour afficher le PROFIT (Valeur finale - Coût initial)
            pnl_at_exp -= total_cost
            pnl_t_days -= total_cost

            fig_pnl = go.Figure()
            # Ligne Zéro
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            # Courbe à l'échéance
            fig_pnl.add_trace(go.Scatter(x=s_range, y=pnl_at_exp, name="À l'échéance (T=0)", line=dict(color='cyan', width=3)))
            # Courbe à T jours
            fig_pnl.add_trace(go.Scatter(x=s_range, y=pnl_t_days, name=f"À {t_days_sim} jours", line=dict(color='magenta', width=2, dash='dot')))
            
            fig_pnl.update_layout(
                title=f"Profil de Risque (P&L) - {ticker}",
                xaxis_title="Prix du Sous-jacent ($)",
                yaxis_title="Profit / Perte ($)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

# --- REPRISE DES GRECQUES 3D ---
with tabs[1]:
    # (Ici on place le code des surfaces 3D du message précédent)
    st.write("Visualisation des surfaces 3D (voir onglet précédent)")
    # ... (code compute_greek_surface et render_3d ici)

# --- ANALYSE IA ---
with tabs[2]:
    st.subheader("Analyse de la stratégie par Gemini")
    if st.button("Analyser ma stratégie"):
        config_desc = ", ".join([f"{l['qty']}x {l['type']} {l['side']} @ {l['strike']}" for l in legs])
        prompt = f"Analyse cette stratégie d'options sur {ticker} : {config_desc}. Le prix actuel est {s_curr}. Explique le risque max, le gain max et le point mort."
        # Appel API Gemini...
        st.write("*(Connectez votre clé API pour l'analyse en temps réel)*")
