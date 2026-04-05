import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="OptiVisualizer Pro", layout="wide")

# --- MOTEUR DE CALCUL (BLACK-SCHOLES) ---
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
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))) / 365
    return delta, gamma, theta, vega

# --- CACHE DATA ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="5d")
        if hist.empty: return None
        cp = hist['Close'].iloc[-1]
        vola = tk.history(period="1y")['Close'].pct_change().std() * np.sqrt(252)
        return {"price": cp, "vol": vola, "name": tk.info.get('longName', ticker_symbol)}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    market = get_market_data(ticker)
    s_curr = st.number_input("Prix Sous-jacent ($)", value=float(market['price']) if market else 150.0)
    vol_glob = st.slider("Volatilité (%)", 5, 150, int(market['vol']*100) if market else 25) / 100
    rate = st.slider("Taux (%)", 0.0, 10.0, 3.7) / 100 # Taux US 2026 approx

# --- TABS ---
st.title(f"Terminal d'Analyse : {ticker}")
tabs = st.tabs(["📉 Stratégies P&L", "🔺 Delta", "🧬 Gamma", "⏳ Theta", "🌊 Vega", "🤖 Analyse IA"])

# --- STRATÉGIES P&L ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    legs = []
    with col1:
        t_sim = st.slider("Jours pour courbe T", 0, 365, 30)
        for i in range(4):
            with st.expander(f"Jambe {i+1}", expanded=(i==0)):
                c1, c2 = st.columns(2)
                l_type = c1.selectbox("Type", ["Call", "Put"], key=f"t{i}")
                l_side = c1.selectbox("Action", ["Achat", "Vente"], key=f"s{i}")
                l_strike = c2.number_input("Strike", value=float(s_curr), key=f"k{i}")
                l_qty = c2.number_input("Qté", value=0, key=f"q{i}")
                if l_qty > 0:
                    legs.append({"type": l_type.lower(), "side": 1 if l_side == "Achat" else -1, "strike": l_strike, "qty": l_qty})

    with col2:
        if legs:
            s_range = np.linspace(s_curr * 0.7, s_curr * 1.3, 100)
            pnl_exp, pnl_t, total_cost = np.zeros_like(s_range), np.zeros_like(s_range), 0
            for l in legs:
                cost = bs_price(s_curr, l['strike'], t_sim/365, rate, vol_glob, l['type'])
                total_cost += cost * l['qty'] * l['side']
                pnl_exp += (np.array([bs_price(s, l['strike'], 0, rate, vol_glob, l['type']) for s in s_range]) * l['qty'] * l['side'])
                pnl_t += (np.array([bs_price(s, l['strike'], t_sim/365, rate, vol_glob, l['type']) for s in s_range]) * l['qty'] * l['side'])
            
            fig = go.Figure()
            fig.add_hline(y=0, line_color="white")
            fig.add_trace(go.Scatter(x=s_range, y=pnl_exp - total_cost, name="Échéance", line=dict(color='#00ffcc', width=3)))
            fig.add_trace(go.Scatter(x=s_range, y=pnl_t - total_cost, name=f"T={t_sim}j", line=dict(color='#ff00ff', dash='dot')))
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

# --- GRECQUES 3D (RETOUR AUX COULEURS D'ORIGINE) ---
p_grid = np.linspace(s_curr * 0.8, s_curr * 1.2, 30)
d_grid = np.linspace(1, 365, 30)
P, D = np.meshgrid(p_grid, d_grid)

def plot_3d(idx, title, label, colorscale):
    target_k = legs[0]['strike'] if legs else s_curr
    # Note: On utilise le type de la première jambe ou Call par défaut
    t_opt = legs[0]['type'] if legs else "call"
    func = np.vectorize(lambda p, d: bs_greeks(p, target_k, d/365, rate, vol_glob, t_opt)[idx])
    Z = func(P, D)
    fig = go.Figure(data=[go.Surface(z=Z, x=p_grid, y=d_grid, colorscale=colorscale)])
    fig.update_layout(
        title=title, scene=dict(xaxis_title="Prix", yaxis_title="Jours", zaxis_title=label, yaxis_autorange="reversed"),
        height=700, template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]: plot_3d(0, "Surface du Delta", "Delta", "Viridis")
with tabs[2]: plot_3d(1, "Surface du Gamma", "Gamma", "Magma")
with tabs[3]: plot_3d(2, "Surface du Theta", "Theta", "Thermal")
with tabs[4]: plot_3d(3, "Surface du Vega", "Vega", "Cividis")

# --- IA ---
with tabs[5]:
    st.subheader("Analyse Stratégique Gemini")
    if st.button("Analyser"):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            resp = model.generate_content(f"Analyse ce ticker {ticker} au prix {s_curr}. Volatilité {vol_glob*100}%.")
            st.info(resp.text)
        except: st.error("Vérifiez votre clé API dans les secrets.")
