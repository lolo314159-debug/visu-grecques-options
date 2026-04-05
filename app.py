import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import google.generativeai as genai

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="OptiVisualizer Pro 2026", layout="wide")

# --- MOTEUR DE CALCUL BLACK-SCHOLES ---
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

# --- ACCÈS DONNÉES MARCHÉ (CACHE) ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="5d")
        if hist.empty: return None
        cp = hist['Close'].iloc[-1]
        # Volatilité historique simplifiée
        hist_year = tk.history(period="1y")['Close']
        vola = hist_year.pct_change().std() * np.sqrt(252)
        return {"price": cp, "vol": vola, "name": tk.info.get('longName', ticker_symbol)}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Paramètres Marché")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    market = get_market_data(ticker)
    
    if market:
        st.success(f"{market['name']}")
        s_curr = st.number_input("Prix Sous-jacent ($)", value=float(market['price']))
        vol_init = float(market['vol'] * 100)
    else:
        st.error("Ticker non trouvé")
        s_curr = 100.0
        vol_init = 25.0
        
    vol_glob = st.slider("Volatilité Implicite (%)", 5, 150, int(vol_init)) / 100
    rate = st.slider("Taux sans risque (%)", 0.0, 10.0, 4.3) / 100 # Valeur Avril 2026
    st.divider()
    st.caption("Application par IA Collaboratrice")

# --- INTERFACE PRINCIPALE ---
st.title(f"Terminal d'Analyse d'Options : {ticker}")

tabs = st.tabs(["📉 Stratégies P&L", "🔺 Delta", "🧬 Gamma", "⏳ Theta", "🌊 Vega", "🤖 Analyse IA"])

# --- ONGLET 1 : STRATÉGIES MULTI-JAMBES ---
with tabs[0]:
    col_cfg, col_plot = st.columns([1, 2])
    legs = []
    
    with col_cfg:
        st.subheader("Configuration")
        t_sim = st.slider("Jours restants pour la courbe T", 0, 365, 30)
        
        for i in range(4):
            with st.expander(f"Jambe {i+1}", expanded=(i==0)):
                c1, c2 = st.columns(2)
                with c1:
                    l_type = st.selectbox(f"Type", ["Call", "Put"], key=f"t{i}")
                    l_side = st.selectbox(f"Action", ["Achat", "Vente"], key=f"s{i}")
                with c2:
                    l_strike = st.number_input(f"Strike", value=float(s_curr), key=f"k{i}")
                    l_qty = st.number_input(f"Qté", value=0, step=1, key=f"q{i}")
                
                if l_qty > 0:
                    legs.append({"type": l_type.lower(), "side": 1 if l_side == "Achat" else -1, "strike": l_strike, "qty": l_qty})

    with col_plot:
        if legs:
            s_range = np.linspace(s_curr * 0.7, s_curr * 1.3, 100)
            pnl_exp = np.zeros_like(s_range)
            pnl_t = np.zeros_like(s_range)
            total_cost = 0
            
            for l in legs:
                cost = bs_price(s_curr, l['strike'], t_sim/365, rate, vol_glob, l['type'])
                total_cost += cost * l['qty'] * l['side']
                pnl_exp += (np.array([bs_price(s, l['strike'], 0, rate, vol_glob, l['type']) for s in s_range]) * l['qty'] * l['side'])
                pnl_t += (np.array([bs_price(s, l['strike'], t_sim/365, rate, vol_glob, l['type']) for s in s_range]) * l['qty'] * l['side'])
            
            pnl_exp -= total_cost
            pnl_t -= total_cost

            fig_pnl = go.Figure()
            fig_pnl.add_hline(y=0, line_color="white", line_width=1)
            fig_pnl.add_trace(go.Scatter(x=s_range, y=pnl_exp, name="Échéance (T=0)", line=dict(color='#00ffcc', width=3)))
            fig_pnl.add_trace(go.Scatter(x=s_range, y=pnl_t, name=f"À T={t_sim} jours", line=dict(color='#ff00ff', width=2, dash='dot')))
            fig_pnl.update_layout(title="Profil de Profit & Perte", template="plotly_dark", height=500)
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("Configurez au moins une jambe avec une quantité > 0")

# --- ONGLETS 2 À 5 : LES GRECQUES EN 3D ---
# Paramètres de grille communs
p_grid = np.linspace(s_curr * 0.8, s_curr * 1.2, 30)
d_grid = np.linspace(1, 365, 30)
P, D = np.meshgrid(p_grid, d_grid)

def render_greek_3d(greek_idx, title, label, color):
    # On fixe le strike pour la 3D au strike de la Jambe 1 ou au prix actuel
    target_k = legs[0]['strike'] if legs else s_curr
    func = np.vectorize(lambda p, d: bs_greeks(p, target_k, d/365, rate, vol_glob, "call")[greek_idx])
    Z = func(P, D)
    fig = go.Figure(data=[go.Surface(z=Z, x=p_grid, y=d_grid, colorscale=color)])
    fig.update_layout(
        title=f"{title} (Strike: {target_k})",
        scene=dict(xaxis_title="Prix ($)", yaxis_title="Jours", zaxis_title=label, yaxis_autorange="reversed"),
        height=600, template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]: render_greek_3d(0, "Surface du Delta", "Delta", "Viridis")
with tabs[2]: render_greek_3d(1, "Surface du Gamma", "Gamma", "Magma")
with tabs[3]: render_greek_3d(2, "Surface du Theta", "Theta", "Thermal")
with tabs[4]: render_greek_3d(3, "Surface du Vega", "Vega", "Cividis")

# --- ONGLET 6 : IA GEMINI ---
with tabs[5]:
    st.subheader("Conseiller IA Stratégique")
    user_q = st.text_area("Question sur votre montage :", value=f"Analyse ma stratégie sur {ticker} avec un strike à {s_curr}. Est-ce risqué si la volatilité augmente de 10% ?")
    
    if st.button("Demander à Gemini"):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            resp = model.generate_content(user_q)
            st.markdown("---")
            st.info(resp.text)
        except:
            st.error("Clé API Gemini absente ou invalide dans les Secrets Streamlit.")
