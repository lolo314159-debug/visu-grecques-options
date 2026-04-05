import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import google.generativeai as genai

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="OptionVisualizer 3D", layout="wide")

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE CALCUL BLACK-SCHOLES ---
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    T doit être en ANNÉES dans cette fonction.
    """
    if T <= 0: T = 1e-6
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

# --- CACHE DONNÉES YAHOO FINANCE ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="1y")
        if hist.empty: return None
        
        cp = hist['Close'].iloc[-1]
        vola = hist['Close'].pct_change().std() * np.sqrt(252)
        return {"price": cp, "vol": vola, "name": tk.info.get('longName', ticker_symbol)}
    except:
        return None

# --- SIDEBAR : PARAMÈTRES ---
with st.sidebar:
    st.header("🎯 Configuration")
    symbol = st.text_input("Symbole Boursier", value="AAPL").upper()
    
    market = get_market_data(symbol)
    
    if market:
        st.success(f"{market['name']}")
        s_price = st.number_input("Prix actuel ($)", value=float(market['price']))
        vol_init = float(market['vol'] * 100)
    else:
        st.error("Symbole non trouvé. Valeurs par défaut utilisées.")
        s_price = 150.0
        vol_init = 25.0

    strike = st.number_input("Strike (Prix d'exercice)", value=s_price)
    vol = st.slider("Volatilité Implicite (%)", 5.0, 150.0, vol_init) / 100
    rate = st.slider("Taux d'intérêt (%)", 0.0, 10.0, 4.0) / 100
    opt_type = st.selectbox("Type d'option", ["Call", "Put"])

# --- CORPS DE L'APPLI ---
st.title(f"Visualisation des Grecques : {symbol}")

if s_price:
    # 1. Préparation des axes (Prix et Jours)
    prices = np.linspace(s_price * 0.7, s_price * 1.3, 40)
    days = np.linspace(1, 365, 40) # De 1 à 365 jours
    P, D = np.meshgrid(prices, days)

    # 2. Fonction de calcul vectorisée (D/365 pour repasser en années)
    def compute_greek_surface(greek_idx):
        func = np.vectorize(lambda p, d: bs_greeks(p, strike, d/365, rate, vol, opt_type.lower())[greek_idx])
        return func(P, D)

    # 3. Création des Onglets
    t_delta, t_gamma, t_theta, t_vega, t_ai = st.tabs([
        "🔺 Delta", "🧬 Gamma", "⏳ Theta", "🌊 Vega", "🤖 Analyse IA"
    ])

    # Fonction utilitaire pour le rendu Plotly
    def render_3d(Z, title, z_label, colorscale):
        fig = go.Figure(data=[go.Surface(z=Z, x=prices, y=days, colorscale=colorscale)])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Prix ($)",
                yaxis_title="Jours avant échéance",
                zaxis_title=z_label,
                yaxis_autorange="reversed" # On met 0 jour au premier plan
            ),
            height=700
        )
        return fig

    with t_delta:
        st.subheader("Sensibilité au prix (Delta)")
        st.plotly_chart(render_3d(compute_greek_surface(0), "Surface du Delta", "Delta", "Viridis"), use_container_width=True)

    with t_gamma:
        st.subheader("Accélération du Delta (Gamma)")
        st.plotly_chart(render_3d(compute_greek_surface(1), "Surface du Gamma", "Gamma", "Magma"), use_container_width=True)

    with t_theta:
        st.subheader("Érosion temporelle journalière (Theta)")
        st.plotly_chart(render_3d(compute_greek_surface(2), "Surface du Theta", "Theta", "Thermal"), use_container_width=True)

    with t_vega:
        st.subheader("Sensibilité à la Volatilité (Vega)")
        st.plotly_chart(render_3d(compute_greek_surface(3), "Surface du Vega", "Vega", "Cividis"), use_container_width=True)

    with t_ai:
        st.subheader("Analyse stratégique par Gemini")
        
        # Contexte pour l'IA
        current_g = bs_greeks(s_price, strike, 30/365, rate, vol, opt_type.lower())
        
        prompt = st.text_area("Votre question :", 
                             value=f"Analyse le profil de risque pour ce {opt_type} {symbol}. "
                                   f"Prix actuel: {s_price}$, Strike: {strike}$, Vol: {vol*100:.1f}%. "
                                   f"Le Delta est de {current_g[0]:.2f} et le Theta journalier est de {current_g[2]:.4f}$. "
                                   "Quels conseils pour la gestion du risque à 30 jours de l'échéance ?")

        if st.button("Lancer l'Analyse IA"):
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-pro')
                with st.spinner("L'IA étudie les surfaces de risque..."):
                    response = model.generate_content(prompt)
                    st.info("### Rapport d'Analyse Gemini")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Erreur de configuration IA : {e}")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Dernière mise à jour : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}. Modèle Black-Scholes standard.")
