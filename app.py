
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.express as px

st.set_page_config(page_title="Calculadora VaR", layout="wide")

# ==========================================================
# BLACK-SCHOLES
# ==========================================================

def black_scholes_call(S, K, T, r, sigma):

    if T <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call


def black_scholes_put(S, K, T, r, sigma):

    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return put


# ==========================================================
# VAR
# ==========================================================

def historical_var(returns, confidence_level=0.95):

    return abs(np.percentile(
        returns,
        (1 - confidence_level) * 100
    ))


def parametric_var(std, value, confidence_level=0.95):

    z = norm.ppf(confidence_level)

    return z * std * value


def monte_carlo_var(
    value,
    mu,
    sigma,
    confidence_level=0.95,
    simulations=10000
):

    simulated_returns = np.random.normal(
        mu,
        sigma,
        simulations
    )

    simulated_values = value * (1 + simulated_returns)

    losses = value - simulated_values

    var = np.percentile(
        losses,
        confidence_level * 100
    )

    return var, losses


# ==========================================================
# STATUS
# ==========================================================

def classify_limit(utilization):

    if utilization <= 0.7:
        return "Verde"

    elif utilization <= 1:
        return "Amarelo"

    else:
        return "Vermelho"


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Parâmetros")

confidence_level = st.sidebar.slider(
    "Nível de Confiança",
    0.90,
    0.99,
    0.95
)

horizon = st.sidebar.number_input(
    "Horizonte (dias)",
    min_value=1,
    max_value=30,
    value=1
)

methodology = st.sidebar.selectbox(
    "Metodologia",
    [
        "Histórico",
        "Paramétrico",
        "Monte Carlo"
    ]
)

simulations = st.sidebar.number_input(
    "Simulações Monte Carlo",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)
# ==========================================================
# TÍTULO
# ==========================================================

st.title("📊 Calculadora de VaR")

st.write(
    "Sistema de gestão de risco para mesas de trading."
)

# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

else:

    df = pd.read_csv("positions_example.csv")

# ==========================================================
# POSIÇÕES
# ==========================================================

st.subheader("Posições")

st.dataframe(df)

# ==========================================================
# PRECIFICAÇÃO
# ==========================================================

prices_model = []

for _, row in df.iterrows():

    if row["Tipo"] == "call":

        price = black_scholes_call(
            row["Preco"],
            row["Strike"],
            row["Tempo_Ate_Vencimento"],
            row["Taxa_Livre_Risco"],
            row["Volatilidade"]
        )

    elif row["Tipo"] == "put":

        price = black_scholes_put(
            row["Preco"],
            row["Strike"],
            row["Tempo_Ate_Vencimento"],
            row["Taxa_Livre_Risco"],
            row["Volatilidade"]
        )

    else:

        price = row["Preco"]

    prices_model.append(price)

df["Preco_Modelado"] = prices_model

df["Valor_Posicao"] = (
    df["Preco_Modelado"] * df["Quantidade"]
)

# ==========================================================
# TICKERS
# ==========================================================

tickers = []

for asset in df["Ativo"]:

    if "_" in asset:

        ticker = asset.split("_")[0]

    else:

        ticker = asset

    ticker_sa = ticker + ".SA"

    if ticker_sa not in tickers:

        tickers.append(ticker_sa)

# ==========================================================
# YAHOO FINANCE
# ==========================================================

try:

    data = yf.download(
        tickers,
        period="1y",
        auto_adjust=True
    )

    if isinstance(data.columns, pd.MultiIndex):

        if "Close" in data.columns.get_level_values(0):

            prices = data["Close"]

        else:

            prices = data

    else:

        prices = data

    returns = prices.pct_change().dropna()

except Exception as e:

    st.error(f"Erro ao baixar dados: {e}")

    st.stop()

# ==========================================================
# RETORNOS
# ==========================================================

st.subheader("Retornos")

fig_returns = px.line(
    returns,
    title="Retornos Históricos"
)

st.plotly_chart(
    fig_returns,
    use_container_width=True
)

# ==========================================================
# VAR POR MESA
# ==========================================================

results = []

for mesa in df["Mesa"].unique():

    mesa_df = df[df["Mesa"] == mesa]

    portfolio_value = mesa_df["Valor_Posicao"].sum()

    weights = (
        mesa_df["Valor_Posicao"] / portfolio_value
    )

    asset_names = []

    for asset in mesa_df["Ativo"]:

        if "_" in asset:

            ticker = asset.split("_")[0]

        else:

            ticker = asset

        asset_names.append(ticker + ".SA")

    mesa_returns = returns[asset_names]

    weighted_returns = mesa_returns.dot(
        weights.values
    )

    weighted_returns = (
        weighted_returns * np.sqrt(horizon)
    )

    std = np.std(weighted_returns)

    mu = weighted_returns.mean()

    if methodology == "Histórico":

        var_percent = abs(np.percentile(
            weighted_returns,
            (1 - confidence_level) * 100
        ))

        var_value = (
            var_percent
            * portfolio_value
        )

    elif methodology == "Paramétrico":

        z = norm.ppf(confidence_level)

        var_value = (
            z
            * std
            * portfolio_value
        )

    else:

        simulated_returns = np.random.normal(
            mu,
            std,
            simulations
        )

        simulated_values = (
            portfolio_value
            * (1 + simulated_returns)
        )

        losses = (
            portfolio_value
            - simulated_values
        )

        var_value = np.percentile(
            losses,
            confidence_level * 100
        )

    limit_var = mesa_df["Limite_VaR"].iloc[0]

    utilization = var_value / limit_var

    status = classify_limit(utilization)

    results.append({
        "Mesa": mesa,
        "Valor Carteira": round(portfolio_value, 2),
        "VaR": round(var_value, 2),
        "Limite": round(limit_var, 2),
        "Utilização %": round(utilization * 100, 2),
        "Status": status
    })

# ==========================================================
# RESULTADOS
# ==========================================================

results_df = pd.DataFrame(results)

st.subheader("Resultados")

st.dataframe(results_df)

# ==========================================================
# GRÁFICO VAR
# ==========================================================

fig_var = px.bar(
    results_df,
    x="Mesa",
    y="VaR",
    color="Status",
    title="VaR por Mesa",
    color_discrete_map={
        "Verde": "green",
        "Amarelo": "yellow",
        "Vermelho": "red"
    }
)

st.plotly_chart(
    fig_var,
    use_container_width=True
)

# ==========================================================
# ALERTAS
# ==========================================================

st.subheader("Alertas")

for _, row in results_df.iterrows():

    if row["Status"] == "Vermelho":

        st.error(
            f"🚨 {row['Mesa']} ultrapassou o limite!"
        )

    elif row["Status"] == "Amarelo":

        st.warning(
            f"⚠️ {row['Mesa']} próxima do limite."
        )

    else:

        st.success(
            f"✅ {row['Mesa']} dentro do limite."
        )

# ==========================================================
# RANKING
# ==========================================================

st.subheader("Ranking de Risco")

ranking = results_df.sort_values(
    by="VaR",
    ascending=False
)

st.dataframe(ranking)

# ==========================================================
# DOWNLOAD
# ==========================================================

csv = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Resultados",
    csv,
    "resultados_var.csv",
    "text/csv"
)
