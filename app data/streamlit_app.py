import os
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ===========================================
# Utilidades
# ===========================================
def minmax_norm(s: pd.Series) -> pd.Series:
    vmin, vmax = float(s.min()), float(s.max())
    if vmax == vmin:
        return pd.Series(np.zeros_like(s), index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)

def normalize_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = minmax_norm(out[c].astype(float))
    return out

def project_value(col: pd.Series, years: int, rate: float) -> pd.Series:
    """Proyección simple tipo CAGR/linealidad multiplicativa."""
    return col * ((1.0 + rate) ** years)

def default_rates(scenario: str) -> dict:
    """
    Tasas de evolución por variable según escenario.
    Ajusta estos valores si tienes datos reales.
    Signo:
      - acceso: negativo -> mejora del acceso (menos minutos)
      - demanda, impacto: positivo -> crecen
      - coste, competencia: positivo -> tienden a aumentar
    """
    if scenario == "Optimista":
        return dict(acceso=-0.03, demanda=0.02, coste=0.01, competencia=0.005, impacto=0.01)
    if scenario == "Pesimista":
        return dict(acceso=-0.005, demanda=0.005, coste=0.03, competencia=0.015, impacto=0.0)
    # Base
    return dict(acceso=-0.02, demanda=0.01, coste=0.02, competencia=0.01, impacto=0.005)

def ensure_data(path_csv: str = "data/zones.csv") -> pd.DataFrame:
    """
    Si no existe data real, genera un MOCK alrededor de Barcelona.
    Lat/Lon ~ (41.36–41.44, 2.10–2.21), 800 puntos sintéticos.
    Sustituye este CSV por tus datos reales con columnas:
      lat, lon, acceso, demanda, coste, competencia, impacto
    """
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)

    np.random.seed(7)
    n = 800
    lat = 41.36 + np.random.rand(n) * (41.44 - 41.36)
    lon = 2.10  + np.random.rand(n) * (2.21 - 2.10)

    # Estructura espacial simple para que el heatmap sea "bonito"
    centro_lat, centro_lon = 41.39, 2.17
    dist_centro = np.sqrt((lat - centro_lat)**2 + (lon - centro_lon)**2)

    demanda = 1000*np.exp(-25*dist_centro) + 50*np.random.randn(n) + 500
    demanda = np.clip(demanda, 50, None)

    acceso = 10 + 80*dist_centro + 10*np.random.rand(n)            # minutos al punto financiero
    coste = 10 + 30*np.exp(-20*dist_centro) + 5*np.random.randn(n) # €/m2 (proxy)
    competencia = 5*np.exp(-18*dist_centro) + np.random.rand(n)*2
    impacto = 0.5 + 0.5*np.sin(10*dist_centro) + 0.3*np.random.rand(n)  # indicador social proxy

    df = pd.DataFrame(dict(lat=lat, lon=lon,
                           acceso=acceso, demanda=demanda, coste=coste,
                           competencia=competencia, impacto=impacto))
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df.to_csv(path_csv, index=False)
    return df

def compute_pca_importance(df_norm: pd.DataFrame, feature_cols: list[str], top_k: int = 3) -> list[str]:
    """
    PCA ligera con NumPy (SVD) para evitar dependencias pesadas:
    - Estandariza columnas
    - SVD en la matriz (n_samples x n_features)
    - Importancia por variable = suma de |loading| en los 2 primeros componentes
    """
    X = df_norm[feature_cols].to_numpy(dtype=float)

    # Estandariza: media 0, var 1
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    Xz = (X - mu) / sigma

    # SVD: Xz = U * S * Vt  -> filas de Vt = componentes principales
    _, _, Vt = np.linalg.svd(Xz, full_matrices=False)

    # Toma 2 primeros componentes y calcula "importancia" por variable
    comps = Vt[:2, :]                 # (2 x n_features)
    importance = np.abs(comps).sum(axis=0)  # suma de cargas absolutas
    order = np.argsort(importance)[::-1]
    ranked = [feature_cols[i] for i in order]
    return ranked[:top_k]

def compute_score(df_norm: pd.DataFrame, active_vars: list[str], weights: dict) -> pd.Series:
    """
    Suma ponderada (respetando signo: coste/competencia restan).
    Variables esperadas en df_norm: acceso, demanda, coste, competencia, impacto
    """
    w_acc  = weights["Acceso"]
    w_dem  = weights["Demanda"]
    w_cost = weights["Coste"]
    w_comp = weights["Competencia"]
    w_imp  = weights["Impacto social"]

    acc  = df_norm["acceso"]      if "acceso" in active_vars else 0.0
    dem  = df_norm["demanda"]     if "demanda" in active_vars else 0.0
    cost = df_norm["coste"]       if "coste" in active_vars else 0.0
    comp = df_norm["competencia"] if "competencia" in active_vars else 0.0
    imp  = df_norm["impacto"]     if "impacto" in active_vars else 0.0

    return (w_acc*acc + w_dem*dem - w_cost*cost - w_comp*comp + w_imp*imp)

# ===========================================
# Streamlit UI
# ===========================================
st.set_page_config(page_title="Heatmap Demo — Caixa Enginyers", layout="wide")

st.title("Mapa de calor interactivo — Oportunidad de apertura")
st.caption("Explora el potencial por zona ajustando años, escenario y pesos de las variables.")

# Carga datos (mock si no hay reales)
df_raw = ensure_data("data/zones.csv")

with st.sidebar:
    st.header("Controles")
    years = st.slider("Horizonte (años)", min_value=1, max_value=10, value=3)
    scenario = st.selectbox("Escenario", ["Base", "Optimista", "Pesimista"])
    rates = default_rates(scenario)

    st.markdown("**Pesos del Score** (se renormalizan):")
    w_acc  = st.slider("Acceso (necesidad)", 0.0, 1.0, 0.30)
    w_dem  = st.slider("Demanda",              0.0, 1.0, 0.30)
    w_cost = st.slider("Coste",                0.0, 1.0, 0.20)
    w_comp = st.slider("Competencia",          0.0, 1.0, 0.10)
    w_imp  = st.slider("Impacto social",       0.0, 1.0, 0.10)

    # Renormaliza a suma = 1
    ws = np.array([w_acc, w_dem, w_cost, w_comp, w_imp], dtype=float)
    if ws.sum() == 0:
        ws = np.array([0.30, 0.30, 0.20, 0.10, 0.10], dtype=float)
    ws = ws / ws.sum()
    weights = {
        "Acceso": ws[0], "Demanda": ws[1], "Coste": ws[2], "Competencia": ws[3], "Impacto social": ws[4]
    }

    radius = st.slider("Radio heatmap (metros aprox.)", 50, 500, 200, step=25)

# Proyección por escenario y años
df_proj = df_raw.copy()
for var, rate in rates.items():
    df_proj[var] = project_value(df_proj[var].astype(float), years, rate)

# Normaliza variables (0–1)
feature_cols = ["acceso", "demanda", "coste", "competencia", "impacto"]
df_norm = normalize_df(df_proj, feature_cols)

# Variables prominentes (PCA NumPy) y selección en UI
ranked = compute_pca_importance(df_norm, feature_cols, top_k=3)
with st.sidebar:
    st.divider()
    st.markdown("**Variables prominentes (PCA)**")
    active_vars = st.multiselect(
        "Activar variables",
        options=feature_cols,
        default=list(dict.fromkeys(ranked + ["acceso", "demanda"]))  # asegura Acceso/Demanda
    )

# Calcula Score
df_norm["score"] = compute_score(df_norm, active_vars, weights)

# Tabla Top-N (antes del mapa para feedback rápido)
st.subheader("Top zonas (según Score)")
top_n = st.number_input("¿Cuántas zonas mostrar en el ranking?", min_value=3, max_value=50, value=10, step=1)
df_top = df_norm[["lat", "lon", "score", "acceso", "demanda", "coste", "competencia", "impacto"]].copy()
df_top = df_top.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
st.dataframe(df_top.style.format(precision=3), use_container_width=True)
st.download_button("Descargar ranking (CSV)", df_top.to_csv(index=False).encode("utf-8"), file_name="ranking_top.csv")

# Vista inicial del mapa (centroide del dataset)
center_lat, center_lon = float(df_norm["lat"].mean()), float(df_norm["lon"].mean())

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=df_norm,
    get_position='[lon, lat]',
    aggregation='"MEAN"',
    get_weight="score",
    radiusPixels=int(radius/2),  # aproximación rápida a metros -> px
    threshold=0.05
)

scatter_top_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_top.assign(size=80),
    get_position='[lon, lat]',
    get_radius=80,
    pickable=True,
)

tooltip = {
    "html": "<b>Score:</b> {score}<br/>"
            "<b>Acceso:</b> {acceso}<br/>"
            "<b>Demanda:</b> {demanda}<br/>"
            "<b>Coste:</b> {coste}<br/>"
            "<b>Competencia:</b> {competencia}<br/>"
            "<b>Impacto:</b> {impacto}",
    "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
}

deck = pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=center_lat, longitude=center_lon, zoom=11, pitch=0
    ),
    map_provider="carto",        # evita tokens
    map_style="light",
    layers=[heatmap_layer, scatter_top_layer],
    tooltip=tooltip
)

st.subheader("Mapa de calor")
st.pydeck_chart(deck, use_container_width=True)

# Notas finales
with st.expander("Cómo interpretar el mapa y el Score"):
    st.markdown("""
- **Rojo/amarillo** = mayor oportunidad (Score más alto).
- **Años + Escenario**: simulan evolución de variables (acceso, demanda, coste, competencia, impacto).
- **Pesos**: ajusta prioridades (p. ej., dar más peso a Acceso para maximizar impacto social).
- **Top zonas**: lista para justificar **≥3 ubicaciones** en la demo.
""")
