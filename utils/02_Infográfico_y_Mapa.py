# pages/02_Infográfico_y_Mapa.py
# Página Streamlit: pesos ajustables, top-3 por IOF, infográficos y mapa de España
from __future__ import annotations
import base64
import io
import json
import pandas as pd
import streamlit as st
from utils.scoring import compute_iof
from utils.infographic import make_infographic

st.set_page_config(page_title="Infográfico & Mapa", layout="wide")

st.sidebar.header("Pesos del IOF")
default_weights = {
    "acceso": 1.0,
    "demanda": 1.0,
    "impacto": 1.0,
    "coste": 1.0,
    "competencia": 0.75,
    "renta_m2_mediana_aprox": 0.75,
    "densidad": 0.5,
    "poblacion": 0.5,
    "renta_bruta_media": 0.5,
    "crec_poblacional_pct": 0.75,
    "num_bancos": 0.5,
    "num_atm": 0.25,
    "pct_mayores_65": 0.4,
    "desierto_financiero": 0.75,
}
weights = {}
for k, v in default_weights.items():
    weights[k] = st.sidebar.slider(k, 0.0, 2.0, float(v), 0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Ajusta los pesos para ver cómo cambian los resultados.")

st.title("Infográfico y Mapa Interactivo (España)")
st.write(
    "Este módulo calcula el **IOF** a partir del dataset unificado existente, "
    "muestra los **3 mejores municipios** y despliega un **mapa interactivo** con infográficos embebidos."
)

# === Carga del dataset unificado ===
# Usa el archivo que ya generas en tu repo (cámbialo si tu output tiene otro nombre/ruta)
@st.cache_data
def load_merged():
    # Acepta CSV/Parquet según tu pipeline actual
    try:
        df = pd.read_parquet("data/merged.parquet")
    except Exception:
        df = pd.read_csv("data/merged.csv")
    return df

df_raw = load_merged()
st.success(f"Filas en dataset unificado: {len(df_raw):,}")

# === Cálculo IOF ===
df = compute_iof(df_raw, weights=weights)
if "municipio" not in df.columns:
    st.warning("No se encontró columna 'municipio'. Se usará un identificador genérico.")
    df["municipio"] = df.index.astype(str)

# Orden y top-3
df_sorted = df.sort_values("IOF", ascending=False).reset_index(drop=True)
top_n = st.number_input("¿Cuántos mostrar en el ranking?", 3, 50, 3)
top = df_sorted.head(int(top_n))

# === Panel de ranking + infográficos ===
col_rank, col_figs = st.columns([0.35, 0.65])

with col_rank:
    st.subheader("Ranking")
    st.dataframe(top[["municipio", "IOF"] + [c for c in df.columns if c.startswith("n_")]][:top_n])

with col_figs:
    st.subheader("Infográficos de los mejores")
    figs_png = []
    for _, row in top.iterrows():
        title = f"{row.get('municipio','—')} — IOF {row.get('IOF',0):.3f}"
        png = make_infographic(row, title=title)
        figs_png.append(png)
        st.image(png, caption=title, use_column_width=True)

# === Mapa interactivo (folium) que embebe infográficos del top-3 ===
st.subheader("Mapa interactivo de España (click para ver infográfico)")
try:
    import folium
    from folium import IFrame
    from streamlit_folium import st_folium
    # Centro aproximado España
    m = folium.Map(location=[40.3, -3.7], zoom_start=5, tiles="cartodbpositron")

    # vinculamos por municipio usando centroides si existen
    lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude", "y")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("lon", "lng", "long", "x")), None)

    for i, row in top.iterrows():
        # HTML con imagen base64
        png = figs_png[i] if i < len(figs_png) else make_infographic(row)
        b64 = base64.b64encode(png).decode("ascii")
        html = f'<img src="data:image/png;base64,{b64}" width="380"/>'
        iframe = IFrame(html, width=400, height=420)

        # fallback de coordenadas: si no hay lat/lon, colocamos en el centro y evitamos error
        lat = float(row.get(lat_col, 40.3)) if lat_col else 40.3
        lon = float(row.get(lon_col, -3.7)) if lon_col else -3.7

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(iframe, max_width=450),
            tooltip=f"{row.get('municipio','—')} — IOF {row.get('IOF',0):.3f}",
            icon=folium.Icon(icon="info-sign"),
        ).add_to(m)

    st_folium(m, height=540, width=None)
except Exception as e:
    st.error(f"No se pudo renderizar el mapa: {e}")
    st.info("Instala dependencias: `pip install folium streamlit-folium`")

st.caption("Nota: si usas shapes/centroides por municipio, añade columnas `lat` y `lon` en el merged.")
