# utils/infographic.py
# Genera un infográfico compacto por municipio/área y lo devuelve como PNG (bytes)
from __future__ import annotations
import io
import pandas as pd
import matplotlib.pyplot as plt

RADAR_FEATURES = [
    ("n_acceso", "Acceso"),
    ("n_demanda", "Demanda"),
    ("n_coste", "Coste(↓)"),
    ("n_competencia", "Compet(↓)"),
    ("n_impacto", "Impacto"),
    ("n_densidad", "Densidad"),
    ("n_renta_m2_mediana_aprox", "Alquiler(↓)"),
    ("n_crec_poblacional_pct", "Crec."),
]

def _get(df: pd.DataFrame, row, col, default=float("nan")):
    return float(row.get(col, default)) if col in df.columns else default

def make_infographic(row: pd.Series, title: str = "") -> bytes:
    # figura
    fig = plt.figure(figsize=(6, 6), dpi=180)
    fig.suptitle(title or str(row.get("municipio", "")), fontsize=12)

    # --- panel 1: radar simplificado ---
    try:
        import numpy as np
        labels = [lbl for _, lbl in RADAR_FEATURES if f"{_}" in row.index or True]
        values = [float(row.get(col, 0.0)) for col, _ in RADAR_FEATURES]
        N = len(values)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax1 = fig.add_axes([0.08, 0.12, 0.42, 0.42], polar=True)
        ax1.plot(angles, values)
        ax1.fill(angles, values, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([lbl for _, lbl in RADAR_FEATURES], fontsize=8)
        ax1.set_yticklabels([])
        ax1.set_title("Perfil normalizado", pad=14, fontsize=10)
    except Exception:
        pass

    # --- panel 2: métricas clave ---
    ax2 = fig.add_axes([0.56, 0.12, 0.36, 0.42])
    keys = [
        ("IOF", "IOF"),
        ("renta_m2_mediana_aprox", "Alquiler €/m²"),
        ("renta_bruta_media", "Renta bruta media"),
        ("densidad", "Densidad hab/km²"),
        ("crec_poblacional_pct", "Crec. anual %"),
        ("num_bancos", "Nº bancos"),
    ]
    text = []
    for k, label in keys:
        val = row.get(k, None)
        if pd.notnull(val):
            try:
                v = float(val)
                text.append(f"{label}: {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            except Exception:
                text.append(f"{label}: {val}")
    ax2.axis("off")
    ax2.text(0, 1, "\n".join(text), va="top", fontsize=9)

    # --- pie de página ---
    ax3 = fig.add_axes([0.06, 0.58, 0.86, 0.33])
    ax3.axis("off")
    nombre = row.get("municipio", "—")
    codigo = row.get("codigo_ine", row.get("codigo_ine_corto", "—"))
    ax3.text(0, 0.8, f"{nombre} (INE: {codigo})", fontsize=11, weight="bold")
    ax3.text(0, 0.55, "Resumen ejecutivo generado automáticamente.", fontsize=9)

    # export
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
