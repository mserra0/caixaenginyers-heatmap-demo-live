# scripts/make_infographics.py
import os
import pandas as pd
from utils.scoring import compute_iof
from utils.infographic import make_infographic

OUT = "outputs/infographics"
os.makedirs(OUT, exist_ok=True)

try:
    df = pd.read_parquet("data/merged.parquet")
except Exception:
    df = pd.read_csv("data/merged.csv")

df = compute_iof(df)
df = df.sort_values("IOF", ascending=False).reset_index(drop=True)

for i, row in df.head(10).iterrows():
    name = str(row.get("municipio", f"area_{i}")).replace("/", "-").replace("\\", "-")
    png = make_infographic(row, title=f"{name} — IOF {row['IOF']:.3f}")
    with open(f"{OUT}/{i:02d}_{name}.png", "wb") as f:
        f.write(png)

print(f"Generados en: {OUT}")
