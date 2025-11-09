import unicodedata
import pandas as pd
from typing import Optional, List


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join([c for c in normalized if not unicodedata.combining(c)])


def _move_trailing_article(name: str) -> str:
    """
    Move trailing articles like '..., El' to 'El ...'.
    Also handle Galician 'A', 'O', 'As', 'Os' and Catalan "L'".
    """
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if "," in s:
        base, art = s.rsplit(",", 1)
        article = art.strip()
        base = base.strip()
        if article.lower() in {"el", "la", "los", "las"}:
            return f"{article.title()} {base}"
        if article in {"A", "O", "As", "Os"}:
            return f"{article} {base}"
        if article.startswith("L'") or article.startswith("l'"):
            return f"{article} {base}"
    return s


def normalize_municipality(raw_name: Optional[str]) -> str:
    """
    Build a robust merge key for municipality names across datasets.
    Steps:
      - Keep only the municipality portion before trailing codes (split last '-')
      - Keep only the first variant before '/' if present
      - Move trailing article (', El' → 'El ...')
      - Lowercase, remove accents, collapse whitespace
    """
    if raw_name is None:
        return ""
    s = str(raw_name).strip()

    # Household file style: "Terrassa-08279" → "Terrassa"
    if "-" in s:
        left, right = s.rsplit("-", 1)
        if right.strip().isdigit():
            s = left.strip()

    # Multi-language variants: "Gijón/Xixón" → take first token
    if "/" in s:
        s = s.split("/", 1)[0].strip()

    # Move trailing article: "Pobla de Benifassà, la" → "la Pobla de Benifassà"
    s = _move_trailing_article(s)

    # Normalize whitespace and accents for key
    s = " ".join(s.split())
    s = _strip_accents(s).lower()
    return s


def sum_columns(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([None] * len(df), index=df.index)
    return df[present].sum(axis=1, numeric_only=True)


def load_population() -> pd.DataFrame:
    # population.csv columns (per README): NAMEUNIT (municipio), POB21, HOMBRES, MUJERES, Densidad, Superficie_km2
    pop = pd.read_csv("data/population.csv")
    # Prefer NAMEUNIT as municipality label
    pop["municipio_key"] = pop["NAMEUNIT"].apply(normalize_municipality)
    pop_out = pd.DataFrame({
        "municipio_key": pop["municipio_key"],
        "municipio": pop["NAMEUNIT"],
        "poblacion_total": pop["POB21"],
        "hombres": pop["HOMBRES"],
        "mujeres": pop["MUJERES"],
        "densidad": pop["Densidad"],
        "superficie_km2": pop["Superficie_km2"],
    })
    return pop_out


def load_age_population() -> pd.DataFrame:
    ages = pd.read_csv("data/age_population.csv")
    # Municipality name column appears as "Poblacion_edad_Poblacion_nueva_"
    name_col = "Poblacion_edad_Poblacion_nueva_"
    if name_col not in ages.columns:
        # Fallback: sometimes the 2nd column is the municipality name
        name_col = ages.columns[1]
    ages["municipio_key"] = ages[name_col].apply(normalize_municipality)

    # Sum 65+ → PAD_2C16..PAD_2C20; <65 → PAD_2C03..PAD_2C15; <35 → PAD_2C03..PAD_2C09
    cols_65_plus = [f"PAD_2_MU_2018_PAD_2C{n:02d}" for n in range(16, 21)]
    cols_under_65 = [f"PAD_2_MU_2018_PAD_2C{n:02d}" for n in range(3, 16)]
    cols_under_35 = [f"PAD_2_MU_2018_PAD_2C{n:02d}" for n in range(3, 10)]

    ages_out = pd.DataFrame({
        "municipio_key": ages["municipio_key"],
        "poblacion_65_mas": sum_columns(ages, cols_65_plus),
        "poblacion_menor_65": sum_columns(ages, cols_under_65),
        "poblacion_menor_35": sum_columns(ages, cols_under_35),
    })
    # Some files may contain multiple entries per municipality; aggregate by sum
    ages_out = ages_out.groupby("municipio_key", as_index=False).sum(numeric_only=True)
    return ages_out


def load_rent() -> pd.DataFrame:
    rent = pd.read_csv("data/rent.csv")
    # NMUN: municipality, multiple districts per municipality. Average required.
    rent["municipio_key"] = rent["NMUN"].apply(normalize_municipality)
    grouped = rent.groupby("municipio_key", as_index=False).agg({
        "Renta_Medi": "mean",
        "Renta_Me_1": "mean",
    })
    grouped = grouped.rename(columns={
        "Renta_Medi": "alquiler_m2_colectiva",
        "Renta_Me_1": "alquiler_m2_unifamiliar",
    })
    return grouped


def load_household_income() -> pd.DataFrame:
    hh = pd.read_csv("data/household_municipality.csv")
    # 'Name' looks like "Terrassa-08279" or "\"Ejido, El-04902\""
    hh["MunicipioParsed"] = hh["Name"].astype(str).str.strip().str.replace('"', "", regex=False)
    hh["municipio_key"] = hh["MunicipioParsed"].apply(normalize_municipality)
    out = hh[["municipio_key", "RENTA BRUTA MEDIA"]].rename(columns={
        "RENTA BRUTA MEDIA": "renta_bruta_media"
    })
    # Multiple entries should not exist per muni, but if so, take mean
    out = out.groupby("municipio_key", as_index=False).mean(numeric_only=True)
    return out


def load_banks() -> pd.DataFrame:
    banks = pd.read_csv("data/banks-by-population.csv")
    banks["municipio_key"] = banks["Municipality"].apply(normalize_municipality)
    grouped = banks.groupby("municipio_key", as_index=False).agg(
        num_bancos=("Municipality", "size"),
        longitud_bancos=("Longitude", "mean"),
        latitud_bancos=("Latitude", "mean"),
    )
    return grouped


def main(out_csv: str = "data/merged_es.csv"):
    # Load individual datasets
    pop = load_population()
    ages = load_age_population()
    rents = load_rent()
    hh = load_household_income()
    banks = load_banks()

    # Progressive OUTER merges to keep all municipalities from any source
    merged = pop.merge(ages, on="municipio_key", how="outer")
    merged = merged.merge(rents, on="municipio_key", how="outer")
    merged = merged.merge(hh, on="municipio_key", how="outer")
    merged = merged.merge(banks, on="municipio_key", how="outer")

    # Determine display municipality name: prefer 'municipio' from population, else try to rebuild from keys where possible
    if "municipio" not in merged.columns:
        merged["municipio"] = None

    # Fill missing display name from any available source columns (best-effort: we don't carry them; fallback to key)
    merged["municipio"] = merged["municipio"].fillna(merged["municipio_key"])

    # Reorder columns and fill Unknowns
    desired_order = [
        "municipio",
        "poblacion_total",
        "hombres",
        "mujeres",
        "densidad",
        "superficie_km2",
        "poblacion_menor_35",
        "poblacion_menor_65",
        "poblacion_65_mas",
        "renta_bruta_media",
        "alquiler_m2_colectiva",
        "alquiler_m2_unifamiliar",
        "num_bancos",
        "longitud_bancos",
        "latitud_bancos",
    ]

    # Ensure columns exist even if missing from sources
    for c in desired_order:
        if c not in merged.columns:
            merged[c] = None

    out = merged[["municipio_key"] + desired_order].copy()

    # For bank-related columns, fill missing with 0 instead of "Unknown"
    zero_fill_cols = ["num_bancos", "longitud_bancos", "latitud_bancos"]
    for c in zero_fill_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # Replace NaNs with "Unknown" across all other non-key output columns
    for c in desired_order:
        if c in zero_fill_cols:
            continue
        out[c] = out[c].where(out[c].notna(), "Unknown")

    # Drop key from final output
    out = out.drop(columns=["municipio_key"])

    out.to_csv(out_csv, index=False)
    print("OK ->", out_csv, "rows:", len(out))


if __name__ == "__main__":
    main()
