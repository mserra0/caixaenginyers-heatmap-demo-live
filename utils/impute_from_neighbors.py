import math
import unicodedata
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join([c for c in normalized if not unicodedata.combining(c)])


def _move_trailing_article(name: str) -> str:
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
    if raw_name is None:
        return ""
    s = str(raw_name).strip()
    if "-" in s:
        left, right = s.rsplit("-", 1)
        if right.strip().isdigit():
            s = left.strip()
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    s = _move_trailing_article(s)
    s = " ".join(s.split())
    s = _strip_accents(s).lower()
    return s


def to_numeric_or_nan(series: pd.Series) -> pd.Series:
    # Replace string "Unknown" with NaN and coerce to numeric
    s = series.replace("Unknown", np.nan)
    return pd.to_numeric(s, errors="coerce")


def zscore(col: pd.Series) -> pd.Series:
    vals = col.astype(float)
    mu = vals.mean(skipna=True)
    sd = vals.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0] * len(vals), index=vals.index)
    return (vals - mu) / sd


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    feats = []
    for c in feature_cols:
        if c not in df.columns:
            feats.append(pd.Series([np.nan] * len(df), index=df.index))
        else:
            feats.append(zscore(to_numeric_or_nan(df[c])))
    mat = np.vstack([f.fillna(0.0).values for f in feats]).T  # shape: (n_rows, n_features)
    return mat


def nearest_neighbor_impute(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: Optional[str] = None,
    k: int = 1,
) -> pd.Series:
    y = to_numeric_or_nan(df[target_col])
    X = build_feature_matrix(df, feature_cols)

    known_idx = np.where(~y.isna().values)[0]
    unknown_idx = np.where(y.isna().values)[0]

    if len(known_idx) == 0 or len(unknown_idx) == 0:
        return y  # nothing to do

    # Optional within-group filtering (e.g., by province)
    group_vals = df[group_col].astype(str).values if group_col and group_col in df.columns else None

    y_out = y.copy()
    for i in unknown_idx:
        # candidate pool
        cand = known_idx
        if group_vals is not None and isinstance(group_vals[i], str):
            same = np.array([j for j in known_idx if group_vals[j] == group_vals[i]])
            if len(same) > 0:
                cand = same
        if len(cand) == 0:
            # fallback to global knowns
            cand = known_idx
        # distance
        diffs = X[cand] - X[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        order = np.argsort(dists)
        top = order[:k]
        chosen_idx = cand[top]
        # assign mean of neighbors (k=1 means exact nearest)
        y_out.iat[i] = float(np.nanmean(y.iloc[chosen_idx].values))
    return y_out


def main(in_csv: str = "data/merged_es.csv", out_csv: str = "data/merged_es_imputed.csv"):
    df = pd.read_csv(in_csv)
    # Province mapping from population.csv (NAMEUNIT -> PROVINCIA)
    pop = pd.read_csv("data/population.csv")
    pop["municipio_key"] = pop["NAMEUNIT"].apply(normalize_municipality)
    pop_map = pop[["municipio_key", "PROVINCIA"]].rename(columns={"PROVINCIA": "provincia"})

    df["municipio_key"] = df["municipio"].apply(normalize_municipality)
    df = df.merge(pop_map, on="municipio_key", how="left")

    # Identify numeric-like columns to consider for features
    candidate_features = [
        "poblacion_total",
        "hombres",
        "mujeres",
        "densidad",
        "superficie_km2",
        "poblacion_menor_65",
        "poblacion_65_mas",
        "renta_bruta_media",
        "alquiler_m2_colectiva",
        "alquiler_m2_unifamiliar",
        "num_bancos",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    # Choose target columns to impute = all numeric-like columns that contain NaNs/"Unknown"
    targets: List[str] = []
    for c in feature_cols:
        col = to_numeric_or_nan(df[c])
        if col.isna().any():
            targets.append(c)

    if not targets:
        df.to_csv(out_csv, index=False)
        print("OK ->", out_csv, "rows:", len(df), "(no missing values to impute)")
        return

    # Impute per target using nearest neighbor within same province; fallback later via medians
    for target in targets:
        imputed = nearest_neighbor_impute(
            df=df,
            target_col=target,
            feature_cols=[c for c in feature_cols if c != target],
            group_col="provincia",
            k=1,
        )
        df[target] = imputed

    # Final fallback: fill any residual NaNs with province median, then global median
    for target in targets:
        series = to_numeric_or_nan(df[target])
        if series.isna().any():
            if "provincia" in df.columns:
                df[target] = series.fillna(df.groupby("provincia")[target].transform(lambda s: to_numeric_or_nan(s).median()))
            df[target] = to_numeric_or_nan(df[target]).fillna(to_numeric_or_nan(df[target]).median())

    # Keep original string "Unknown" in non-target columns; targets become numeric
    df.to_csv(out_csv, index=False)
    print("OK ->", out_csv, "rows:", len(df))


if __name__ == "__main__":
    main()

