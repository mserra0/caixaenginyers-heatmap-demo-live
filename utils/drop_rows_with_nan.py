import argparse
import pandas as pd
import numpy as np


def main(in_csv: str = "data/merged_es.csv", out_csv: str = "data/merged_es_dropna.csv", treat_unknown_as_nan: bool = True) -> None:
    df = pd.read_csv(in_csv)
    if treat_unknown_as_nan:
        df = df.replace("Unknown", np.nan)
    before = len(df)
    df = df.dropna(how="any")
    after = len(df)
    df.to_csv(out_csv, index=False)
    print(f"OK -> {out_csv} rows: {after} (dropped {before - after})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop rows that contain any NaN (or 'Unknown') values.")
    parser.add_argument("--in", dest="in_csv", default="data/merged_es.csv", help="Input CSV path")
    parser.add_argument("--out", dest="out_csv", default="data/merged_es_dropna.csv", help="Output CSV path")
    parser.add_argument("--keep-unknown", dest="keep_unknown", action="store_true", help="Do not treat 'Unknown' as NaN")
    args = parser.parse_args()
    main(in_csv=args.in_csv, out_csv=args.out_csv, treat_unknown_as_nan=not args.keep_unknown)

