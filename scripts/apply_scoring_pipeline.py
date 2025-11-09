"""
    1. Loads the merged_es_imputed.csv data
    2. Preprocesses and creates derived features
    3. Applies economic and social scoring functions
    4. Runs PCA analysis
    5. Saves the enriched dataset with all scores
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
from pathlib import Path
from time import time
import argparse as ap
from clean_and_normalize_scores import check_and_fix_data_quality

from scores import calculate_economic_score, calculate_social_score, calculate_total_score, run_pca_scoring 


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} municipalities")
    return df


def preprocess_data(df):
    """
    Preprocess the data to create features needed for scoring functions.
    
    Maps CSV columns to the required format for scoring functions:
    - avg_income: from renta_bruta_media (in thousands €)
    - population_density: from densidad
    - avg_rent_price: average of alquiler_m2_colectiva and alquiler_m2_unifamiliar (€/m²)
    - normalized_bank_count: num_bancos normalized to [0-1] range
    - perc_over_65: percentage of population 65 years and older
    - perc_under_35: percentage of population under 35 years (estimated from menor_35)
    """
    print("\nPreprocessing data...")
    
    df_processed = df.copy()
    
    numeric_columns = [
        'poblacion_total', 'densidad', 'renta_bruta_media', 
        'alquiler_m2_colectiva', 'alquiler_m2_unifamiliar', 'num_bancos',
        'poblacion_menor_35', 'poblacion_menor_65', 'poblacion_65_mas'
    ]
    
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    critical_cols = ['poblacion_total', 'densidad', 'renta_bruta_media', 'num_bancos']
    initial_count = len(df_processed)
    df_processed = df_processed.dropna(subset=critical_cols)
    removed_count = initial_count - len(df_processed)
    if removed_count > 0:
        print(f"  Removed {removed_count} rows with missing critical data")
    
    df_processed['avg_income'] = df_processed['renta_bruta_media']
    
    df_processed['population_density'] = df_processed['densidad']
    
    df_processed['alquiler_m2_colectiva'] = df_processed['alquiler_m2_colectiva'].fillna(0)
    df_processed['alquiler_m2_unifamiliar'] = df_processed['alquiler_m2_unifamiliar'].fillna(0)
    
    df_processed['avg_rent_price'] = (
        df_processed['alquiler_m2_colectiva'] + df_processed['alquiler_m2_unifamiliar']
    ) / 2
    
    # Calculate median from actual non-zero values in the dataset
    median_rent = df_processed[df_processed['avg_rent_price'] > 0]['avg_rent_price'].median()
    
    # Only use fallback if dataset has no valid rent values at all
    if pd.isna(median_rent) or median_rent == 0:
        print("  Warning: No valid rent prices found, using Spanish national average fallback")
        median_rent = 10.5  # Spanish national average ~10.5 €/m² as of 2024
    
    df_processed['avg_rent_price'] = df_processed['avg_rent_price'].replace(0, median_rent)
    
    max_banks = df_processed['num_bancos'].max()
    df_processed['normalized_bank_count'] = df_processed['num_bancos'] / max_banks
    
    # 5. Calculate percentage over 65
    df_processed['perc_over_65'] = df_processed['poblacion_65_mas'] / df_processed['poblacion_total']
    
    # 6. Calculate percentage under 30 (approximate from menor_35)
    # Fill NaN in poblacion_menor_35 with a proportion based on total population
    median_ratio_under_35 = (df_processed['poblacion_menor_35'] / df_processed['poblacion_total']).median()
    df_processed['poblacion_menor_35'] = df_processed['poblacion_menor_35'].fillna(
        df_processed['poblacion_total'] * median_ratio_under_35
    )
    
    df_processed['perc_under_35'] = df_processed['poblacion_menor_35'] / df_processed['poblacion_total']

    # Handle any NaN or infinite values
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    
    # For rows with missing age data, use median values
    df_processed['perc_over_65'] = df_processed['perc_over_65'].fillna(df_processed['perc_over_65'].median())
    df_processed['perc_under_35'] = df_processed['perc_under_35'].fillna(df_processed['perc_under_35'].median())
    
    print("Preprocessing complete")

    return df_processed


def apply_economic_scoring(df, base_office_size=100, employee_cost_per_user=50, 
                          infrastructure_penalty=0.3, operational_cost_multiplier=1.5):
    """
    Apply economic scoring to all municipalities.
    """
    print("\nCalculating economic scores...")
    
    df['economic_score'] = df.apply(
        lambda row: calculate_economic_score(
            row,
            base_office_size=base_office_size,
            employee_cost_per_user=employee_cost_per_user,
            infrastructure_penalty=infrastructure_penalty,
            operational_cost_multiplier=operational_cost_multiplier
        ),
        axis=1
    )
    
    print(f"  Range: {df['economic_score'].min():.2f} - {df['economic_score'].max():.2f} (mean: {df['economic_score'].mean():.2f})")
    
    return df


def apply_social_scoring(df):
    """
    Apply social scoring to all municipalities.
    """
    print("\nCalculating social scores...")
    
    df['social_score'] = df.apply(calculate_social_score, axis=1)
    
    print(f"  Range: {df['social_score'].min():.4f} - {df['social_score'].max():.4f} (mean: {df['social_score'].mean():.4f})")
    
    return df


def apply_total_scoring(df, alpha=0.5):
    """
    Calculate total scores for different alpha values.
    """
    df[f'total_score_alpha_{int(alpha*100)}'] = calculate_total_score(
        df['economic_score'], 
        df['social_score'], 
        alpha
    )
    
    return df


def apply_pca_analysis(df):
    """
    Apply PCA to extract principal components representing economic and social dimensions.
    """
    print("\nRunning PCA analysis...")
    
    df = run_pca_scoring(df)
    
    print("PCA analysis complete")

    return df


def save_results(df, output_path):
    """
    Save the enriched dataset with all scores.
    """
    print(f"\nSaving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records with {len(df.columns)} columns")


def generate_summary_stats(df):
    """
    Generate and display summary statistics.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    score_columns = ['economic_score', 'social_score'] + [col for col in df.columns if 'total_score' in col]
    
    summary = df[score_columns].describe()
    print(summary.round(4))
    
    # Top 10 municipalities by different scores
    print("\nTop 10 by Economic Score:")
    top_economic = df.nlargest(10, 'economic_score')[
        ['municipio', 'poblacion_total', 'num_bancos', 'economic_score']
    ]
    print(top_economic.to_string(index=False))
    
    print("\nTop 10 by Social Score:")
    top_social = df.nlargest(10, 'social_score')[
        ['municipio', 'poblacion_total', 'num_bancos', 'social_score']
    ]
    print(top_social.to_string(index=False))
    
    # Municipalities with best balance (assuming alpha=0.5)
    if 'total_score_alpha_50' in df.columns:
        print("\nTop 10 by Balanced Score (alpha=0.5):")
        top_balanced = df.nlargest(10, 'total_score_alpha_50')[
            ['municipio', 'poblacion_total', 'num_bancos', 'total_score_alpha_50']
        ]
        print(top_balanced.to_string(index=False))
        
    #Top 10 municipes with the combined PCA score for economic and social
    if 'PC1_economic' in df.columns and 'PC2_social' in df.columns:
        print("\nTop 10 by Combined PCA Score:")
        df['combined_pca_score'] = df['PC1_economic'] + df['PC2_social']
        top_pca = df.nlargest(10, 'combined_pca_score')[
            ['municipio', 'poblacion_total', 'num_bancos', 'combined_pca_score']
        ]
        print(top_pca.to_string(index=False))

def main(data_path=None, output_path=None):
    print("\n" + "="*80)
    print("BRANCH LOCATION SCORING PIPELINE")
    print("="*80)
    
    try:

        df = load_data(data_path)
        df = preprocess_data(df)
        
        df = apply_economic_scoring(
            df,
            base_office_size=100,
            employee_cost_per_user=50,
            infrastructure_penalty=0.3,
            operational_cost_multiplier=1.5
        )
        df = apply_social_scoring(df)
        
        # Calculate total scores for all alpha values (needed by Streamlit and cleaning script)
        print("\nCalculating total scores for all alpha values...")
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            df = apply_total_scoring(df, alpha=alpha)
        print("  Calculated scores for alpha: 0.0, 0.25, 0.5, 0.75, 1.0")
        
        df = apply_pca_analysis(df)
        
        save_results(df, output_path)
        
        generate_summary_stats(df)
        
        print("\n" + "="*80)
        print("Pipeline completed successfully")
        print("="*80)
        print(f"\nOutput file: {output_path}")
        print("Added columns: economic_score, social_score, total_score_alpha_[0,25,50,75,100], PC1_economic, PC2_social\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Apply scoring pipeline to municipalities data.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument("--out-path", type=str, required=True, help="Path to save the output scored CSV file.")
    args = parser.parse_args()
    main(data_path=args.data_path, output_path="out/temp.csv")
    check_and_fix_data_quality(input_path="out/temp.csv", output_path=args.out_path)
