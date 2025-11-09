"""
Script to apply Economic and Social scoring functions to the real Spanish municipality dataset.
Generates 4 new columns: economic_score, social_score, total_score, and optionally PC1/PC2.
"""

import pandas as pd
import numpy as np
from scores import calculate_economic_score, calculate_social_score, calculate_total_score, run_pca_scoring
from pathlib import Path

def prepare_data_for_scoring(df):
    """
    Transforms the raw dataset columns into the format expected by scoring functions.
    
    Mapping:
    - avg_income -> renta_bruta_media (in thousands of euros)
    - population_density -> densidad (hab/km2)
    - avg_rent_price -> average of alquiler_m2_colectiva and alquiler_m2_unifamiliar
    - normalized_bank_count -> calculated from num_bancos
    - perc_over_65 -> calculated from poblacion_65_mas / poblacion_total
    - perc_under_30 -> calculated from poblacion_menor_35 / poblacion_total
    """
    
    df_processed = df.copy()
    
    # 1. Average income (already in thousands)
    df_processed['avg_income'] = df_processed['renta_bruta_media']
    
    # 2. Population density
    df_processed['population_density'] = df_processed['densidad']
    
    # 3. Average rent price (€/m2/month)
    # Take the average of collective and single-family housing, handle zeros
    df_processed['alquiler_colectiva'] = df_processed['alquiler_m2_colectiva'].fillna(0)
    df_processed['alquiler_unifamiliar'] = df_processed['alquiler_m2_unifamiliar'].fillna(0)
    
    # Calculate average, but if both are 0, use a default value
    df_processed['avg_rent_price'] = (df_processed['alquiler_colectiva'] + 
                                       df_processed['alquiler_unifamiliar']) / 2
    
    # For municipalities with no rent data, use median of non-zero values
    median_rent = df_processed[df_processed['avg_rent_price'] > 0]['avg_rent_price'].median()
    df_processed['avg_rent_price'] = df_processed['avg_rent_price'].replace(0, median_rent)
    
    # 4. Normalized bank count (0 to 1 scale)
    # We'll use a log-based normalization to handle the wide range of values
    # Formula: log(1 + num_bancos) / log(1 + max_bancos)
    max_bancos = df_processed['num_bancos'].max()
    df_processed['normalized_bank_count'] = (
        np.log1p(df_processed['num_bancos']) / np.log1p(max_bancos)
    )
    
    # 5. Percentage over 65
    df_processed['perc_over_65'] = (
        df_processed['poblacion_65_mas'] / df_processed['poblacion_total']
    ).fillna(0)
    
    # 6. Percentage under 30 (using poblacion_menor_35 as proxy)
    # Note: We'll assume roughly 85% of <35 are <30 based on demographic distributions
    df_processed['perc_under_30'] = (
        (df_processed['poblacion_menor_35'] * 0.85) / df_processed['poblacion_total']
    ).fillna(0)
    
    return df_processed

def apply_scores(df, alpha=0.5, verbose=True):
    """
    Apply economic, social, and total scores to the dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with Spanish municipality data
    alpha : float
        Weight for economic score in total score (0-1)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    DataFrame with new columns: economic_score, social_score, total_score
    """
    
    if verbose:
        print("🔄 Preparing data for scoring...")
    
    df_scored = prepare_data_for_scoring(df)
    
    if verbose:
        print(f"📊 Dataset size: {len(df_scored)} municipalities")
        print(f"📈 Bank count range: {df_scored['num_bancos'].min():.0f} to {df_scored['num_bancos'].max():.0f}")
        print(f"📍 Density range: {df_scored['densidad'].min():.1f} to {df_scored['densidad'].max():.1f} hab/km²")
        print(f"💰 Income range: {df_scored['renta_bruta_media'].min():.1f}k to {df_scored['renta_bruta_media'].max():.1f}k €")
        print()
    
    if verbose:
        print("🎯 Calculating Economic Scores...")
    
    # Apply economic scoring
    df_scored['economic_score'] = df_scored.apply(
        lambda row: calculate_economic_score(
            row,
            base_office_size=100,
            employee_cost_per_user=50,
            infrastructure_penalty=0.3,
            operational_cost_multiplier=1.5
        ),
        axis=1
    )
    
    if verbose:
        print("🤝 Calculating Social Scores...")
    
    # Apply social scoring
    df_scored['social_score'] = df_scored.apply(calculate_social_score, axis=1)
    
    if verbose:
        print(f"⚖️  Calculating Total Scores (α={alpha})...")
    
    # Calculate total score
    df_scored['total_score'] = calculate_total_score(
        df_scored['economic_score'], 
        df_scored['social_score'], 
        alpha
    )
    
    if verbose:
        print("\n✅ Scoring complete!")
        print("\n📊 Score Statistics:")
        print("-" * 60)
        for score_type in ['economic_score', 'social_score', 'total_score']:
            print(f"\n{score_type.replace('_', ' ').title()}:")
            print(f"  Mean:   {df_scored[score_type].mean():.2f}")
            print(f"  Median: {df_scored[score_type].median():.2f}")
            print(f"  Min:    {df_scored[score_type].min():.2f}")
            print(f"  Max:    {df_scored[score_type].max():.2f}")
            print(f"  Std:    {df_scored[score_type].std():.2f}")
    
    return df_scored

def add_pca_components(df, verbose=True):
    """
    Add PCA components (PC1 and PC2) to the dataset.
    
    Requires these columns to be present:
    - poblacion_total
    - densidad
    - renta_bruta_media
    - num_bancos
    - perc_over_65
    - perc_under_30
    """
    
    if verbose:
        print("\n🔬 Performing PCA Analysis...")
    
    # Prepare data for PCA
    df_pca = df.copy()
    
    # Calculate depopulation risk (negative growth proxy)
    # Using elderly population as a proxy for depopulation risk
    df_pca['depopulation_risk'] = df_pca['perc_over_65']
    
    # Rename columns to match PCA function expectations
    df_pca_input = df_pca.rename(columns={
        'poblacion_total': 'poblacion_2023_total',
        'densidad': 'densidad_hab_km2',
        'num_bancos': 'bancos'
    })
    
    # Run PCA
    try:
        df_with_pca, loadings = run_pca_scoring(df_pca_input)
        
        if verbose:
            print("\n✅ PCA complete!")
            print("\n📊 Principal Component Loadings:")
            print(loadings)
            print("\n💡 Interpretation:")
            print("  PC1_Economic: Captures economic viability factors")
            print("  PC2_Social: Captures social/demographic factors")
        
        return df_with_pca, loadings
    
    except Exception as e:
        print(f"\n❌ PCA failed: {e}")
        print("   Adding PCA columns with NaN values...")
        df['PC1_Economic'] = np.nan
        df['PC2_Social'] = np.nan
        return df, None

def main():
    """Main execution function"""
    
    print("="*60)
    print("🏦 CAIXA ENGINYERS - MUNICIPALITY SCORING SYSTEM")
    print("="*60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "merged_es_imputed.csv"
    print(f"\n📁 Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} municipalities")
    
    # Apply scores
    df_scored = apply_scores(df, alpha=0.5, verbose=True)
    
    # Add PCA components
    df_final, pca_loadings = add_pca_components(df_scored, verbose=True)
    
    # Save results
    output_path = data_path.parent / "merged_es_scored.csv"
    print(f"\n💾 Saving scored data to: {output_path}")
    
    # Select relevant columns for output
    output_columns = [
        'municipio', 'provincia', 'poblacion_total', 'densidad', 
        'renta_bruta_media', 'num_bancos', 
        'perc_over_65', 'perc_under_30',
        'normalized_bank_count',
        'economic_score', 'social_score', 'total_score',
        'PC1_Economic', 'PC2_Social'
    ]
    
    df_output = df_final[output_columns].copy()
    df_output.to_csv(output_path, index=False)
    
    print("✅ Saved!")
    
    # Show top 10 municipalities by total score
    print("\n" + "="*60)
    print("🏆 TOP 10 MUNICIPALITIES BY TOTAL SCORE (α=0.5)")
    print("="*60)
    
    top_10 = df_output.nlargest(10, 'total_score')[
        ['municipio', 'provincia', 'poblacion_total', 'num_bancos', 
         'economic_score', 'social_score', 'total_score']
    ]
    
    print(top_10.to_string(index=False))
    
    # Show municipalities with best economic score
    print("\n" + "="*60)
    print("💰 TOP 10 MUNICIPALITIES BY ECONOMIC SCORE")
    print("="*60)
    
    top_econ = df_output.nlargest(10, 'economic_score')[
        ['municipio', 'provincia', 'poblacion_total', 'num_bancos', 
         'economic_score', 'normalized_bank_count']
    ]
    
    print(top_econ.to_string(index=False))
    
    # Show municipalities with best social score
    print("\n" + "="*60)
    print("🤝 TOP 10 MUNICIPALITIES BY SOCIAL SCORE")
    print("="*60)
    
    top_social = df_output.nlargest(10, 'social_score')[
        ['municipio', 'provincia', 'poblacion_total', 'num_bancos',
         'social_score', 'perc_over_65', 'perc_under_30']
    ]
    
    print(top_social.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ SCORING COMPLETE!")
    print("="*60)
    print(f"\n📊 Output file: {output_path}")
    print(f"📈 Total municipalities scored: {len(df_output)}")
    print("\n💡 Next steps:")
    print("   1. Load merged_es_scored.csv into your Streamlit app")
    print("   2. Create heatmap visualization with total_score")
    print("   3. Add alpha slider to adjust economic vs social weight")
    print("   4. Use PC1/PC2 for additional analysis/visualization")
    
if __name__ == '__main__':
    main()
