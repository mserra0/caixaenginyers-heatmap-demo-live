"""
This script applies PCA to the dataset to generate two principal components
that can be used as economic and social scores.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_pca_analysis(df):
    """
    Performs PCA on the input DataFrame and returns the DataFrame with principal components.

    Args:
        df (pd.DataFrame): DataFrame with the input features.

    Returns:
        pd.DataFrame: DataFrame with the original data and the two principal components.
        pd.DataFrame: DataFrame with the component loadings.
    """
    # Select the features for PCA
    features = [
        'poblacion_2023_total',
        'densidad_hab_km2',
        'renta_bruta_media',
        'bancos',
        # Assuming a 'depopulation_risk' can be calculated from population change
        'depopulation_risk',
        # Assuming 'perc_over_65' can be derived or is available
        'perc_over_65'
    ]
    
    # Make sure all feature columns exist
    for feature in features:
        assert feature in df.columns, f"Feature '{feature}' is missing from DataFrame"
            
    X = df[features].values

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1_Economic', 'PC2_Social'])

    # Concatenate with the original DataFrame
    result_df = pd.concat([df, pca_df], axis=1)

    # Get component loadings
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1_Economic', 'PC2_Social'], index=features)

    return result_df, loadings

if __name__ == '__main__':
    # Create a dummy DataFrame since we don't have the real data yet
    data = {
        'municipio': ['A', 'B', 'C', 'D', 'E'],
        'poblacion_2023_total': [10000, 50000, 100000, 2000, 80000],
        'poblacion_2018_total': [9000, 52000, 110000, 2500, 75000],
        'densidad_hab_km2': [100, 500, 1000, 20, 800],
        'renta_bruta_media': [25000, 40000, 50000, 18000, 35000],
        'bancos': [2, 10, 20, 1, 15],
        'perc_over_65': [0.20, 0.15, 0.10, 0.30, 0.12]
    }
    df = pd.DataFrame(data)

    # Calculate depopulation risk as a proxy
    df['depopulation_risk'] = (df['poblacion_2018_total'] - df['poblacion_2023_total']) / df['poblacion_2018_total']

    # Run the PCA analysis
    result_df, loadings = run_pca_analysis(df.copy())

    print("--- Component Loadings ---")
    print("These values show how much each original feature contributes to the new Principal Components.")
    print("PC1 is hypothesized to be the 'Urban Wealth' axis.")
    print("PC2 is hypothesized to be the 'Social Need' axis.")
    print(loadings)
    print("\n--- DataFrame with PCA Scores ---")
    print(result_df.head())

    # Save the results to a CSV file
    result_df.to_csv('municipios_con_pca.csv', index=False)
    print("\nResults saved to 'municipios_con_pca.csv'")

    # Example of how to use the new scores with the alpha slider
    alpha = 0.5
    result_df['total_score'] = alpha * result_df['PC1_Economic'] + (1 - alpha) * result_df['PC2_Social']
    print("\n--- DataFrame with Total Score (alpha=0.5) ---")
    print(result_df[['municipio', 'PC1_Economic', 'PC2_Social', 'total_score']].head())