import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def calculate_economic_score(data_point, 
                            base_office_size=100,  # m2
                            employee_cost_per_user=50,  # €/user/year
                            infrastructure_penalty=0.3,  # cost multiplier for underserved areas
                            operational_cost_multiplier=1.5):  # general operational costs
    """
    Calculates the Economic Score (ROI-Proxy) for a given data point.
    
    This function models a more realistic ROI by considering:
    1. Potential Revenue: Income potential from the local population
    2. Fixed Costs: Office rent/construction scaled by expected demand
    3. Variable Costs: Staffing costs proportional to user base
    4. Infrastructure Costs: costs in underbanked areas (lack of financial infrastructure)
    5. Market Opportunity: Competition and saturation effects

    """
    clients_proxy = data_point['avg_income'] * data_point['population_density']
    
    economic_activity_factor = 1.0
    if 'perc_under_30' in data_point and 'perc_over_65' in data_point:
        perc_economically_active = 1 - data_point['perc_under_30'] - data_point['perc_over_65']
        economic_activity_factor = max(0.5, perc_economically_active * 1.5)  # Boost active population
    
    potential_revenue = clients_proxy * economic_activity_factor
    
    # === MARKET OPPORTUNITY (Sigmoid Function) ===
    # Models inverted S-curve: sweet spot around 0.3-0.5 saturation
    # - Low saturation (0-0.3): High opportunity but penalized for infrastructure risk
    # - Medium saturation (0.3-0.7): Optimal - validated demand, manageable competition
    # - High saturation (0.7-1.0): Diminishing returns due to market saturation
    
    bank_saturation = data_point['normalized_bank_count']
    
    # Sigmoid parameters tuned for banking competition dynamics
    # Peak opportunity at ~40% saturation, gentle decline on both sides
    steepness = 10  # Controls how sharp the sigmoid curve is
    optimal_saturation = 0.4  # Where market opportunity peaks
    
    # Inverted and shifted to peak at optimal_saturation
    sigmoid_value = 1 / (1 + np.exp(steepness * (bank_saturation - optimal_saturation)))
    
    # Scale sigmoid output to range [0.2, 1.0]
    # 0.2 minimum ensures even saturated markets have some opportunity
    market_opportunity = 0.2 + 0.8 * sigmoid_value
    
    expected_users = data_point['population_density'] * market_opportunity * 0.1  # 10% capture rate
    
    # === COST ESTIMATION ===
    
    # 1. Fixed Costs: Office space
    # Scale office size with expected demand (more users = larger office needed)
    required_office_size = base_office_size * (1 + expected_users / 1000)
    rent_price = data_point['avg_rent_price'] if data_point['avg_rent_price'] > 0 else 1000  # default
    annual_rent_cost = rent_price * required_office_size * 12
    
    # 2. Variable Costs: Staffing
    # More users require more employees
    annual_staffing_cost = employee_cost_per_user * expected_users
    
    # 3. Infrastructure Costs: Higher in underbanked areas
    # Areas with no banks likely lack proper internet, security, ATM networks, etc.
    infrastructure_cost_multiplier = 1.0
    if bank_saturation < 0.2:
        # Financial desert: high infrastructure investment needed
        infrastructure_cost_multiplier = 1 + infrastructure_penalty * (0.2 - bank_saturation) * 5
    elif bank_saturation < 0.5:
        # Some infrastructure exists but still costs
        infrastructure_cost_multiplier = 1 + infrastructure_penalty * (0.5 - bank_saturation) * 2
    
    # 4. Total Operational Costs
    base_operational_cost = (annual_rent_cost + annual_staffing_cost) * operational_cost_multiplier
    total_cost = base_operational_cost * infrastructure_cost_multiplier
    
    # Avoid division by zero
    total_cost = max(total_cost, 1)
    
    # === FINAL ROI CALCULATION ===
    # Revenue efficiency adjusted by market opportunity
    roi = (potential_revenue / total_cost) * market_opportunity
    
    # Normalize to reasonable scale (0-100)
    economic_score = roi * 10
    
    return economic_score

def calculate_social_score(data_point):
    """
    Calculates the Social Score (Sustainable Impact) for a given data point.

    The score models the trade-off between serving the current need (elderly population)
    and ensuring future viability (youth population).
    """
    perc_over_65 = data_point['perc_over_65']
    perc_under_30 = data_point['perc_under_30']
    
    # A location with no youth population should score zero.
    if perc_under_30 == 0:
        return 0

    community_score = perc_over_65 * perc_under_30 
    financial_need = 1 - data_point['normalized_bank_count']

    social_score = community_score * financial_need
    return social_score

def calculate_total_score(economic_score, social_score, alpha):
    """
    Calculates the combined total score based on the economic and social scores and the alpha value.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
        
    total_score = alpha * economic_score + (1 - alpha) * social_score
    return total_score


def run_pca_scoring(df):
    """
    Performs PCA on the input DataFrame and returns the DataFrame with principal components.
    """
    features = [
        'poblacion_2023_total',
        'densidad_hab_km2',
        'renta_bruta_media',
        'bancos',
        'depopulation_risk',
        'perc_over_65',
        'perc_under_30'
    ]
    
    for feature in features:
        assert feature in df.columns, f"Feature '{feature}' is missing from DataFrame"
            
    X = df[features].values

    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1_Economic', 'PC2_Social'])

    result_df = pd.concat([df, pca_df], axis=1)

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1_Economic', 'PC2_Social'], index=features)

    return result_df, loadings


if __name__ == '__main__':
    ##Debuging example
    data = {
        'avg_income': [30000, 40000, 25000],
        'population_density': [5000, 2000, 8000],
        'avg_rent_price': [1200, 1500, 1000],
        'normalized_bank_count': [0.2, 0.5, 0.1],  # 0.1 = financial desert
        'perc_over_65': [0.15, 0.20, 0.10],
        'perc_under_30': [0.25, 0.15, 0.35]
    }
    df = pd.DataFrame(data)

    # Apply the scoring functions to the DataFrame
    # You can customize the cost parameters based on your specific market research
    df['economic_score'] = df.apply(
        lambda row: calculate_economic_score(
            row,
            base_office_size=100,
            employee_cost_per_user=50,
            infrastructure_penalty=0.3,
            operational_cost_multiplier=1.5
        ),
        axis=1
    )
    df['social_score'] = df.apply(calculate_social_score, axis=1)

    # Calculate total score for a given alpha
    alpha = 0.5
    df['total_score'] = calculate_total_score(df['economic_score'], df['social_score'], alpha)

    print("DataFrame with calculated scores:")
    print(df[['avg_income', 'population_density', 'normalized_bank_count', 'economic_score', 'social_score', 'total_score']])
    print("\nDetailed Analysis:")
    for idx, row in df.iterrows():
        print(f"\nLocation {idx + 1}:")
        print(f"  Banking Saturation: {row['normalized_bank_count']:.1%}")
        print(f"  Economic Score: {row['economic_score']:.2f}")
        print(f"  Social Score: {row['social_score']:.4f}")
        print(f"  Total Score (α=0.5): {row['total_score']:.2f}")
