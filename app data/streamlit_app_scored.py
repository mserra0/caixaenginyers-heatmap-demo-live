"""
Streamlit App for Caixa Enginyers Branch Location Optimization
Using the sophisticated Economic and Social Scoring System
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path
import argparse as ap

# ===========================================
# COMMAND LINE ARGUMENTS
# ===========================================

def parse_arguments():
    """Parse command line arguments."""
    parser = ap.ArgumentParser(description="Streamlit app for branch location optimization")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the scored municipalities CSV file (e.g., municipalities_scored_clean.csv)"
    )
    
    # Parse known args to avoid conflicts with Streamlit's own arguments
    args, unknown = parser.parse_known_args()
    return args

# Parse arguments at module level
ARGS = parse_arguments()

# ===========================================
# DATA LOADING
# ===========================================

@st.cache_data
def load_scored_data(data_path=None):
    """Load the pre-scored municipality data."""
    # Use provided path or default
    if data_path is None:
        if ARGS.data_path:
            data_path = Path(ARGS.data_path)
        else:
            # Default path
            data_path = Path(__file__).parent.parent / "out" / "result.csv"
    
    if not data_path.exists():
        st.error(f"Scored data not found at {data_path}")
        st.info("Run: `python scripts/apply_scoring_pipeline.py` to generate scores")
        st.stop()
    
    df = pd.read_csv(data_path)
    return df


@st.cache_data
def load_geospatial_data():
    """Load geospatial data for municipalities from GeoJSON."""
    geojson_path = Path(__file__).parent.parent / "data" / "citylocation.geojson"
    
    if not geojson_path.exists():
        st.warning(f"GeoJSON file not found at {geojson_path}")
        return pd.DataFrame(columns=['name', 'lat', 'lon'])
    
    try:
        import json
        
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # Extract municipality names and coordinates
        locations = []
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            
            if geom.get('type') == 'Point' and 'coordinates' in geom:
                name = props.get('name', '')
                coords = geom['coordinates']
                
                if name and len(coords) >= 2:
                    locations.append({
                        'name': name,
                        'lon': coords[0],  # GeoJSON is [lon, lat]
                        'lat': coords[1]
                    })
        
        df_geo = pd.DataFrame(locations)
        print(f"Loaded {len(df_geo)} municipalities from GeoJSON")
        return df_geo
        
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        return pd.DataFrame(columns=['name', 'lat', 'lon'])


def merge_geospatial(df, df_geo):
    """Add lat/lon to municipalities by matching names with GeoJSON data."""
    df = df.copy()
    
    if df_geo.empty:
        st.warning("No geospatial data available. Using default coordinates.")
        df['lat'] = 40.4168
        df['lon'] = -3.7038
        return df
    
    # Check if municipio column exists
    if 'municipio' not in df.columns:
        st.error("Error: 'municipio' column not found in data")
        df['lat'] = 40.4168
        df['lon'] = -3.7038
        return df
    
    # Normalize municipality names for better matching
    df['municipio_normalized'] = df['municipio'].str.strip().str.lower()
    df_geo['name_normalized'] = df_geo['name'].str.strip().str.lower()
    
    # Collapse multiple GeoJSON features with the same name to a single row (mean coordinates)
    df_geo_collapsed = (
        df_geo
        .groupby('name_normalized', as_index=False)
        .agg({'lat': 'mean', 'lon': 'mean'})
    )
    
    # Merge on normalized names
    df_merged = df.merge(
        df_geo_collapsed[['name_normalized', 'lat', 'lon']],
        left_on='municipio_normalized',
        right_on='name_normalized',
        how='left'
    )
    
    # Count matches
    matched = df_merged['lat'].notna().sum()
    total = len(df_merged)
    match_rate = (matched / total * 100) if total > 0 else 0
    
    print(f"Matched {matched}/{total} municipalities ({match_rate:.1f}%)")
    
    # For unmatched municipalities, try alternative matching strategies
    unmatched_mask = df_merged['lat'].isna()
    if unmatched_mask.sum() > 0:
        # Try removing common prefixes/suffixes
        for prefix in ['el ', 'la ', 'les ', 'els ', 'los ', 'las ']:
            still_unmatched = df_merged['lat'].isna()
            df_merged.loc[still_unmatched, 'municipio_alt'] = (
                df_merged.loc[still_unmatched, 'municipio_normalized']
                .str.replace(f'^{prefix}', '', regex=True)
            )
            
            # Try to match with alternative name
            for idx in df_merged[still_unmatched].index:
                alt_name = df_merged.loc[idx, 'municipio_alt']
                if pd.notna(alt_name):
                    match = df_geo[df_geo['name_normalized'] == alt_name]
                    if not match.empty:
                        df_merged.loc[idx, 'lat'] = match.iloc[0]['lat']
                        df_merged.loc[idx, 'lon'] = match.iloc[0]['lon']
    
    # Final fallback: use Spain's geographic center for unmatched
    df_merged['lat'] = df_merged['lat'].fillna(40.4168)  # Madrid coordinates
    df_merged['lon'] = df_merged['lon'].fillna(-3.7038)
    
    # Clean up temporary columns
    df_merged = df_merged.drop(columns=['municipio_normalized', 'name_normalized', 'municipio_alt'], errors='ignore')
    
    # Ensure uniqueness by municipio after merge (avoid one-to-many explosions)
    df_merged = df_merged.drop_duplicates(subset=['municipio'], keep='first')
    
    final_matched = (df_merged['lat'] != 40.4168).sum()
    st.info(f"📍 Mapped coordinates for {final_matched}/{total} municipalities ({final_matched/total*100:.1f}%)")
    
    return df_merged


# ===========================================
# SCORING FUNCTIONS
# ===========================================

def get_score_column(alpha):
    """Get the appropriate score column name for given alpha."""
    alpha_int = int(alpha * 100)
    return f'total_score_alpha_{alpha_int}'


def filter_data(df, filters):
    """Apply filters to the dataframe."""
    filtered = df.copy()
    
    # Population filter
    if filters['min_population'] > 0:
        filtered = filtered[filtered['poblacion_total'] >= filters['min_population']]
    if filters['max_population'] < 1000000:
        filtered = filtered[filtered['poblacion_total'] <= filters['max_population']]
    
    # Province filter
    if filters['provinces'] and 'provincia' in filtered.columns:
        filtered = filtered[filtered['provincia'].isin(filters['provinces'])]
    
    # Bank saturation filter
    if filters['max_bank_saturation'] < 1.0:
        filtered = filtered[filtered['normalized_bank_count'] <= filters['max_bank_saturation']]
    
    return filtered


# ===========================================
# VISUALIZATION FUNCTIONS
# ===========================================

def create_color_scale(scores):
    """Create color scale from blue (low) to red (high)."""
    # Normalize scores to 0-1 for color mapping
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return [[100, 100, 100, 200]] * len(scores)
    
    normalized = (scores - min_score) / (max_score - min_score)
    
    colors = []
    for norm_score in normalized:
        # Blue (low) -> Yellow (medium) -> Red (high)
        if norm_score < 0.5:
            # Blue to Yellow
            r = int(255 * (norm_score * 2))
            g = int(255 * (norm_score * 2))
            b = int(255 * (1 - norm_score * 2))
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (norm_score - 0.5) * 2))
            b = 0
        colors.append([r, g, b, 200])
    
    return colors


# ===========================================
# STREAMLIT UI
# ===========================================

st.set_page_config(
    page_title="Caixa Enginyers - Branch Location Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏦 Caixa Enginyers - Branch Location Optimizer")
st.markdown("**Interactive heatmap for optimal bank branch placement in Spain**")
st.caption("Powered by sophisticated Economic and Social Scoring algorithms with sigmoid market opportunity modeling")

# Load data
with st.spinner("Loading scored municipality data..."):
    df = load_scored_data()
    df_geo = load_geospatial_data()
    df = merge_geospatial(df, df_geo)

st.success(f"✅ Loaded {len(df)} municipalities with scoring data")

# ===========================================
# SIDEBAR - MAIN CONTROLS
# ===========================================

with st.sidebar:
    st.header("🎚️ Strategy Control")
    
    # Scoring method toggle
    st.markdown("### **Scoring Method**")
    use_pca = st.toggle(
        "🔬 Use PCA Components (Experimental)",
        value=False,
        help="Toggle to use raw PCA components (PC1=Economic, PC2=Social) instead of sophisticated scoring function"
    )
    
    if use_pca:
        st.info("💡 Using normalized PCA components for real-time calculation")
    else:
        st.info("✨ Using sophisticated scoring function with sigmoid modeling")
    
    st.divider()
    
    # THE KEY CONTROL: Alpha slider
    st.markdown("### **Trade-off: Economic ↔ Social**")
    alpha = st.slider(
        "Strategic Focus",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.0 = Pure Social Impact | 0.5 = Balanced | 1.0 = Pure Economic ROI"
    )
    
    # Visual indicator
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Social Weight", f"{(1-alpha)*100:.0f}%")
    with col2:
        st.metric("Economic Weight", f"{alpha*100:.0f}%")
    
    # Strategy label
    if alpha <= 0.2:
        strategy_label = "🤝 **Pure Social Impact**"
        strategy_color = "blue"
    elif alpha <= 0.4:
        strategy_label = "🌱 **Social-Leaning**"
        strategy_color = "lightblue"
    elif alpha <= 0.6:
        strategy_label = "⚖️ **Balanced** (Recommended)"
        strategy_color = "green"
    elif alpha <= 0.8:
        strategy_label = "💼 **Economic-Leaning**"
        strategy_color = "orange"
    else:
        strategy_label = "💰 **Pure Economic ROI**"
        strategy_color = "red"
    
    st.markdown(f"**Current Strategy:** {strategy_label}")
    
    st.divider()
    
    # ===========================================
    # FILTERS
    # ===========================================
    
    st.header("🔍 Filters")
    
    with st.expander("Population Range", expanded=False):
        min_pop = st.number_input(
            "Minimum Population",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000
        )
        max_pop = st.number_input(
            "Maximum Population",
            min_value=1000,
            max_value=1000000,
            value=1000000,
            step=10000
        )
    
    with st.expander("Geographic Filters", expanded=False):
        if 'provincia' in df.columns:
            all_provinces = sorted(df['provincia'].dropna().unique())
            selected_provinces = st.multiselect(
                "Select Provinces (empty = all)",
                options=all_provinces,
                default=[]
            )
        else:
            st.info("Province data not available in this dataset")
            selected_provinces = []
    
    with st.expander("Market Conditions", expanded=False):
        max_bank_sat = st.slider(
            "Max Bank Saturation",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Filter out municipalities with too many banks"
        )
    
    # Apply filters
    filters = {
        'min_population': min_pop,
        'max_population': max_pop,
        'provinces': selected_provinces,
        'max_bank_saturation': max_bank_sat
    }
    
    df_filtered = filter_data(df, filters)
    
    st.info(f"📊 {len(df_filtered)} municipalities match filters")
    
    st.divider()
    
    # ===========================================
    # VISUALIZATION SETTINGS
    # ===========================================
    
    st.header("🎨 Visualization")
    
    top_n = st.slider(
        "Show Top N Locations",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    radius = st.slider(
        "Point Size",
        min_value=50,
        max_value=500,
        value=150,
        step=25
    )

# ===========================================
# MAIN CONTENT
# ===========================================

# Calculate scores for filtered data
df_filtered = df_filtered.copy()

if use_pca:
    # Use PCA components with real-time alpha calculation
    # Check if PCA columns exist
    if 'PC1_economic' not in df_filtered.columns or 'PC2_social' not in df_filtered.columns:
        st.error("❌ PCA components not found in data. Please run the scoring pipeline first.")
        st.stop()
    
    # Normalize PCA components to 0-100 scale for comparison
    pc1_min = df_filtered['PC1_economic'].min()
    pc1_max = df_filtered['PC1_economic'].max()
    pc2_min = df_filtered['PC2_social'].min()
    pc2_max = df_filtered['PC2_social'].max()
    
    # Avoid division by zero
    pc1_range = pc1_max - pc1_min if pc1_max != pc1_min else 1
    pc2_range = pc2_max - pc2_min if pc2_max != pc2_min else 1
    
    df_filtered['PC1_normalized'] = ((df_filtered['PC1_economic'] - pc1_min) / pc1_range) * 100
    df_filtered['PC2_normalized'] = ((df_filtered['PC2_social'] - pc2_min) / pc2_range) * 100
    
    # Calculate combined PCA score based on alpha
    df_filtered['current_score'] = alpha * df_filtered['PC1_normalized'] + (1 - alpha) * df_filtered['PC2_normalized']
    
    # For metrics display
    df_filtered['economic_score_display'] = df_filtered['PC1_normalized']
    df_filtered['social_score_display'] = df_filtered['PC2_normalized']
    
    scoring_method = "PCA Components"
    
else:
    # Use sophisticated scoring function with real-time alpha calculation
    # Check if required columns exist
    if 'economic_score' not in df_filtered.columns or 'social_score' not in df_filtered.columns:
        st.error("❌ Economic/social scores not found in data. Please run the scoring pipeline first.")
        st.stop()
    
    # Normalize economic and social scores to 0-100 scale
    econ_min = df_filtered['economic_score'].min()
    econ_max = df_filtered['economic_score'].max()
    social_min = df_filtered['social_score'].min()
    social_max = df_filtered['social_score'].max()
    
    # Avoid division by zero
    econ_range = econ_max - econ_min if econ_max != econ_min else 1
    social_range = social_max - social_min if social_max != social_min else 1
    
    df_filtered['economic_score_normalized'] = ((df_filtered['economic_score'] - econ_min) / econ_range) * 100
    df_filtered['social_score_normalized'] = ((df_filtered['social_score'] - social_min) / social_range) * 100
    
    # Calculate combined sophisticated score based on alpha in real-time
    df_filtered['current_score'] = (
        alpha * df_filtered['economic_score_normalized'] + 
        (1 - alpha) * df_filtered['social_score_normalized']
    )
    
    # For metrics display
    df_filtered['economic_score_display'] = df_filtered['economic_score_normalized']
    df_filtered['social_score_display'] = df_filtered['social_score_normalized']
    
    scoring_method = "Sophisticated Scoring"

# Sort by score and ensure unique municipalities (keep best-scoring row)
df_filtered = df_filtered.sort_values('current_score', ascending=False)
df_filtered = df_filtered.drop_duplicates(subset=['municipio'], keep='first')
df_sorted = df_filtered

# ===========================================
# METRICS ROW
# ===========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_economic = df_filtered['economic_score_display'].mean()
    metric_label = "Avg PC1 (Economic)" if use_pca else "Avg Economic Score"
    st.metric(
        metric_label,
        f"{avg_economic:.1f}",
        help="Average economic component across filtered municipalities"
    )

with col2:
    avg_social = df_filtered['social_score_display'].mean()
    metric_label = "Avg PC2 (Social)" if use_pca else "Avg Social Score"
    st.metric(
        metric_label,
        f"{avg_social:.1f}",
        help="Average social component across filtered municipalities"
    )

with col3:
    top_municipality = df_sorted.iloc[0]['municipio'] if len(df_sorted) > 0 else "N/A"
    st.metric(
        "Top Location",
        top_municipality,
        help="Highest scoring municipality with current settings"
    )

with col4:
    financial_deserts = len(df_filtered[df_filtered['normalized_bank_count'] < 0.2])
    st.metric(
        "Financial Deserts",
        financial_deserts,
        help="Municipalities with <20% bank saturation"
    )

st.divider()

# ===========================================
# TOP LOCATIONS TABLE
# ===========================================

st.subheader(f"🏆 Top {top_n} Recommended Locations ({scoring_method})")

# Select columns to display
display_cols = ['municipio']
if 'provincia' in df_sorted.columns:
    display_cols.append('provincia')
display_cols.extend([
    'poblacion_total', 'num_bancos',
    'economic_score_display', 'social_score_display', 'current_score'
])

df_top = df_sorted[display_cols].head(top_n).reset_index(drop=True)
column_names = ['Municipality']
if 'provincia' in df_sorted.columns:
    column_names.append('Province')

# Column names depend on scoring method
if use_pca:
    column_names.extend([
        'Population', 'Banks',
        'PC1 (Economic)', 'PC2 (Social)', 'Combined Score'
    ])
else:
    column_names.extend([
        'Population', 'Banks',
        'Economic Score', 'Social Score', 'Total Score'
    ])
df_top.columns = column_names

# Style the dataframe
format_dict = {
    'Population': '{:,.0f}',
    'Banks': '{:.0f}',
}

if use_pca:
    format_dict.update({
        'PC1 (Economic)': '{:.1f}',
        'PC2 (Social)': '{:.1f}',
        'Combined Score': '{:.1f}'
    })
    gradient_col = 'Combined Score'
else:
    format_dict.update({
        'Economic Score': '{:.1f}',
        'Social Score': '{:.1f}',
        'Total Score': '{:.1f}'
    })
    gradient_col = 'Total Score'

styled_df = df_top.style.format(format_dict).background_gradient(subset=[gradient_col], cmap='RdYlGn')

st.dataframe(styled_df, use_container_width=True, height=400)

# Download button
csv = df_top.to_csv(index=False).encode('utf-8')
method_suffix = "pca" if use_pca else "sophisticated"
st.download_button(
    label="📥 Download Top Locations (CSV)",
    data=csv,
    file_name=f"top_locations_{method_suffix}_alpha_{int(alpha*100)}.csv",
    mime="text/csv"
)

st.divider()

# ===========================================
# INTERACTIVE MAP
# ===========================================

st.subheader("🗺️ Interactive Heatmap")

# Visualization mode selector
viz_mode = st.radio(
    "Visualization Style:",
    options=["Continuous Heatmap", "Points Only", "Heatmap + Points"],
    index=2,
    horizontal=True,
    help="Choose how to display the data on the map"
)

# Prepare data for visualization
df_map = df_sorted.head(min(1000, len(df_sorted))).copy()  # Limit for performance
df_map['color'] = create_color_scale(df_map['current_score'])
df_map['size'] = radius
df_map['weight'] = df_map['current_score']  # For heatmap intensity

# Top locations for highlighting
df_top_map = df_sorted.head(top_n).copy()
df_top_map['size'] = radius * 1.5
df_top_map['weight'] = df_top_map['current_score']

# Calculate map center
center_lat = df_map['lat'].mean()
center_lon = df_map['lon'].mean()

# Create layers based on visualization mode
layers = []

# Continuous heatmap layer using HexagonLayer
if viz_mode in ["Continuous Heatmap", "Heatmap + Points"]:
    heatmap_layer = pdk.Layer(
        "HexagonLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_weight='weight',
        radius=5000,  # 5km hexagon radius
        elevation_scale=50,
        elevation_range=[0, 1000],
        extruded=True,
        coverage=0.95,
        pickable=True,
        auto_highlight=True,
        color_range=[
            [65, 182, 196],    # Light blue (low)
            [127, 205, 187],   # Teal
            [199, 233, 180],   # Light green
            [237, 248, 177],   # Yellow-green
            [255, 255, 204],   # Light yellow
            [255, 237, 160],   # Yellow
            [254, 217, 118],   # Orange-yellow
            [254, 178, 76],    # Orange
            [253, 141, 60],    # Red-orange
            [240, 59, 32],     # Red (high)
        ],
        opacity=0.7 if viz_mode == "Heatmap + Points" else 0.8,
    )
    layers.append(heatmap_layer)

# Individual points layer
if viz_mode in ["Points Only", "Heatmap + Points"]:
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_color='color',
        get_radius='size',
        pickable=True,
        opacity=0.4 if viz_mode == "Heatmap + Points" else 0.6,
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
    )
    layers.append(scatter_layer)

# Top locations layer (always show as gold stars)
top_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_top_map,
    get_position='[lon, lat]',
    get_color=[255, 215, 0, 255],  # Gold
    get_radius='size',
    pickable=True,
    opacity=1.0,
    stroked=True,
    filled=True,
    line_width_min_pixels=2,
)
layers.append(top_layer)

# Tooltip
provincia_line = "<b>Province:</b> {provincia}<br/>" if 'provincia' in df_filtered.columns else ""

if use_pca:
    tooltip = {
        "html": f"""
        <b>{{municipio}}</b><br/>
        {provincia_line}
        <b>Population:</b> {{poblacion_total:,.0f}}<br/>
        <b>Banks:</b> {{num_bancos:.0f}}<br/>
        <b>PC1 (Economic):</b> {{PC1_normalized:.1f}}<br/>
        <b>PC2 (Social):</b> {{PC2_normalized:.1f}}<br/>
        <b>Combined Score:</b> {{current_score:.1f}}
        """,
        "style": {
            "backgroundColor": "rgba(30,30,30,0.95)",
            "color": "white",
            "fontSize": "12px",
            "padding": "10px"
        }
    }
else:
    tooltip = {
        "html": f"""
        <b>{{municipio}}</b><br/>
        {provincia_line}
        <b>Population:</b> {{poblacion_total:,.0f}}<br/>
        <b>Banks:</b> {{num_bancos:.0f}}<br/>
        <b>Economic Score:</b> {{economic_score_normalized:.1f}}<br/>
        <b>Social Score:</b> {{social_score_normalized:.1f}}<br/>
        <b>Total Score:</b> {{current_score:.1f}}
        """,
        "style": {
            "backgroundColor": "rgba(30,30,30,0.95)",
            "color": "white",
            "fontSize": "12px",
            "padding": "10px"
        }
    }

# Create deck
view_pitch = 45 if viz_mode == "Continuous Heatmap" else 0
deck = pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=5.5,
        pitch=view_pitch,
        bearing=0
    ),
    map_provider="carto",
    map_style="light",
    layers=layers,
    tooltip=tooltip
)

st.pydeck_chart(deck, use_container_width=True)

# Legend
if viz_mode == "Continuous Heatmap":
    st.markdown("""
    **Map Legend:**
    - 🟦 Blue/Teal → Low Score | 🟨 Yellow → Medium Score | 🟥 Red → High Score
    - � Hexagon height represents score density
    - ⭐ Gold circles = Top {} locations
    - 💡 Tip: Rotate the map by holding Ctrl (Cmd on Mac) and dragging
    """.format(top_n))
else:
    st.markdown("""
    **Map Legend:**
    - �🔵 Blue → Low Score | 🟡 Yellow → Medium Score | 🔴 Red → High Score
    - ⭐ Gold circles = Top {} locations
    """.format(top_n))

st.divider()

# ===========================================
# COMPARISON SECTION (if PCA mode)
# ===========================================

if use_pca:
    with st.expander("🔬 PCA vs Sophisticated Scoring Comparison", expanded=True):
        st.markdown(f"""
        ### Experimental Mode: PCA Components
        
        You are currently viewing results using **raw PCA components** combined in real-time:
        
        **Formula:**
        ```
        Combined Score = α × PC1_normalized + (1-α) × PC2_normalized
        ```
        
        Where:
        - **PC1 (Economic)**: First principal component from economic variables
        - **PC2 (Social)**: Second principal component from social variables
        - **α = {alpha:.2f}**: Your current strategic focus
        
        ### Why Compare?
        
        **PCA Method (Current):**
        - ✅ Simple linear combination
        - ✅ Captures main variance in data
        - ❌ No domain-specific modeling (sigmoid, costs, etc.)
        - ❌ Linear assumption may miss non-linear relationships
        
        **Sophisticated Scoring (Toggle Off):**
        - ✅ Domain-specific knowledge (sigmoid market opportunity)
        - ✅ Non-linear relationships modeled
        - ✅ Scalable costs and infrastructure penalties
        - ✅ Multi-factor interactions
        
        ### Expected Alignment
        
        If PCA and sophisticated scoring produce similar rankings, it suggests:
        - The sophisticated scoring captures the natural structure of the data
        - The added complexity is justified by domain knowledge
        - Both methods agree on opportunity identification
        
        **💡 Tip:** Toggle between modes and compare the top 20 municipalities to see how rankings change!
        """)

st.divider()

# ===========================================
# INSIGHTS & INTERPRETATION
# ===========================================

with st.expander("📊 How to Interpret the Results", expanded=False):
    st.markdown(f"""
    ### Current Strategy: {strategy_label}
    
    **Scoring Method:** {scoring_method}
    
    **Economic Score (α={alpha:.2f})** considers:
    - Potential revenue (income × density × economic activity)
    - Scalable costs (office size, staffing per customer)
    - Infrastructure penalty in underbanked areas
    - **Sigmoid market opportunity** (peaks at 40% bank saturation)
    
    **Social Score (1-α={1-alpha:.2f})** considers:
    - Community sustainability (elderly × youth balance)
    - Financial need (inverse of bank saturation)
    - Future viability (requires youth population)
    
    ### Key Innovations:
    1. **Sigmoid Function**: Recognizes that moderate competition (30-50%) is optimal
    2. **Scalable Costs**: Office size and staffing scale with expected demand
    3. **Infrastructure Penalty**: Higher costs in financial deserts
    4. **Sustainable Social**: Requires both current need (elderly) AND future viability (youth)
    
    ### Recommended Actions:
    - **α=0.0-0.3**: Focus on underserved communities, financial inclusion
    - **α=0.4-0.6**: Balanced approach (recommended for Caixa Enginyers)
    - **α=0.7-1.0**: Maximize profitability, urban centers
    """)

with st.expander("🔧 Technical Details", expanded=False):
    st.markdown(f"""
    ### Scoring Methodology
    
    **Total Score Formula:**
    ```
    Total Score = α × Economic Score + (1-α) × Social Score
    ```
    
    **Economic Score Components:**
    - Revenue = Income × Density × Economic_Activity_Factor
    - Costs = (Fixed + Variable) × Operational × Infrastructure
    - Market_Opportunity = Sigmoid(bank_saturation, peak=0.4)
    - ROI = (Revenue / Costs) × Market_Opportunity
    
    **Social Score Components:**
    - Community_Sustainability = %_over_65 × %_under_30
    - Financial_Need = 1 - normalized_bank_count
    - Social_Score = Community × Financial_Need
    
    ### Data Quality:
    - ✅ {len(df)} municipalities processed
    - ✅ Outliers capped at 99th percentile
    - ✅ Percentages normalized to [0-1]
    - ✅ Duplicates removed
    - ✅ Missing values imputed
    """)

# Footer
st.divider()
st.caption("🏦 Caixa Enginyers Branch Location Optimization Tool | Built with Streamlit | Data: INE Spain")
