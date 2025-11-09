# Enhanced Economic Scoring Methodology

## Overview
The enhanced `calculate_economic_score()` function provides a sophisticated ROI estimation model for bank branch placement that accounts for real-world operational complexities.

## Key Improvements Over Basic Model

### 1. **Scalable Fixed Costs**
- **Problem**: Original model used flat rent costs regardless of demand
- **Solution**: Office size now scales with expected user base
  ```
  required_office_size = base_office_size * (1 + expected_users / 1000)
  ```
- **Impact**: High-traffic locations correctly show higher infrastructure costs

### 2. **Variable Staffing Costs**
- **New Feature**: Employee costs proportional to customer base
  ```
  annual_staffing_cost = employee_cost_per_user * expected_users
  ```
- **Rationale**: More customers = more tellers, advisors, support staff needed

### 3. **Infrastructure Penalty for Financial Deserts**
- **Problem**: Opening in unbanked areas has hidden costs (ATM networks, internet infrastructure, security systems)
- **Solution**: Progressive cost multiplier based on banking saturation
  ```
  if bank_saturation < 0.2:  # Financial desert
      infrastructure_cost_multiplier = 1 + penalty * (0.2 - saturation) * 5
  ```
- **Impact**: Models the reality that "first mover" has setup costs

### 4. **Non-Linear Market Opportunity**
- **Problem**: Original assumed linear relationship (no banks = best opportunity)
- **Solution**: Sigmoid function recognizing that:
  - **Too few banks** (< 30%): High opportunity BUT infrastructure risks
  - **Moderate competition** (30-70%): **Sweet spot** - validated demand with room to grow
  - **Saturated** (> 70%): Diminishing returns
- **Banking Industry Insight**: Some competition validates market demand and shares infrastructure costs

### 5. **Economic Activity Adjustment**
- **New Feature**: Revenue adjusted by working-age population
  ```
  perc_economically_active = 1 - perc_under_30 - perc_over_65
  ```
- **Rationale**: 30-65 age group has highest banking needs and account activity

## Cost Structure Breakdown

```
Total Cost = (Fixed Costs + Variable Costs) × Operational Multiplier × Infrastructure Multiplier

Where:
- Fixed Costs = Office rent (scaled by demand)
- Variable Costs = Staffing (per customer)
- Operational Multiplier = Utilities, IT, security, compliance (~1.5x)
- Infrastructure Multiplier = Setup costs in underbanked areas (1.0 - 2.5x)
```

## Parameters (Tunable for Spanish Market)

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `base_office_size` | 100 m² | Starting office size | Adjust based on Caixa Enginyers typical branch size |
| `employee_cost_per_user` | €50/user/year | Staffing cost per customer | Calculate from average salary ÷ customers per employee |
| `infrastructure_penalty` | 0.3 (30%) | Extra cost in unbanked areas | Higher in rural Spain, lower in urban areas |
| `operational_cost_multiplier` | 1.5 | General overhead factor | Industry standard is 1.3-1.7 |

## Example Scenarios

### Scenario A: Financial Desert (normalized_bank_count = 0.1)
- **High Opportunity**: Large underserved population
- **High Costs**: Must build ATM network, establish digital infrastructure
- **Result**: Good score only if population/income justify infrastructure investment

### Scenario B: Competitive Market (normalized_bank_count = 0.5)
- **Moderate Opportunity**: Validated demand, room to differentiate
- **Moderate Costs**: Shared infrastructure (ATM networks, business services)
- **Result**: Often the **best ROI** - demand proven with manageable costs

### Scenario C: Saturated Market (normalized_bank_count = 0.9)
- **Low Opportunity**: Must steal customers from competitors
- **Low Costs**: Excellent infrastructure already exists
- **Result**: Poor score unless income/density extremely high

## Validation Recommendations

1. **Calibrate with Historical Data**: If Caixa Enginyers has data on actual branch profitability, use it to tune parameters
2. **Regional Adjustments**: Consider creating separate parameters for:
   - Urban vs Rural
   - High-income regions (Catalonia, Madrid) vs others
   - Tourist areas (seasonal factors)
3. **Sensitivity Analysis**: Test how scores change with ±20% parameter variation
4. **Expert Review**: Validate with bank operations managers on cost assumptions

## Integration with Social Score

The combined score formula remains:
```
Total_Score = α × Economic_Score + (1 - α) × Social_Score
```

**Recommended α ranges:**
- **Pure profit focus**: α = 0.8-1.0
- **Balanced approach**: α = 0.4-0.6 (recommended for Caixa Enginyers' social mission)
- **Maximum social impact**: α = 0.0-0.2

## Next Steps for Phase 2 (Forecasting)

To forecast economic scores 1/3/5 years ahead, incorporate:

1. **Demographic Trends**: Population growth/decline projections
2. **Income Evolution**: Regional GDP growth forecasts
3. **Digital Transformation**: Predicted reduction in branch traffic (e.g., -5% annually)
4. **Competition Dynamics**: Planned branch openings/closures by competitors
5. **Regulatory Changes**: Digital euro, open banking impact on revenue

The current model provides a solid foundation - just make the input variables time-dependent!
