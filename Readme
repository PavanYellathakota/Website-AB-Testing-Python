# 🌗 Light vs Dark Theme Website Performance

### Analyzing User Behavior Insights & A/B Testing Website Metrics with Python


This project analyzes website performance metrics to compare "Light Theme" and "Dark Theme" using a dataset (`data/abt.csv`) with 1000 user records. Built in Python within a Jupyter Notebook (`abt.ipynb`), it leverages `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`, and `scikit-learn` for data processing, statistical testing, and visualizations.

---

## Dataset Overview

The dataset (`data/abt.csv`) contains website user interaction data with 1000 rows (486 Light Theme, 514 Dark Theme). A preview of key columns:

| Theme       | Click_Through_Rate | Conversion_Rate | Bounce_Rate | Scroll_Depth | Age | Location  | Session_Duration | Purchases | Added_to_Cart |
|-------------|--------------------|-----------------|-------------|--------------|-----|-----------|------------------|-----------|---------------|
| Light Theme | 0.032              | 0.208           | 0.754       | 73.02        | 52  | Chennai   | 1176             | No        | Yes           |
| Light Theme | 0.143              | 0.028           | 0.306       | 35.02        | 33  | Pune      | 780              | Yes       | Yes           |
| Dark Theme  | 0.323              | 0.179           | 0.297       | 45.74        | 27  | New Delhi | 912              | No        | No            |
| Light Theme | 0.489              | 0.326           | 0.649       | 79.37        | 41  | Chennai   | 1345             | No        | Yes           |
| Light Theme | 0.098              | 0.273           | 0.438       | 72.18        | 36  | Pune      | 645              | Yes       | Yes           |

- **Size**: 1000 rows (sample: 5 shown).
- **Key Metrics**: CTR, Conversion, Bounce, Scroll, Purchases, Demographics.

---

## Analysis Sections and Visualizations

### 1. Data Summary
Provides descriptive statistics for numerical and categorical variables.

- **Metrics**: Means, std, ranges for `Click_Through_Rate`, `Conversion_Rate`, etc.
- **Visualizations**: None (tabular output).
- **Insights**:
  - Mean CTR: 0.256, Conversion: 0.253, Bounce: 0.506, Scroll: 50.32.
  - Age range: 18-65, Session: ~15.5min avg.

---

### 2. Theme Performance Comparison
Compares mean metrics between themes.

- **Metrics**: `Click_Through_Rate`, `Conversion_Rate`, `Bounce_Rate`, `Scroll_Depth`.
- **Visualizations**: Styled table.
- **Insights**:
  - Dark Theme CTR: 0.265 vs. Light 0.247 (higher).
  - Light Theme Conversion: 0.255 vs. Dark 0.251 (slight edge).

---

### 3. Hypothesis Testing
Tests for significant differences using Welch’s t-tests (alpha = 0.05).

- **Metrics**: `Click_Through_Rate`, `Conversion_Rate`, `Bounce_Rate`, `Scroll_Depth`.
- **Hypotheses**: 
  - H₀: No difference in means.
  - H₁: Difference exists.
- **Visualizations**: Styled table with p-values.
- **Insights**:
  - CTR: p = 0.048 (significant, Dark > Light).
  - Conversion: p = 0.635 (not significant).

---

### 4. Correlation Analysis
Examines linear relationships between metrics.

- **Metrics**: `Click_Through_Rate`, `Conversion_Rate`, `Bounce_Rate`, `Scroll_Depth`.
- **Visualizations**: Heatmap (`seaborn`).
- **Insights**:
  - Weak correlations (< 0.3) → metrics are independent.

---

### 5. Distribution Analysis
Explores metric distributions by theme.

- **Metrics**: `Conversion_Rate`.
- **Visualizations**: Histogram with KDE (`seaborn`).
- **Insights**:
  - Light Theme: Bimodal (~0.15, ~0.40) → two user groups.
  - Dark Theme: Unimodal (~0.25) → consistent, lower conversions.

---

### 6. Location-Based Analysis
Assesses bounce rates by location and theme.

- **Metrics**: `Bounce_Rate`.
- **Visualizations**: Box plot (`seaborn`).
- **Insights**:
  - New Delhi: Dark Theme higher median (~0.60) vs. Light (~0.50).
  - Location > Theme influence.

---

### 7. Predictive Modeling
Predicts `Purchases` using logistic regression.

- **Features**: `Click_Through_Rate`, `Conversion_Rate`, `Bounce_Rate`, `Scroll_Depth`, `Age`.
- **Visualizations**: None (tabular coefficients).
- **Insights**:
  - Accuracy: ~60% (weak).
  - Positive: `Conversion_Rate` (0.02), `Scroll_Depth` (0.01).
  - Negative: `Bounce_Rate` (-0.03), `Click_Through_Rate` (-0.01).

---

## Key Findings
- **Theme Performance**: Dark Theme boosts CTR significantly (p = 0.048); Light Theme slightly better for conversions (not significant).
- **User Behavior**: Weak metric correlations; bimodal Conversion in Light Theme suggests segmentation.
- **Location Impact**: New Delhi favors Light Theme (lower bounce).
- **Prediction**: Current features poorly predict purchases—needs non-linear models.

---

## Methodology
- **Tools**: Python with `pandas`, `numpy`, `scipy.stats`, `seaborn`, `matplotlib`, `scikit-learn`.
- **Visualizations**: Static plots (heatmap, histogram, box plot) with `seaborn`.
- **Processing**: T-tests, correlation, regression, data cleaning (`pd.to_numeric`).

---

## Usage
1. **Setup**: Install dependencies:
   ```bash
   pip install pandas numpy scipy seaborn matplotlib scikit-learn IPython

---
  
## Author

<div align="center">
  <img src="assets/PYE.svg" alt="Author Banner" style="width:100%; height:auto; border-radius: 8px;">
</div>

**Author**: [Pavan Yellathakota]  
**Date**: MAR 2025  

---

## Contact Information

You can reach out to me through the following channels:

- **Email**: [pavanyellathakota@gmail.com](mailto:pavanyellathakota@gmail.com)
- **LinkedIn**: [Pavan Yellathakota](https://www.linkedin.com/in/pavanyellathakota/)

For more projects and resources, check out:

- **GitHub**: [Pavan Yellathakota](https://github.com/PavanYellathakota)
- **Portfolio**: [pye.pages.dev](https://pye.pages.dev)

---


"""
