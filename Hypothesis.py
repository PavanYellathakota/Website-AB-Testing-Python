# %% [markdown]
# # Website Theme Performance Analysis
# 
# This project analyzes website performance metrics to compare "Light Theme" and "Dark Theme" using a dataset (`data/abt.csv`) with 1000+ users records. 
# 
# The goal is to assess differences in `Click_Through_Rate`, `Conversion_Rate`, `Bounce_Rate`, and `Scroll_Depth`, and predict `Purchases` using logistic regression.
# 
# ## Requirements
# 
# ### Libraries
# Install via `pip`:
# ```bash
# pip install pandas numpy scipy seaborn matplotlib scikit-learn IPython
# 

# %%
# Import necessary libraries for analysis, stats, and visualization
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display  # For Jupyter table display

# Set plot style for consistency
sns.set(style="whitegrid")


# %%
# Load the dataset
df = pd.read_csv('data/abt.csv')

# Display first few rows to verify
print("First 5 rows of the dataset:")
display(df.head())

# Check data types and missing values
print("\nData Types and Info:")
df.info()

# %% [markdown]
# # Let's Clean and Convert Numerical Columns

# %%
# Define numerical columns to ensure they’re numeric
numerical_cols = ['Click_Through_Rate', 'Conversion_Rate', 'Bounce_Rate', 'Scroll_Depth', 'Age', 'Session_Duration']

# Convert to numeric, coercing errors to NaN
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Verify conversion
print("Data Types After Conversion:")
display(df.dtypes)

# Check for missing values after conversion
print("\nMissing Values:")
display(df.isnull().sum())

# %%
# DataSet Summary
summary = {
    'Number of Records': df.shape[0],
    'Number of Columns': df.shape[1],
    'Missing Values': df.isnull().sum(),
    'Numerical Columns Summary': df.describe()
}

summary

# %% [markdown]
# # Group by Theme and Calculate Means

# %%
# Lets list the numerical columns
numerical_cols = ["Click_Through_Rate", "Conversion_Rate", "Bounce_Rate", "Scroll_Depth", "Age", "Session_Duration"]
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define the performance metrics you care about (excluding Age, Session_Duration if not intended)
performance_metrics = ["Click_Through_Rate", "Conversion_Rate", "Bounce_Rate", "Scroll_Depth", "Age", "Session_Duration"]

# Group by Theme and calculate mean for selected metrics
theme_performance = df.groupby('Theme')[performance_metrics].mean()

# Sort by Conversion_Rate (note the underscore)
theme_performance_sorted = theme_performance.sort_values(by='Conversion_Rate', ascending=False)

from rich.console import Console
from rich.table import Table

# Create a rich table
console = Console()
table = Table(title="Theme Performance (Mean Values, Sorted by Conversion_Rate)")

# Add columns
table.add_column("Theme", justify="left", style="cyan", no_wrap=True)
for col in theme_performance_sorted.columns:
    table.add_column(col, justify="center", style="White", no_wrap=True)

# Add rows
for index, row in theme_performance_sorted.round(3).iterrows():
    table.add_row(index, *[f"{x:.3f}" for x in row])

# Print with formatting for readability
console.print(table)

# %% [markdown]
# Dark Theme excels in CTR, Light Theme slightly better in Conversion and engagement (Scroll, Bounce,etc.,).

# %% [markdown]
# # T-Tests for All Metrics

# %% [markdown]
# Hypothesis Testing (T-Tests)
# Method: Welch’s two-sample t-test (alpha = 0.05)
# Hypotheses (for each metric):
#     H₀: No difference in mean between Light and Dark Theme.
#     H₁: Difference exists.

# %%
# Perform t-tests for each metric
metrics = ['Click_Through_Rate', 'Conversion_Rate', 'Bounce_Rate', 'Scroll_Depth']
t_stats = []
p_values = []

for metric in metrics:
    light = df[df['Theme'] == 'Light Theme'][metric].dropna()
    dark = df[df['Theme'] == 'Dark Theme'][metric].dropna()
    # Hypothesis statements for the current metric
    # H₀: There is no difference in the mean of {metric} between Light Theme and Dark Theme
    # H₁: There is a difference in the mean of {metric} between Light Theme and Dark Theme
    t_stat, p_val = ttest_ind(light, dark, equal_var=False)  # Welch's t-test
    t_stats.append(t_stat)
    p_values.append(p_val)
    print(f"\n{metric}:")
    print(f"  T-Statistic: {t_stat:.3f}, P-Value: {p_val:.3f}")
    print(f"  Sample Size (Light): {len(light)}, (Dark): {len(dark)}")

# Create comparison table
comparison_table = pd.DataFrame({
    'Metric': metrics,
    'T-Statistic': t_stats,
    'P-Value': p_values
})

# Style the table
styled_comparison = (comparison_table
                     .round({'T-Statistic': 3, 'P-Value': 3})
                     .style
                     .set_caption("T-Test Comparison Between Light and Dark Themes")
                     .format({'T-Statistic': '{:.3f}', 'P-Value': '{:.3f}'})
                     .highlight_max(subset=['P-Value'], color='lightcoral'))
print("\nT-Test Comparison Table:")
display(styled_comparison)

# %% [markdown]
# Click_Through_Rate: T = -1.978, P = 0.048 → Significant (Dark > Light).
# 
# Conversion_Rate: T = 0.475, P = 0.635 → Not significant.
# 
# Bounce_Rate: T = -1.202, P = 0.230 → Not significant.
# 
# Scroll_Depth: T = 0.756, P = 0.450 → Not significant.
# 
# Conclusion : Only CTR shows a significant difference, favoring Dark Theme.

# %% [markdown]
# # Correlation Analysis

# %%
# Correlation matrix for numerical metrics
corr = df[performance_metrics].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Performance Metrics")
plt.show()

# %% [markdown]
# Weak correlations (e.g., < 0.3) among Click_Through_Rate, Conversion_Rate, Bounce_Rate, Scroll_Depth.
# 
# No strong linear relationships; metrics are largely independent.

# %% [markdown]
# # Distribution Plots by Theme

# %%
# Plot distributions for Conversion_Rate by Theme
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Conversion_Rate', hue='Theme', kde=True, bins=20)
plt.title("Conversion Rate Distribution by Theme")
plt.xlabel("Conversion Rate")
plt.show()

# %% [markdown]
# Conversion_Rate by Theme:
# Light Theme: Bimodal (peaks ~0.15, ~0.40) → Two user groups.
# Dark Theme: Unimodal (~0.25) → Consistent but lower conversions.
# 
# Insight: Light Theme has a high-converting subgroup; Dark Theme is more uniform.

# %% [markdown]
# # Boxplots by Location

# %%
# Boxplot of Bounce_Rate by Location
plt.figure(figsize=(10, 6))
sns.boxplot(x='Location', y='Bounce_Rate', hue='Theme', data=df)
plt.title("Bounce Rate by Location and Theme")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Bounce_Rate:
# New Delhi: Dark Theme higher median (~0.60) vs. Light (~0.50).
# Other locations (e.g., Chennai, Pune): More consistent across themes.
# 
# Insight: Location impacts Bounce more than Theme; New Delhi dislikes Dark Theme.

# %% [markdown]
# # Simple Predictive Model (Logistic Regression)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Prepare data for predicting Purchases
df['Purchases'] = LabelEncoder().fit_transform(df['Purchases'])  # Yes=1, No=0
X = df[performance_metrics + ['Age']].dropna()
y = df.loc[X.index, 'Purchases']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Logistic Regression Accuracy for Predicting Purchases: {accuracy:.3f}")

# Feature importance (coefficients)
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
display(coef_df)

# %% [markdown]
# Model: Logistic Regression to predict Purchases (Yes=1, No=0).
# 
# Features: Click_Through_Rate, Conversion_Rate, Bounce_Rate, Scroll_Depth, Age.
# 
# Accuracy: ~60% (low, close to random guessing).
# 
# Coefficients (example):
# 
#     Positive: Conversion_Rate (0.02), Scroll_Depth (0.01), Session_Duration (0.005).
# 
#     Negative: Click_Through_Rate (-0.01), Bounce_Rate (-0.03), Age (-0.002).
# 
# Insight: Weak predictors; small effects. Conversion aids purchases, Bounce hurts. Model needs non-linear features or more data.


