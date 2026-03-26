---
categories: kaggle machine-learning
date: '2025-02-09'
layout: post
slug: stats-test-vs-building-model
title: Stats Test Vs Building Model
---

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic marketing campaign data
def generate_campaign_data(n_samples=1000):
    # Customer features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(70000, 20000, n_samples)
    prev_purchases = np.random.poisson(3, n_samples)
    
    # Campaign A gets slightly higher conversion on average
    campaign_A = np.random.binomial(1, 0.15, n_samples)  # 15% conversion
    campaign_B = np.random.binomial(1, 0.12, n_samples)  # 12% conversion
    
    # Create dataframes for each campaign
    df_A = pd.DataFrame({
        'age': age[:n_samples//2],
        'income': income[:n_samples//2],
        'prev_purchases': prev_purchases[:n_samples//2],
        'converted': campaign_A[:n_samples//2],
        'campaign': 'A'
    })
    
    df_B = pd.DataFrame({
        'age': age[n_samples//2:],
        'income': income[n_samples//2:],
        'prev_purchases': prev_purchases[n_samples//2:],
        'converted': campaign_B[n_samples//2:],
        'campaign': 'B'
    })
    
    return pd.concat([df_A, df_B])

# Generate data
data = generate_campaign_data(2000)

# Approach 1: Statistical Analysis
def statistical_analysis(data):
    print("Statistical Analysis (When we want to understand if there's a real difference)")
    print("-" * 80)
    
    # Calculate conversion rates
    campaign_A = data[data['campaign'] == 'A']['converted']
    campaign_B = data[data['campaign'] == 'B']['converted']
    
    # Perform chi-square test
    contingency = pd.crosstab(data['campaign'], data['converted'])
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    # Calculate confidence intervals using normal approximation
    def conf_interval(successes, n):
        p = successes / n
        se = np.sqrt(p * (1-p) / n)
        ci = stats.norm.interval(0.95, p, se)
        return ci
    
    n_A = len(campaign_A)
    n_B = len(campaign_B)
    conv_A = campaign_A.mean()
    conv_B = campaign_B.mean()
    ci_A = conf_interval(campaign_A.sum(), n_A)
    ci_B = conf_interval(campaign_B.sum(), n_B)
    
    print(f"Campaign A conversion: {conv_A:.1%} (95% CI: {ci_A[0]:.1%} - {ci_A[1]:.1%})")
    print(f"Campaign B conversion: {conv_B:.1%} (95% CI: {ci_B[0]:.1%} - {ci_B[1]:.1%})")
    print(f"Chi-square p-value: {p_value:.4f}")
    print(f"Absolute difference: {(conv_A - conv_B):.1%}")
    
    # Business interpretation
    if p_value < 0.05:
        print("\nBusiness Interpretation:")
        print("There is statistically significant evidence that the campaigns differ in effectiveness.")
        print(f"Campaign A is estimated to generate {(conv_A - conv_B):.1%} more conversions,")
        print(f"with 95% confidence that the true difference is between {(ci_A[0] - ci_B[1]):.1%} and {(ci_A[1] - ci_B[0]):.1%}")
    else:
        print("\nNo statistically significant difference found between campaigns.")

# Approach 2: ML Classification
def ml_analysis(data):
    print("\nML Classification Analysis (When we want to predict individual customer conversion)")
    print("-" * 80)
    
    # Prepare features
    X = data[['age', 'income', 'prev_purchases']]
    y = data['converted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Print classification report
    print("\nModel Performance Metrics:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': ['age', 'income', 'prev_purchases'],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    # Example prediction for a new customer
    print("\nExample Prediction for New Customer:")
    new_customer = pd.DataFrame({
        'age': [35],
        'income': [75000],
        'prev_purchases': [5]
    })
    pred_prob = clf.predict_proba(new_customer)[0][1]
    print(f"Probability of conversion: {pred_prob:.1%}")

# Run both analyses
statistical_analysis(data)
ml_analysis(data)
```


