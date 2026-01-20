---
categories: kaggle machine-learning
date: '2025-02-17'
layout: post
slug: how-to-debug-a-model
title: How To Debug A Model
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import shap

class ModelDebugger:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        """
        Initialize debugger with model and data
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def check_data_quality(self):
        """
        Basic data quality checks
        """
        checks = {
            'missing_values': self.X_train.isnull().sum(),
            'feature_correlation': self.X_train.corr(),
            'class_distribution': pd.Series(self.y_train).value_counts(normalize=True),
            'feature_statistics': self.X_train.describe()
        }
        
        # Check for data leakage
        train_test_overlap = len(set(map(tuple, self.X_train.values)) & 
                               set(map(tuple, self.X_test.values)))
        checks['data_leakage'] = train_test_overlap
        
        return checks
    
    def analyze_model_performance(self):
        """
        Comprehensive model performance analysis
        """
        # Make predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Basic metrics
        performance = {
            'train_score': self.model.score(self.X_train, self.y_train),
            'test_score': self.model.score(self.X_test, self.y_test),
            'train_report': classification_report(self.y_train, train_pred),
            'test_report': classification_report(self.y_test, test_pred)
        }
        
        return performance
    
    def plot_learning_curve(self):
        """
        Generate learning curves to diagnose bias/variance
        """
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        
    def analyze_feature_importance(self):
        """
        Analyze feature importance using SHAP values
        """
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_train)
            
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': np.abs(shap_values).mean(0)
            })
            return feature_importance.sort_values('importance', ascending=False)
            
        except:
            if hasattr(self.model, 'feature_importances_'):
                return pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                return "Model doesn't support feature importance analysis"
    
    def error_analysis(self):
        """
        Analyze specific examples where model makes mistakes
        """
        test_pred = self.model.predict(self.X_test)
        
        # Find misclassified examples
        mistakes_idx = np.where(test_pred != self.y_test)[0]
        mistakes_analysis = pd.DataFrame({
            'true_label': self.y_test.iloc[mistakes_idx],
            'predicted_label': test_pred[mistakes_idx],
            'data': self.X_test.iloc[mistakes_idx].values.tolist()
        })
        
        return mistakes_analysis
    
    def suggest_improvements(self):
        """
        Suggest potential improvements based on analysis
        """
        performance = self.analyze_model_performance()
        train_score = performance['train_score']
        test_score = performance['test_score']
        
        suggestions = []
        
        # Check for overfitting
        if train_score - test_score > 0.1:
            suggestions.append({
                'issue': 'Potential overfitting',
                'suggestions': [
                    'Increase regularization',
                    'Reduce model complexity',
                    'Collect more training data',
                    'Use cross-validation',
                    'Try feature selection'
                ]
            })
            
        # Check for underfitting
        if train_score < 0.8:  # Threshold can be adjusted
            suggestions.append({
                'issue': 'Potential underfitting',
                'suggestions': [
                    'Increase model complexity',
                    'Add more features',
                    'Reduce regularization',
                    'Try feature engineering',
                    'Consider a more complex model architecture'
                ]
            })
            
        return suggestions

# Example usage
"""
debugger = ModelDebugger(model, X_train, y_train, X_test, y_test)

# Run basic checks
data_quality = debugger.check_data_quality()
performance = debugger.analyze_model_performance()

# Visualize learning curves
debugger.plot_learning_curve()

# Analyze feature importance
importance = debugger.analyze_feature_importance()

# Analyze errors
errors = debugger.error_analysis()

# Get improvement suggestions
suggestions = debugger.suggest_improvements()
"""
```


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y, title="Learning Curves", ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate and plot learning curves for a given model.
    
    Parameters:
    -----------
    model : estimator instance
        The model to analyze (e.g., RandomForestClassifier, LogisticRegression)
    X : array-like
        Training data
    y : array-like
        Target values
    title : string
        Plot title
    ylim : tuple, shape (ymin, ymax), optional
        Defines min/max y-values plotted
    cv : int, cross-validation generator
        Determines cross-validation splitting strategy
    n_jobs : int or None, optional
        Number of jobs to run in parallel
    train_sizes : array-like
        Points at which to evaluate learning curves
        
    Returns:
    --------
    plt : matplotlib plot object
    """
    plt.figure(figsize=(10, 6))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title(title)
    
    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # Calculate mean and std for training set scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    # Calculate mean and std for test set scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot training scores
    plt.plot(train_sizes, train_scores_mean, label="Training score",
             color="r", marker='o')
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color="r")
    
    # Plot cross-validation scores
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score",
             color="g", marker='o')
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    
    plt.grid(True)
    plt.legend(loc="best")
    
    # Add analysis text
    gap = train_scores_mean[-1] - test_scores_mean[-1]
    final_score = test_scores_mean[-1]
    
    analysis_text = f"""
    Analysis:
    - Final CV Score: {final_score:.3f}
    - Train-Test Gap: {gap:.3f}
    - Learning Status: {get_learning_status(gap, final_score)}
    """
    plt.text(0.02, 0.02, analysis_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=8, verticalalignment='bottom')
    
    return plt

def get_learning_status(gap, score, gap_threshold=0.1, score_threshold=0.8):
    """
    Determine the learning status based on the gap between training and CV scores
    and the final CV score.
    """
    if gap > gap_threshold and score < score_threshold:
        return "High Variance & High Bias"
    elif gap > gap_threshold:
        return "High Variance (Overfitting)"
    elif score < score_threshold:
        return "High Bias (Underfitting)"
    else:
        return "Good Fit"

def demonstrate_common_patterns():
    """
    Demonstrate common learning curve patterns using synthetic examples.
    """
    # Example curves for different scenarios
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Good fit
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Good Fit
    plt.subplot(2, 2, 1)
    train_scores = 0.95 - np.exp(-2*train_sizes) * 0.2
    test_scores = 0.90 - np.exp(-2*train_sizes) * 0.2
    plt.plot(train_sizes, train_scores, 'r-', label='Training score')
    plt.plot(train_sizes, test_scores, 'g-', label='CV score')
    plt.title('Good Fit')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Overfitting
    plt.subplot(2, 2, 2)
    train_scores = 0.99 - np.exp(-3*train_sizes) * 0.1
    test_scores = 0.85 - np.exp(-train_sizes) * 0.1
    plt.plot(train_sizes, train_scores, 'r-', label='Training score')
    plt.plot(train_sizes, test_scores, 'g-', label='CV score')
    plt.title('Overfitting (High Variance)')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Underfitting
    plt.subplot(2, 2, 3)
    train_scores = 0.75 - np.exp(-2*train_sizes) * 0.1
    test_scores = 0.70 - np.exp(-2*train_sizes) * 0.1
    plt.plot(train_sizes, train_scores, 'r-', label='Training score')
    plt.plot(train_sizes, test_scores, 'g-', label='CV score')
    plt.title('Underfitting (High Bias)')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Need More Data
    plt.subplot(2, 2, 4)
    train_scores = 0.90 - np.exp(-train_sizes) * 0.3
    test_scores = 0.85 - np.exp(-train_sizes) * 0.3
    plt.plot(train_sizes, train_scores, 'r-', label='Training score')
    plt.plot(train_sizes, test_scores, 'g-', label='CV score')
    plt.title('Need More Data')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    return plt

# Example usage:
"""
# For a real model:
model = RandomForestClassifier()
plot_learning_curves(model, X, y, "Random Forest Learning Curves")

# To see common patterns:
demonstrate_common_patterns()
"""
```


```python
# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20)

# Initialize diagnostics
diagnostics = ModelDiagnostics(X, y)

# Add your models
diagnostics.add_model(LogisticRegression(), "Logistic Regression")
diagnostics.add_model(RandomForestClassifier(), "Random Forest")

# Generate all diagnostics
diagnostics.plot_learning_curves("Random Forest")
diagnostics.plot_confusion_matrices()
diagnostics.analyze_feature_importance("Random Forest")
```

Learning curves are plots that show how model performance changes as you add more training data. They're incredibly useful for diagnosing model behavior. Let me break down the key aspects:

1. What Learning Curves Show:
   - Training score: How well the model performs on training data
   - Cross-validation score: How well the model performs on validation data
   - Both are plotted against the number of training examples

2. Common Patterns:

   A. Good Fit:
   - Training and CV scores are close together
   - Both curves reach a plateau at a good performance level
   - Small gap between training and CV performance

   B. Overfitting (High Variance):
   - Large gap between training and CV scores
   - Training score much higher than CV score
   - Adding more data might help

   C. Underfitting (High Bias):
   - Both scores are low
   - Curves are close together
   - Adding more data won't help much
   - Need a more complex model

   D. Need More Data:
   - Curves haven't plateaued yet
   - Scores are still improving with more data
   - Gap between curves is still closing

3. How to Use Them:
   - Look at the final gap between curves (variance)
   - Look at the final performance level (bias)
   - Check if scores are still improving
   - Use to decide between:
     - Getting more training data
     - Adding/removing features
     - Increasing/decreasing model complexity

4. Key Interpretations:
   - If curves are close but performance is poor → Underfitting
   - If curves are far apart → Overfitting
   - If curves haven't plateaued → Need more data
   - If curves plateau at good performance and are close → Good fit



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import shap

class ModelDiagnostics:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """Initialize with data and create train/test split"""
        # Convert X to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
        
        # Convert y to Series if it's not already
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # Store feature names
        self.feature_names = X.columns
        
        # Dictionary to store models
        self.models = {}
        self.predictions = {}
        
    def add_model(self, model, name):
        """Add and train a model"""
        self.models[name] = model
        model.fit(self.X_train_scaled, self.y_train)
        self.predictions[name] = {
            'train': model.predict(self.X_train_scaled),
            'test': model.predict(self.X_test_scaled)
        }
        
    def plot_learning_curves(self, model_name, cv=5):
        """Plot learning curves for a specific model"""
        model = self.models[model_name]
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train_scaled, self.y_train,
            cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label=f'Training score - {model_name}')
        plt.plot(train_sizes, test_mean, label=f'Cross-validation score - {model_name}')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
            
        for ax, (name, model) in zip(axes, self.models.items()):
            cm = confusion_matrix(self.y_test, self.predictions[name]['test'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
    def get_classification_reports(self):
        """Generate classification reports for all models"""
        reports = {}
        for name, model in self.models.items():
            reports[name] = classification_report(
                self.y_test, 
                self.predictions[name]['test']
            )
        return reports
    
    def analyze_feature_importance(self, model_name):
        """Analyze feature importance using SHAP values or built-in feature importance"""
        model = self.models[model_name]
        
        if isinstance(model, RandomForestClassifier):
            # For Random Forest, use built-in feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance, x='importance', y='feature')
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
        else:
            try:
                # For other models, use SHAP values
                # Create a background dataset for SHAP
                background_data = shap.sample(self.X_train_scaled, 100, random_state=42)
                
                # Create small example dataset for analysis
                example_data = self.X_test_scaled.iloc[:100]  # Use first 100 test examples
                
                if hasattr(model, 'predict_proba'):
                    explainer = shap.KernelExplainer(
                        model.predict_proba, 
                        background_data,
                        output_names=['Class 0', 'Class 1']
                    )
                    shap_values = explainer.shap_values(example_data)
                    
                    # If shap_values is a list, take the second element (Class 1)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values, 
                        example_data,
                        feature_names=self.feature_names,
                        show=False
                    )
                    plt.title(f'SHAP Values - {model_name}')
                else:
                    print(f"Model {model_name} doesn't support probability predictions for SHAP analysis")
                    return None
                    
            except Exception as e:
                print(f"Error in SHAP analysis for {model_name}: {str(e)}")
                return None
    
    def analyze_errors(self, model_name):
        """Analyze where the model makes mistakes"""
        model = self.models[model_name]
        y_pred = self.predictions[model_name]['test']
        
        # Find misclassified examples
        mistakes_idx = np.where(y_pred != self.y_test)[0]
        
        # Get probabilities for misclassified examples
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(self.X_test_scaled.iloc[mistakes_idx])
            confidence = probs.max(axis=1)
        else:
            confidence = None
            
        # Create DataFrame with misclassified examples
        mistakes_analysis = pd.DataFrame({
            'true_label': self.y_test.iloc[mistakes_idx],
            'predicted_label': y_pred[mistakes_idx],
            'confidence': confidence,
            **{col: self.X_test_scaled.iloc[mistakes_idx][col] for col in self.X_test_scaled.columns}
        })
        
        return mistakes_analysis

# Example usage
def run_diagnostics_example():
    """Run a complete example of model diagnostics"""
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5, 
        random_state=42
    )
    
    # Initialize diagnostics
    diagnostics = ModelDiagnostics(X, y)
    
    # Add different models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        diagnostics.add_model(model, name)
    
    # Generate and plot learning curves
    for model_name in models.keys():
        print(f"\nGenerating learning curves for {model_name}...")
        diagnostics.plot_learning_curves(model_name)
        plt.show()
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    diagnostics.plot_confusion_matrices()
    plt.show()
    
    # Print classification reports
    print("\nGenerating classification reports...")
    reports = diagnostics.get_classification_reports()
    for name, report in reports.items():
        print(f"\nClassification Report - {name}")
        print(report)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    for model_name in models.keys():
        print(f"\nAnalyzing feature importance for {model_name}...")
        diagnostics.analyze_feature_importance(model_name)
        plt.show()
    
    # Analyze errors
    print("\nAnalyzing errors...")
    for model_name in models.keys():
        print(f"\nError Analysis - {model_name}")
        errors = diagnostics.analyze_errors(model_name)
        print(f"Number of misclassified examples: {len(errors)}")
        if len(errors) > 0:
            print("\nSample of misclassified examples:")
            print(errors.head())

if __name__ == "__main__":
    run_diagnostics_example()
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

class AdvancedShapAnalyzer:
    def __init__(self, X, y, model, random_state=42):
        """Initialize with data and model"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
            
        self.X = X
        self.y = y
        self.model = model
        self.feature_names = X.columns
        
        # Split and fit
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Initialize explainer
        self.background = shap.sample(self.X_train, 100, random_state=random_state)
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            self.background,
            output_names=['Class 0', 'Class 1']
        )
        
    def analyze_patterns(self, n_samples=100):
        """Analyze and explain common SHAP patterns"""
        sample_data = self.X_test.iloc[:n_samples]
        shap_values = self.explainer.shap_values(sample_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate pattern metrics
        patterns = []
        for i, feature in enumerate(self.feature_names):
            feature_shap = shap_values[:, i]
            feature_values = sample_data[feature].values
            
            # Calculate correlation between feature values and SHAP values
            correlation = np.corrcoef(feature_values, feature_shap)[0, 1]
            
            # Calculate impact magnitude
            impact_magnitude = np.abs(feature_shap).mean()
            
            # Calculate consistency (what % of SHAP values have same sign)
            consistency = max(
                np.mean(feature_shap > 0),
                np.mean(feature_shap < 0)
            )
            
            patterns.append({
                'feature': feature,
                'correlation': correlation,
                'impact_magnitude': impact_magnitude,
                'consistency': consistency
            })
            
        patterns_df = pd.DataFrame(patterns)
        return patterns_df.sort_values('impact_magnitude', ascending=False)
    
    def feature_selection(self, threshold_percentile=80):
        """Perform feature selection using SHAP values"""
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(self.X_test.iloc[:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(shap_values).mean(0)
        
        # Calculate importance threshold
        threshold = np.percentile(feature_importance, threshold_percentile)
        
        # Select features
        selected_features = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        selected_features['selected'] = selected_features['importance'] >= threshold
        
        return selected_features
    
    def analyze_interactions(self, top_n_features=5):
        """Analyze feature interactions using SHAP values"""
        # Calculate SHAP values for a sample
        sample_data = self.X_test.iloc[:100]
        shap_values = self.explainer.shap_values(sample_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        # Get top features by importance
        mean_abs_shap = np.abs(shap_values).mean(0)
        top_features_idx = np.argsort(mean_abs_shap)[-top_n_features:]
        
        interactions = []
        for i in top_features_idx:
            for j in top_features_idx:
                if i < j:  # Avoid duplicate pairs
                    # Calculate interaction strength
                    interaction_strength = self._calculate_interaction_strength(
                        sample_data, shap_values, i, j
                    )
                    
                    interactions.append({
                        'feature1': self.feature_names[i],
                        'feature2': self.feature_names[j],
                        'interaction_strength': interaction_strength
                    })
                    
        return pd.DataFrame(interactions).sort_values(
            'interaction_strength', ascending=False
        )
    
    def _calculate_interaction_strength(self, data, shap_values, i, j):
        """Calculate interaction strength between two features"""
        # Get feature values and SHAP values
        f1_values = data.iloc[:, i].values
        f2_values = data.iloc[:, j].values
        f1_shap = shap_values[:, i]
        f2_shap = shap_values[:, j]
        
        # Calculate correlation between SHAP values
        interaction = np.abs(np.corrcoef(f1_shap * f2_values, f2_shap * f1_values)[0, 1])
        return interaction

def demonstrate_advanced_shap():
    """Demonstrate advanced SHAP analysis with example data"""
    # Generate synthetic data with known relationships
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=False
    )
    
    # Create feature names
    feature_names = [
        'Strong_Positive',
        'Strong_Negative',
        'Moderate_Positive',
        'Moderate_Negative',
        'Weak_Positive',
        'Redundant_1',
        'Redundant_2',
        'Noise_1',
        'Noise_2',
        'Noise_3'
    ]
    
    X = pd.DataFrame(X, columns=feature_names)
    
    # Initialize analyzer
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    analyzer = AdvancedShapAnalyzer(X, y, model)
    
    # 1. Analyze Patterns
    print("\n=== Pattern Analysis ===")
    patterns = analyzer.analyze_patterns()
    print(patterns)
    
    # 2. Feature Selection
    print("\n=== Feature Selection ===")
    selected_features = analyzer.feature_selection(threshold_percentile=80)
    print(selected_features)
    
    # 3. Interaction Analysis
    print("\n=== Interaction Analysis ===")
    interactions = analyzer.analyze_interactions(top_n_features=5)
    print(interactions)
    
    return analyzer, patterns, selected_features, interactions

if __name__ == "__main__":
    demonstrate_advanced_shap()
```

1. Interpreting Specific Patterns:
   - Clear Linear Relationship:
     ```
     Feature: Strong_Positive
     - High correlation (> 0.7)
     - Consistent impact direction
     - Red dots clustered on right
     - Blue dots clustered on left
     ```
   
   - Threshold Effect:
     ```
     Feature: Moderate_Impact
     - Medium correlation
     - Clear separation point
     - Sudden change in impact
     ```
   
   - Noisy/Weak Relationship:
     ```
     Feature: Noise
     - Low correlation
     - Mixed colors
     - Small SHAP values
     ```

2. Feature Selection Using SHAP:
   Three main criteria:
   - Impact Magnitude: Mean absolute SHAP value
   - Consistency: How often the feature impacts in same direction
   - Correlation: Relationship between feature and SHAP values

   Selection process:
   ```python
   # High impact features
   impact_threshold = np.percentile(mean_abs_shap, 80)
   selected_features = features[mean_abs_shap > impact_threshold]
   ```

3. Analyzing Feature Interactions:
   - Direct Interactions:
     ```
     Strong interaction example:
     Feature1 × Feature2 has high correlation with outcome
     Both features have high individual SHAP values
     ```

   - Conditional Effects:
     ```
     Feature1's impact changes based on Feature2's value
     Shows up as patterns in 2D SHAP dependency plots
     ```

   Key Metrics for Interactions:
   - Interaction Strength: Correlation between combined effects
   - Consistency: How stable the interaction is
   - Direction: Whether features amplify or dampen each other

Example Interpretation:
```python
# Strong interaction
if interactions.loc[0, 'interaction_strength'] > 0.5:
    print(f"Strong interaction between {interactions.loc[0, 'feature1']} "
          f"and {interactions.loc[0, 'feature2']}")
```

Common Patterns to Look For:
1. High Individual Impact + High Interaction = Key feature pair
2. Low Individual Impact + High Interaction = Hidden relationship
3. High Individual Impact + Low Interaction = Independent important features


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

class ShapVisualizer:
    def __init__(self, X, y, model, random_state=42):
        """Initialize with data and model"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
            
        self.X = X
        self.y = y
        self.model = model
        self.feature_names = X.columns
        
        # Split and fit
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Initialize explainer
        self.background = shap.sample(self.X_train, 100, random_state=random_state)
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            self.background,
            output_names=['Class 0', 'Class 1']
        )
        
        # Calculate SHAP values once
        self.shap_values = self.explainer.shap_values(self.X_test.iloc[:100])
        if isinstance(self.shap_values, list):
            self.shap_values_class1 = self.shap_values[1]
        else:
            self.shap_values_class1 = self.shap_values

    def plot_feature_importance_overview(self):
        """Create a comprehensive feature importance visualization"""
        plt.figure(figsize=(15, 10))
        
        # Main summary plot
        plt.subplot(2, 2, 1)
        shap.summary_plot(
            self.shap_values_class1,
            self.X_test.iloc[:100],
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Values Overview (Violin)')
        
        # Bar plot of absolute SHAP values
        plt.subplot(2, 2, 2)
        shap.summary_plot(
            self.shap_values_class1,
            self.X_test.iloc[:100],
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title('Feature Importance (Mean |SHAP|)')
        
        # Distribution of SHAP values by feature
        plt.subplot(2, 2, 3)
        mean_shap = np.abs(self.shap_values_class1).mean(0)
        std_shap = np.abs(self.shap_values_class1).std(0)
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean |SHAP|': mean_shap,
            'Std |SHAP|': std_shap
        }).sort_values('Mean |SHAP|', ascending=True)
        
        plt.barh(range(len(feature_importance)), feature_importance['Mean |SHAP|'])
        plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
        plt.xlabel('Mean |SHAP|')
        plt.title('Feature Importance with Variability')
        
        # Add error bars
        plt.errorbar(
            feature_importance['Mean |SHAP|'],
            range(len(feature_importance)),
            xerr=feature_importance['Std |SHAP|'],
            fmt='none',
            c='red',
            alpha=0.5
        )
        
        plt.tight_layout()
        
    def plot_feature_interaction_matrix(self, top_n=10):
        """Plot feature interaction matrix"""
        # Calculate interaction values
        feature_importance = np.abs(self.shap_values_class1).mean(0)
        top_features_idx = np.argsort(feature_importance)[-top_n:]
        top_features = self.feature_names[top_features_idx]
        
        interaction_matrix = np.zeros((top_n, top_n))
        for i, idx1 in enumerate(top_features_idx):
            for j, idx2 in enumerate(top_features_idx):
                interaction = self._calculate_interaction_strength(idx1, idx2)
                interaction_matrix[i, j] = interaction
                
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            interaction_matrix,
            xticklabels=top_features,
            yticklabels=top_features,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f'
        )
        plt.title('Feature Interaction Strength Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
    def plot_feature_dependence(self, feature_idx, interaction_idx=None):
        """Plot dependence plot for a feature with optional interaction"""
        plt.figure(figsize=(10, 6))
        if interaction_idx is None:
            shap.dependence_plot(
                feature_idx,
                self.shap_values_class1,
                self.X_test.iloc[:100],
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {self.feature_names[feature_idx]}')
        else:
            shap.dependence_plot(
                (feature_idx, interaction_idx),
                self.shap_values_class1,
                self.X_test.iloc[:100],
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f'SHAP Interaction Plot: {self.feature_names[feature_idx]} vs {self.feature_names[interaction_idx]}')
            
    def plot_decision_path(self, instance_idx=0):
        """Plot decision path for a single instance"""
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values_class1[instance_idx],
                base_values=self.explainer.expected_value[1],
                data=self.X_test.iloc[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title('Decision Path Analysis for Single Instance')
        
    def _calculate_interaction_strength(self, idx1, idx2):
        """Calculate interaction strength between two features"""
        f1_values = self.X_test.iloc[:100, idx1].values
        f2_values = self.X_test.iloc[:100, idx2].values
        f1_shap = self.shap_values_class1[:, idx1]
        f2_shap = self.shap_values_class1[:, idx2]
        
        if idx1 == idx2:
            return 1.0
        
        return np.abs(np.corrcoef(f1_shap * f2_values, f2_shap * f1_values)[0, 1])

def demonstrate_shap_visualizations():
    """Demonstrate all SHAP visualizations with example data"""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=False
    )
    
    # Create meaningful feature names
    feature_names = [
        'Strong_Positive',
        'Strong_Negative',
        'Moderate_Positive',
        'Moderate_Negative',
        'Weak_Positive',
        'Redundant_1',
        'Redundant_2',
        'Noise_1',
        'Noise_2',
        'Noise_3'
    ]
    
    X = pd.DataFrame(X, columns=feature_names)
    
    # Initialize visualizer
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    visualizer = ShapVisualizer(X, y, model)
    
    # Generate all plots
    print("Generating Feature Importance Overview...")
    visualizer.plot_feature_importance_overview()
    plt.show()
    
    print("\nGenerating Feature Interaction Matrix...")
    visualizer.plot_feature_interaction_matrix()
    plt.show()
    
    print("\nGenerating Feature Dependence Plots...")
    # Plot top 3 features
    importance = np.abs(visualizer.shap_values_class1).mean(0)
    top_features = np.argsort(importance)[-3:]
    for idx in top_features:
        visualizer.plot_feature_dependence(idx)
        plt.show()
    
    print("\nGenerating Decision Path Analysis...")
    visualizer.plot_decision_path()
    plt.show()
    
    return visualizer

if __name__ == "__main__":
    demonstrate_shap_visualizations()
```
