import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate
import pickle

# Load data and model
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()
    X_test_unscaled = pd.read_csv("X_test_unscaled.csv")
    feature_df = pd.read_csv("feature_importance.csv")
    return model, X_test, y_test, X_test_unscaled, feature_df

# Evaluation metrics
def get_evaluation_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return {
        "Recall (Test Set)": recall,
        "ROC AUC (Test Set)": roc_auc,
    }

# Confusion matrix plot
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3)) 
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor='black'
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig
    
# Feature importance plot
def plot_feature_importance(feature_df, top_n=20):
    top_feats = feature_df.sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6, 4))  # smaller, more compact
    sns.barplot(x="Importance", y="Feature", data=top_feats, ax=ax, palette="Blues")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig
    
# Create dataframe with risk categories
def get_risk_dataframe(model, X_test, y_test, X_test_unscaled):
    probs = model.predict_proba(X_test)[:, 1]
    result_df = X_test_unscaled.copy()
    result_df['Probability'] = probs

    def categorize(prob):
        if prob <= 0.3:
            return 'Not Risk'
        elif prob <= 0.6:
            return 'Need Review'
        else:
            return 'Risk'

    result_df['Risk_Category'] = result_df['Probability'].apply(categorize)
    return result_df.reset_index(drop=True) 
    
# Top 2 feature breakdown for selected category
def plot_risk_by_top_features(model, X_test, y_test, X_test_unscaled, feature_df, category_filter=None):
    top_features = feature_df.sort_values(by='Importance', ascending=False).head(2)['Feature'].tolist()
    result_df = get_risk_dataframe(model, X_test, y_test, X_test_unscaled)

    if category_filter:
        result_df = result_df[result_df['Risk_Category'] == category_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, feature in enumerate(top_features):
        top_vals = result_df.groupby(feature)['Probability'].mean().sort_values(ascending=False).head(5)
        sns.barplot(x=top_vals.index, y=top_vals.values, ax=axes[i], palette="Blues")
        axes[i].set_title(f"{feature} by Avg Risk Probability")
        axes[i].set_ylabel("Average Risk Probability")
        axes[i].set_xlabel(feature)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return [fig]

# Risk categories table with filtering
def risk_categories(model, X_test, y_test, X_test_unscaled, risk_category=None):
    result_df = get_risk_dataframe(model, X_test, y_test, X_test_unscaled)
    if risk_category in ['Risk', 'Need Review', 'Not Risk']:
        return result_df[result_df['Risk_Category'] == risk_category].reset_index(drop=True)
    return result_df

def plot_bar_distribution(df, column):
    # Calculate mean probability per category
    avg_prob = df.groupby(column)['Probability'].mean().reset_index()
    avg_prob.columns = [column, 'Avg_Risk_Probability']

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=avg_prob, x=column, y='Avg_Risk_Probability', ax=ax, palette='Blues_d')
    ax.set_title(f'Avg Risk Probability by {column}')
    ax.set_xlabel(column)
    ax.set_ylabel("Average Risk Probability")
    ax.tick_params(axis='x')
    return fig
    
def plot_hist_distribution(df, column):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df[column], kde=True, ax=ax, bins=20, color='steelblue')
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    return fig

def plot_top5_bar_distribution(df, column):
    # Calculate mean probability and get top 5
    avg_prob = df.groupby(column)['Probability'].mean().sort_values(ascending=False).head(5).reset_index()
    avg_prob.columns = [column, 'Avg_Risk_Probability']

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=avg_prob, x=column, y='Avg_Risk_Probability', ax=ax, palette='Blues_d')
    ax.set_title(f"Top 5 {column} by Avg Risk Probability")
    ax.set_xlabel(column)
    ax.set_ylabel("Average Risk Probability")
    ax.tick_params(axis='x', rotation=45)
    return fig