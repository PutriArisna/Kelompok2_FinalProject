import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate
import pickle

# Load data and model
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    y_test = pd.read_csv("y_test.csv").squeeze()
    X_test_unscaled = pd.read_csv("X_test_unscaled.csv")
    feature_df = pd.read_csv("feature_importance.csv")
    return model, None, y_test, X_test_unscaled, feature_df

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
def plot_feature_importance(feature_df, top_n=10):
    top_feats = feature_df.sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6, 4))  # smaller, more compact
    sns.barplot(x="Importance", y="Feature", data=top_feats, ax=ax, palette="Blues")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig
# Risk probability by top 2 features (only top 5 categories shown)
def plot_risk_by_top_features(model, X_test, y_test, X_test_unscaled, feature_df):
    top2 = feature_df.sort_values("Importance", ascending=False)["Feature"].head(2).tolist()
    X_test_unscaled = X_test_unscaled.copy()
    probs = model.predict_proba(X_test)[:, 1]
    X_test_unscaled["Risk_Prob"] = probs

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, feat in enumerate(top2):
        grouped = X_test_unscaled.groupby(feat)["Risk_Prob"].mean().sort_values(ascending=False).head(5).reset_index()
        sns.barplot(x=feat, y="Risk_Prob", data=grouped, ax=axes[i], palette="Blues")
        axes[i].set_title(f"Top 5 {feat} by Avg Risk Probability")
        axes[i].set_ylabel("Average Risk Probability")
        axes[i].set_xlabel(feat)
        axes[i].tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return [fig]

# Risk category assignment (with probability threshold)
def risk_categories(model, X_test, y_test, X_test_unscaled, filter_need_review=False):
    probs = model.predict_proba(X_test)[:, 1]
    result_df = X_test_unscaled.copy()
    result_df['True_Label'] = y_test.values
    result_df['Probability'] = probs

    def categorize(prob):
        if prob <= 0.3:
            return 'Not Risk'
        elif prob <= 0.6:
            return 'Need Review'
        else:
            return 'Risk'

    result_df['Risk_Category'] = result_df['Probability'].apply(categorize)

    if filter_need_review:
        return result_df[result_df['Risk_Category'] == 'Need Review'].reset_index(drop=True)

    return result_df.reset_index(drop=True)

