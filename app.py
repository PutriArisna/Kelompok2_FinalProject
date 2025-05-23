import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_elements import elements, mui
from train_and_prepare import train_model, load_model
from analysis import (
    load_assets,
    get_evaluation_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_risk_by_top_features,
    risk_categories
)

# Circular metric display
def show_metric_circles(metrics):
    with elements("metrics"):
        for name, score in metrics.items():
            percent = round(score * 100)
            mui.Typography(name)
            mui.CircularProgress(
                variant="determinate",
                value=percent,
                sx={"color": "#2196f3", "marginBottom": "10px"},
                size=100,
                thickness=5
            )
            mui.Typography(f"{percent}%")

st.set_page_config(page_title="Loan Risk Dashboard", layout="wide")
st.title("🏦 Loan Risk Classification Dashboard")

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Train model manually
if st.button("🔁 Load and Train Model"):
    model, _, y_test = train_model()
    st.session_state.model_loaded = True
    st.success("✅ Model trained and test data ready.")

# Load saved model and supporting data
if st.session_state.model_loaded or st.button("Force Load Assets"):
    model, _, y_test, X_test_unscaled, feature_df = load_assets()
    st.session_state.model_loaded = True

    # File uploader for X_test
    uploaded_file = st.file_uploader("📂 Upload your X_test CSV file", type=["csv"])

    if uploaded_file is not None:
        X_test = pd.read_csv(uploaded_file)
        st.success("✅ X_test loaded successfully.")

        # Evaluation Metrics
        if st.button("📊 Show Evaluation Metrics"):
            metrics = get_evaluation_metrics(model, X_test, y_test)
            show_metric_circles(metrics)

        # Analysis Visuals
        if st.button("📈 Analysis Plots"):
            st.subheader("📌 Confusion Matrix")
            fig = plot_confusion_matrix(model, X_test, y_test)
            st.pyplot(fig, use_container_width=False)

            st.subheader("🔥 Feature Importances")
            fig_feat = plot_feature_importance(feature_df)
            st.pyplot(fig_feat, use_container_width=False)

            st.subheader("🎯 Risk Probability by Top 2 Features")
            plots = plot_risk_by_top_features(model, X_test, y_test, X_test_unscaled, feature_df)
            for fig in plots:
                st.pyplot(fig, use_container_width=False)

        # Need Review Table
        if st.button("⚠️ Show 'Need Review' Data"):
            st.subheader("⚠️ Applicants That Need Further Review")
            review_df = risk_categories(model, X_test, y_test, X_test_unscaled, filter_need_review=True)
            st.write(f"Total 'Need Review' entries: {len(review_df)}")
            st.dataframe(review_df)

    else:
        st.warning("⚠️ Please upload X_test.csv to continue.")

