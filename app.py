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
    risk_categories,
    plot_bar_distribution,
    plot_hist_distribution,
    plot_top5_bar_distribution
)

# Circular metric display in one horizontal row
def show_metric_circles(metrics):
    with elements("metrics"):
        with mui.Stack(direction="row", spacing=4):
            for name, score in metrics.items():
                percent = round(score * 100)
                with mui.Box(textAlign="center"):
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
    st.success("✅ Model trained ready.")

# Load saved model and supporting data
if st.session_state.model_loaded:
    model, _, y_test, X_test_unscaled, feature_df = load_assets()
    st.session_state.model_loaded = True

    # File uploader for X_test
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        X_test = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully.")

        # Evaluation Metrics
        if st.button("Show Evaluation Metrics"):
            metrics = get_evaluation_metrics(model, X_test, y_test)
            show_metric_circles(metrics)

        # Analysis Visuals
        if st.button("📈 Analysis Plots"):
            col1, col2= st.columns([0.3, 0.6])
            with col1:
                st.subheader("Confusion Matrix")
                fig = plot_confusion_matrix(model, X_test, y_test)
                st.pyplot(fig, use_container_width=True)
            with col2:
                st.subheader("Feature Importances")
                fig_feat = plot_feature_importance(feature_df)
                st.pyplot(fig_feat, use_container_width=True)

        # Risk Category Filter Dropdown
        risk_option = st.selectbox("📂 Select Risk Category to Display",["Choose", "Risk", "Need Review", "Not Risk"])

        if risk_option != "Choose":
            st.subheader(f"⚠️ Applicants in '{risk_option}' Category")
            filtered_df = risk_categories(model, X_test, y_test, X_test_unscaled, risk_category=risk_option)
            st.write(f"Total '{risk_option}' entries: {len(filtered_df)}")
            st.dataframe(filtered_df)
  
            # Risk plots under table
            st.subheader(f"Top 2 Features Risk Breakdown for '{risk_option}'")
            category_plots = plot_risk_by_top_features(model, X_test, y_test, X_test_unscaled, feature_df, category_filter=risk_option)
            for fig in category_plots:
                st.pyplot(fig, use_container_width=True)
                
            st.subheader(f"📊 Feature Distributions for '{risk_option}' Category")
    
            cat_names = ['Car_Ownership_yes', 'Income_Level', 'CURRENT_HOUSE_YRS']
            num_names = ['CURRENT_JOB_YRS', 'Age', 'Experience']
            top5s = [('CITY', "Top 5 Cities"), ('Profession', "Top 5 Professions")]
            
            for i in range(3):  # For each index
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.pyplot(plot_bar_distribution(filtered_df, cat_names[i]))
                with col2:
                    st.pyplot(plot_hist_distribution(filtered_df, num_names[i]))
                with col3:
                    # Show CITY for i==0, PROFESSION for i==1, nothing for i==2
                    if i < 2:
                        st.pyplot(plot_top5_bar_distribution(filtered_df, top5s[i][0]))
                    else:
                        st.write("")
                        
    else:
        st.warning("⚠️ Please upload .csv to continue.")

