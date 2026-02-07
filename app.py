import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
        margin: 0.5rem 0;
    }
    .model-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("### BITS Pilani - M.Tech (AIML/DSE) - Machine Learning Assignment 2")

# Sidebar for model selection
st.sidebar.title("🎯 Model Selection")
st.sidebar.markdown("---")

# Load models and data
@st.cache_resource
def load_models():
    models = {}
    model_names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'gradient_boosting']

    for name in model_names:
        try:
            models[name] = joblib.load(f'models/{name}.pkl')
        except FileNotFoundError:
            st.error(f"Model {name}.pkl not found. Please ensure models are trained and saved.")
            return None

    try:
        scaler = joblib.load('models/scaler.pkl')
        models['scaler'] = scaler
    except FileNotFoundError:
        st.error("Scaler not found. Please ensure scaler is saved.")
        return None

    return models

@st.cache_data
def load_data():
    try:
        test_data = pd.read_csv('test_data_sample.csv')
        results_df = pd.read_csv('model_results.csv')
        return test_data, results_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None

# Load resources
models = load_models()
test_data, results_df = load_data()

if models is None or test_data is None or results_df is None:
    st.error("Failed to load required files. Please ensure all models and data are available.")
    st.stop()

# Model selection
model_options = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'knn': 'K-Nearest Neighbor',
    'naive_bayes': 'Naive Bayes (Gaussian)',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting'
}

selected_model_key = st.sidebar.selectbox(
    "Select a Machine Learning Model:",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

selected_model_name = model_options[selected_model_key]
selected_model = models[selected_model_key]
scaler = models['scaler']

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 {selected_model_name} Performance Metrics")

    # Get metrics for selected model
    model_metrics = results_df[results_df['Model'] == selected_model_name].iloc[0]

    # Display metrics in cards
    metrics_cols = st.columns(3)

    with metrics_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Accuracy</h4>
            <h2>{model_metrics['Accuracy']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h4>AUC Score</h4>
            <h2>{model_metrics['AUC']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with metrics_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Precision</h4>
            <h2>{model_metrics['Precision']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h4>Recall</h4>
            <h2>{model_metrics['Recall']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with metrics_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <h4>F1 Score</h4>
            <h2>{model_metrics['F1']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h4>MCC</h4>
            <h2>{model_metrics['MCC']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("📈 Confusion Matrix")

    # Calculate confusion matrix for selected model
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']

    # Scale data if needed (for Logistic Regression and KNN)
    if selected_model_key in ['logistic_regression', 'knn']:
        X_test_scaled = scaler.transform(X_test)
        y_pred = selected_model.predict(X_test_scaled)
    else:
        y_pred = selected_model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'], ax=ax)
    ax.set_title(f'Confusion Matrix - {selected_model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    st.pyplot(fig)

# Model comparison section
st.markdown("---")
st.subheader("🔍 Model Comparison")

# Display comparison table
st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']))

# Visualization
st.subheader("📊 Performance Visualization")

# Create comparison chart
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, metric in enumerate(metrics_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.1])
    ax.tick_params(axis='x', rotation=45)

    # Highlight selected model
    if results_df[results_df['Model'] == selected_model_name].index[0] == idx % len(results_df):
        bars[idx % len(results_df)].set_color('#ff6b6b')

plt.tight_layout()
st.pyplot(fig)

# Prediction section
st.markdown("---")
st.subheader("🔮 Make Predictions")

uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type="csv")

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)

        # Validate columns
        expected_cols = X_test.columns.tolist()
        if not all(col in new_data.columns for col in expected_cols):
            st.error(f"CSV must contain these columns: {expected_cols}")
        else:
            # Scale data if needed
            if selected_model_key in ['logistic_regression', 'knn']:
                new_data_scaled = scaler.transform(new_data[expected_cols])
                predictions = selected_model.predict(new_data_scaled)
                probabilities = selected_model.predict_proba(new_data_scaled)[:, 1]
            else:
                predictions = selected_model.predict(new_data[expected_cols])
                probabilities = selected_model.predict_proba(new_data[expected_cols])[:, 1] if hasattr(selected_model, 'predict_proba') else predictions

            # Display results
            results_df_pred = new_data.copy()
            results_df_pred['Prediction'] = predictions
            results_df_pred['Fraud_Probability'] = probabilities
            results_df_pred['Prediction_Label'] = results_df_pred['Prediction'].map({0: 'Normal', 1: 'Fraud'})

            st.success("Predictions completed!")

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(results_df_pred))
            with col2:
                fraud_count = (predictions == 1).sum()
                st.metric("Predicted Fraud", fraud_count)
            with col3:
                fraud_rate = (fraud_count / len(predictions)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            # Show predictions table
            st.dataframe(results_df_pred[['Prediction_Label', 'Fraud_Probability']].head(50))

            # Download results
            csv = results_df_pred.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Credit Card Fraud Detection System</strong></p>
    <p>Built with Streamlit | Models trained on Credit Card Fraud Detection Dataset</p>
    <p>BITS Pilani - Machine Learning Assignment 2</p>
</div>
""", unsafe_allow_html=True)