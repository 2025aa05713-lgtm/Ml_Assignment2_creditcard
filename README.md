### Machine Learning Assignment 2

**Student ID**: 2025AA05713
**Submission Date**:9th February 2026

---

## 📋 Assignment Overview

This project implements a comprehensive credit card fraud detection system using 6 different machine learning classification algorithms. The system includes:

- **6 Classification Models**: Logistic Regression, Decision Tree, K-Nearest Neighbor, Naive Bayes, Random Forest, and Gradient Boosting
- **6 Evaluation Metrics**: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)
- **Interactive Web Application**: Built with Streamlit for model comparison and predictions
- **Complete Pipeline**: From data preprocessing to model deployment

---

## 📊 Dataset Description

**Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Dataset Characteristics:
- **Source**: Kaggle (European cardholders, September 2013)
- **Time Period**: 2 days of transactions
- **Total Transactions**: 284,807
- **Fraud Cases**: 492 (0.172% of total)
- **Features**: 30 (V1-V28: PCA transformed features, Time, Amount)
- **Target Variable**: Class (0 = Normal, 1 = Fraud)

### Feature Description:
- **Time**: Seconds elapsed between each transaction and the first transaction
- **V1-V28**: Principal components obtained via PCA (anonymized for privacy)
- **Amount**: Transaction amount
- **Class**: Target variable (1 for fraudulent transactions, 0 otherwise)

### Data Challenges:
- **Highly Imbalanced**: Only 0.172% fraud cases
- **Anonymized Features**: V1-V28 are PCA transformed
- **Privacy Protection**: Original features not available

---

## 🏗️ Methodology

### Data Preprocessing:
1. **Handling Imbalance**: Used undersampling to create balanced dataset (492 fraud + 492 normal)
2. **Feature Scaling**: Applied StandardScaler for distance-based algorithms (Logistic Regression, KNN)
3. **Train-Test Split**: 80-20 stratified split to maintain class distribution

### Model Implementation:
All models were trained on the balanced dataset and evaluated on the held-out test set.

---

## 📈 Model Performance Comparison

| Model                | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC      |
|----------------------|----------|-----------|-----------|--------|----------|----------|
| Logistic Regression | 0.9340   | 0.9840    | 0.9474    | 0.9184 | 0.9326   | 0.8684   |
| Decision Tree       | 0.9137   | 0.9134    | 0.9010    | 0.9286 | 0.9146   | 0.8278   |
| K-Nearest Neighbor  | 0.8985   | 0.9517    | 0.9432    | 0.8469 | 0.8925   | 0.8010   |
| Naive Bayes (Gaussian)| 0.8376 | 0.9555    | 0.9714    | 0.6939 | 0.8095   | 0.7038   |
| Random Forest       | 0.9492   | 0.9783    | 0.9783    | 0.9184 | 0.9474   | 0.9001   |
| Gradient Boosting   | 0.9188   | 0.9407    | 0.9100    | 0.9286 | 0.9192   | 0.8377   |

**Best Performing Model**: Random Forest (Highest Accuracy: 0.9492, Best MCC: 0.9001)

---

## 🔍 Model Analysis & Observations

### 1. Logistic Regression
- **Performance**: Strong baseline with 0.934 accuracy
- **Strengths**: Fast training, interpretable coefficients, good AUC (0.984)
- **Best Use**: Quick predictions, when model interpretability is important

### 2. Decision Tree
- **Performance**: Good balance with 0.914 accuracy
- **Strengths**: Highly interpretable, no feature scaling required
- **Limitations**: Prone to overfitting (controlled with max_depth=10)

### 3. K-Nearest Neighbor
- **Performance**: Moderate accuracy (0.898), good AUC (0.952)
- **Strengths**: Instance-based learning, no training phase
- **Limitations**: Computationally expensive, sensitive to feature scaling

### 4. Naive Bayes (Gaussian)
- **Performance**: Lowest accuracy (0.838), but good precision (0.971)
- **Strengths**: Fast, works well with high-dimensional data
- **Limitations**: Assumes feature independence, poor recall (0.694)

### 5. Random Forest ⭐
- **Performance**: Best overall (0.949 accuracy, 0.900 MCC)
- **Strengths**: Robust, reduces overfitting, handles non-linear relationships
- **Best Use**: Production systems requiring high accuracy

### 6. Gradient Boosting
- **Performance**: Strong performance (0.919 accuracy)
- **Strengths**: Handles complex patterns, good with imbalanced data
- **Note**: Used sklearn's GradientBoostingClassifier (alternative to XGBoost)

### Key Insights:
- **Ensemble Methods Superior**: Random Forest and Gradient Boosting outperformed single models
- **Precision vs Recall Trade-off**: Models show varying precision-recall balances
- **AUC Reliability**: All models show good discriminative ability (AUC > 0.91)
- **Imbalance Handling**: Undersampling helped models learn fraud patterns effectively

---

## 🚀 Deployment & Web Application

### Streamlit Application Features:
- **Model Selection**: Dropdown to choose from 6 trained models
- **Performance Metrics**: Real-time display of all 6 evaluation metrics
- **Confusion Matrix**: Visual representation of prediction results
- **Model Comparison**: Side-by-side performance visualization
- **Prediction Interface**: Upload CSV files for batch predictions
- **Interactive Dashboard**: Modern UI with responsive design

### How to Run Locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```


### Live Application:
[Streamlit Community Cloud Link](https://mlassignment2creditcard.streamlit.app/)

---

## 📁 Project Structure

```
ML_Assignment2_CreditCardFraud/
├── ML_Assignment2_CreditCardFraud.ipynb  # Main analysis notebook
├── app.py                                # Streamlit web application
├── requirements.txt                      # Python dependencies
├── README.md                            # Project documentation
├── creditcard.csv                       # Dataset (not included in repo)
├── model_results.csv                    # Model performance results
├── test_data_sample.csv                 # Sample test data
└── models/                              # Trained model files
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── gradient_boosting.pkl
    └── scaler.pkl
```

---

## 🛠️ Technical Implementation

### Libraries Used:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Web Framework**: Streamlit

### Environment:
- **Python Version**: 3.14.1
- **Virtual Environment**: .venv
- **Platform**: macOS

---

## 📊 Results Summary

- **Best Model**: Random Forest with 94.92% accuracy
- **Most Interpretable**: Decision Tree and Logistic Regression
- **Fastest Training**: Naive Bayes and Logistic Regression
- **Best AUC Score**: Logistic Regression (0.984)
- **Highest Precision**: Naive Bayes (0.971)
- **Best MCC Score**: Random Forest (0.900)

---

## 🔗 Links & Resources

- **GitHub Repository**: [(https://github.com/2025aa05713-lgtm/Ml_Assignment2_creditcard.git)]
- **Live Application**: [(https://mlassignment2creditcard.streamlit.app/)]
- **Dataset Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Streamlit Documentation**: [streamlit.io](https://streamlit.io)

---

## 📝 Notes

- Dataset contains sensitive financial information - handle with care
- Models trained on undersampled data for balanced learning
- All models saved using joblib for deployment compatibility
- Web application includes prediction functionality for new data

---

**BITS Pilani - Machine Learning Assignment 2**  
**Credit Card Fraud Detection System**  
**Submitted by: [Arijit Dutta]**  
**ID: 2025AA05713**