
# Loan Risk Prediction System 
> Final Project Stage 2 — Kelompok 2

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)](https://scikit-learn.org/stable/)

---

## 📌 Overview
This project builds a Machine Learning model to **predict the risk of loan default** based on customer profile data.  
The goal is to help financial institutions **automate** and **optimize** loan approvals, especially under conditions of **high application volume** and **limited staff**.

---

## 💬 Problem Statement
- High volume of loan applications
- High loan default rate
- Limited staff to manually assess applications

---
## 🎯 Goals
- Enhance efficiency in loan evaluations to support limited staff.

---
  
## 🎯 Objectives
- Develop a predictive model to classify borrowers into **high-risk** and **low-risk**.
- Improve decision-making accuracy.
- Minimize Financial Risk

---

## 📊 Business Metrics
- **Default Reduction Rate**: Decrease number of loan defaults
- **Approval Rate Optimization**: Increase approvals of low-risk applicants
- **Net Income Increase**: Faster decisions → more loans processed
- **NPL Reduction Rate**: Minimize loans that turn into bad debt

---

## 📚 Dataset Information
- **Source**: Provided dataset for project
- **Total Rows**: 252,000
- **Missing Values**: None
- **Duplicates**: None
- **Features**:
  - `Id` : Unique borrower ID
  - `Income` : Annual income 
  - `Age` : Age of borrower
  - `Experience` : Years of work experience
  - `Married/Single` : Marital status
  - `House_Ownership` : Type of house ownership (Own, Rent, or No Rent No Owned)
  - `Car_Ownership` : Whether the applicant owns a car (Yes or No).
  - `Profession` : Applicant’s profession.
  - `CITY` : City where the applicant resides.
  - `STATE` : State where the applicant resides.
  - `CURRENT_JOB_YRS` : Number of years in the current job.
  - `CURRENT_HOUSE_YRS` : Number of years living in the current house.
  - `Risk_Flag` (target: 1 = high risk, 0 = low risk)

---

## 🔎 Key Insights
- Majority of applicants are between **45-50 years old**.
- Most applicants **do not own** a house.
- **Police officer, Chartered accountant, Education administrator, Army officer** → Top professions with the highest risk probability.
- **Bhubaneswar, Gwalior, and Bettiah** are the top three cities with the highest default risk probabilities, exceeding 25%.
- **Manipur, Tripura, and Kerala** as the top three states with the highest default risk probabilities, with Manipur standing out at over 21% risk.
- High-risk applicants often have **fewer job experience and fewer years at current job.**
- **No missing or duplicated data** detected, ensuring clean input for modeling.
- **Data imbalance** with majority (87.7%) of applicants are classified as "Not Risk", while only 12.3% fall into the "Risk"

---

## 🛠️ Methodology
1. **Data Cleansing**
   - Corrected data types (e.g., Id to string).
   - Checked for nulls and duplicates.
2. **Exploratory Data Analysis (EDA)**
   - Univariate analysis (distribution of each feature).
   - Multivariate analysis (relationships with target Risk_Flag).
   - Analyze feature importance and multicollinearity to refine inputs.
3. **Feature Engineering** (Encoding, Scaling)
   - Numerical Binning.
   - Categorical encoding (one-hot encoding / label encoding).
   - Handling Outliers
   - Feature scaling.
4. **Model Building** 
   - Developed classification models.
   - Train multiple models and apply hyperparameter tuning (e.g., RandomizedSearchCV or GridSearchCV).
5. **Evaluation Metrics**
    - Validate model performance using cross-validation techniques.
    - Emphasized metrics: Recall, F1-Score.
6. **Model Deployment Readiness**
    - Serialize the final model (using Pickle or joblib).
    - Create an inference pipeline for future real-time predictions.
7. **Business Impact Analysis**
    - Estimate the time saved per loan evaluation and increase in daily processing capacity.
    - Quantify financial benefits (loss prevention, revenue gain) by comparing model-based decisions vs. manual decisions.

---

## 🛠 Technologies Used
- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🗂 Project Structure
```plaintext
Final_Project_stage_2_Kelompok_2/
│
├── dataset/
│    └── Loan_Predict.csv                   # Data applicant untuk modeling
│
├── Final_Project_stage_2_Kelompok_2.ipynb  # Main notebook
│
├── requirements.txt                        # Daftar dependensi Python
│
└── README.md                               # Project documentation

```

---

## 🚀 Future Work
- Add model explainability tools (e.g., SHAP, LIME)
- Experiment with stronger models
- Build an API for integration into loan approval software

---

# 📢 How to Run This Project
```bash
# Clone this repository:
  git clone https://github.com/your-username/loan-risk-prediction.git

# Open the Jupyter Notebook:
  jupyter notebook Final_Project_stage_2_Kelompok_2.ipynb

# Install required packages:
   pip install -r requirements.txt
```


---

# 📩 Contact
Feel free to reach out if you have suggestions or questions!  
**Kelompok 2** - Rakamin Academy Final Project
