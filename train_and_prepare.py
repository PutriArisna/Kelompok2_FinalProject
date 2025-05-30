import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

MODEL_PATH = "model.pkl"

# Train and save the model
def train_model(data_path="Loan_Predict.csv"):
    df = pd.read_csv(data_path)

    # Feature engineering
    def categorize_income(income):
        if income < 125000:
            return 'Low'
        elif income < 500000:
            return 'Lower-Middle'
        elif income < 3000000:
            return 'Middle'
        else:
            return 'High'

    df['Income_Level'] = df['Income'].apply(categorize_income)

    # One-hot encoding
    one_hot = ['Car_Ownership', 'Married/Single', 'House_Ownership']
    for cat in one_hot:
        dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    df.drop(columns=one_hot, inplace=True)

    # Ordinal encoding
    df['Income_Level'] = df['Income_Level'].map({'High': 3, 'Middle': 2, 'Lower-Middle': 1, 'Low': 0})

    # Drop unnecessary columns
    df.drop(['Id', 'Income'], axis=1, inplace=True)

    # Separate features and target
    X = df.drop(['Risk_Flag'], axis=1)
    y = df['Risk_Flag']

    # Save full original X_test for future analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Save data before encode
    X_test_unscaled = X_test.copy()
    
    # Remove outliers
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    Q1 = X_train[numeric_cols].quantile(0.25)
    Q3 = X_train[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    filtered = ~((X_train[numeric_cols] < (Q1 - 1.5 * IQR)) | (X_train[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    X_train = X_train[filtered]
    y_train = y_train[filtered]
    
    # Target Encoding
    def target_encode_columns(X_train, X_test, y_train, columns_to_encode):
        train = X_train.copy()
        test = X_test.copy()
        train['Risk_Flag'] = y_train
        overall_mean = y_train.mean()

        for col in columns_to_encode:
            target_mean = train.groupby(col)['Risk_Flag'].mean()
            train[col] = train[col].map(target_mean)
            test[col] = test[col].map(target_mean).fillna(overall_mean)

        return train.drop(columns='Risk_Flag'), test
        
    X_train, X_test = target_encode_columns(X_train, X_test, y_train, ['Profession', 'STATE', 'CITY'])
    
    # Standard scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Undersampling
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Train model
    model = XGBClassifier(learning_rate=0.546, max_depth=6, subsample=1,colsample_bytree=1, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save test set and feature importance
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    X_test_unscaled.to_csv("X_test_unscaled.csv", index=False)

    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importance.to_csv("feature_importance.csv", index=False)

    return model, X_test_unscaled, y_test

# Load trained model
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

