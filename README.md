# titanic-survival-prediction
Machine learning project that predicts passenger survival using the Titanic Dataset with data preprocessing, model training, and deployment using Streamlit.
# Titanic Survival Prediction 🚢

This project is a **Machine Learning classification project** that predicts whether a passenger survived or not on the Titanic using the Titanic dataset.

## 📌 Project Overview

The goal of this project is to build a machine learning model that predicts passenger survival based on features like:

- Passenger Class
- Age
- Sex
- Fare
- Embarked
- SibSp
- Parch

The dataset used in this project is the **Titanic dataset** from Kaggle.

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit (for deployment)

## ⚙️ Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- AdaBoost Classifier

## 📊 Workflow

1. Data Loading
2. Data Cleaning
3. Handling Missing Values
4. Feature Engineering
5. Encoding Categorical Variables
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Deployment with Streamlit

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

## 🚀 Deployment

The trained model is deployed using **Streamlit** to create an interactive web application where users can input passenger details and predict survival.

## 📂 Project Structure
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

st.title("Titanic Survival Prediction 🚢")

# ------------------ Upload ZIP ------------------
uploaded_zip = st.file_uploader("Upload Titanic dataset ZIP file", type="zip")

if uploaded_zip is not None:
    with zipfile.ZipFile(BytesIO(uploaded_zip.read())) as z:
        st.write("Files in ZIP:", z.namelist())
        with z.open('train.csv') as f:
            train_df = pd.read_csv(f)
        with z.open('test.csv') as f:
            test_df = pd.read_csv(f)
    st.success("Files loaded successfully!")
else:
    st.warning("Please upload the Titanic ZIP file")
    st.stop()

# ------------------ Combine for preprocessing ------------------
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# ------------------ Preprocessing ------------------
def preprocess(df):
    # Titles
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
        'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin → Deck
    df['Deck'] = df['Cabin'].str.extract('([A-Za-z])', expand=False)
    df['Deck'] = df['Deck'].fillna('U')
    
    # Drop unnecessary columns
    df.drop(['PassengerId','Name','Ticket','Cabin','dataset'], axis=1, inplace=True, errors='ignore')
    
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    
    # One-hot encode categorical
    df = pd.get_dummies(df, columns=['Embarked','Deck','Title'], drop_first=True)
    
    # Log Fare + MinMax scaling
    df['Fare_log'] = np.log1p(df['Fare'])
    scaler = MinMaxScaler()
    df['Age_minmax'] = scaler.fit_transform(df[['Age']])
    df['Fare_log_minmax'] = scaler.fit_transform(df[['Fare_log']])
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Drop original columns
    df.drop(['Age','Fare','Fare_log','SibSp','Parch'], axis=1, inplace=True)
    
    # Convert booleans to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df

processed_df = preprocess(combined_df.copy())

# ------------------ Train Model ------------------
train_processed = processed_df[processed_df['Survived'].notnull()]
X_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived'].astype(int)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

st.subheader("Enter Passenger Details")

# ------------------ Sidebar Inputs ------------------
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Pclass", [1,2,3])
age = st.slider("Age", 0, 80, 30)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
sibsp = st.number_input("SibSp", 0, 10, 0)
parch = st.number_input("Parch", 0, 10, 0)
embarked = st.selectbox("Embarked", ['C','Q','S'])
cabin = st.selectbox("Cabin Deck", ['A','B','C','D','E','F','G','T','U'])
title = st.selectbox("Title", ['Mr','Mrs','Miss','Master','Rare'])

# ------------------ Prepare input dynamically ------------------
feature_cols = X_train.columns

# Start with all zeros
input_dict = dict.fromkeys(feature_cols, 0)

# Fill numeric features
input_dict['Pclass'] = pclass
input_dict['Sex'] = 0 if sex=='male' else 1
input_dict['Age_minmax'] = age / 80
input_dict['Fare_log_minmax'] = np.log1p(fare) / np.log1p(600)
input_dict['FamilySize'] = sibsp + parch + 1

# Categorical features
if 'Embarked_Q' in input_dict:
    input_dict['Embarked_Q'] = 1 if embarked=='Q' else 0
if 'Embarked_S' in input_dict:
    input_dict['Embarked_S'] = 1 if embarked=='S' else 0

# Deck features
for deck in ['B','C','D','E','F','G','T','U']:
    col = f'Deck_{deck}'
    if col in input_dict:
        input_dict[col] = 1 if cabin==deck else 0

# Title features
for t in ['Master','Miss','Mrs','Rare','Mr']:
    col = f'Title_{t}'
    if col in input_dict:
        input_dict[col] = 1 if title==t else 0

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# ------------------ Predict ------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_df)
    result = "✅ Survived" if prediction[0]==1 else "❌ Did not survive"
    st.success(f"Prediction: {result}")
