import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def MarvellousTitanicLogistic():
    # Step 1: Load data
    st.title("Titanic Survival Prediction Using Logistic Regression")
    
    titanic_data = pd.read_csv("MarvellousTitanicDataset.csv")
    
    st.subheader("First five entries from the loaded dataset")
    st.write(titanic_data.head())

    st.write("Number of passengers: ", len(titanic_data))

    # Step 2: Analyze data
    st.subheader("Survived vs Non-survived Passengers")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=titanic_data, x="Survived", ax=ax1)
    ax1.set_title("Survived vs Non-survived Passengers")
    st.pyplot(fig1)

    st.subheader("Survived vs Non-survived by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=titanic_data, x="Survived", hue="Sex", ax=ax2)
    ax2.set_title("Survived vs Non-survived by Gender")
    st.pyplot(fig2)

    st.subheader("Survived vs Non-survived by Class")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=titanic_data, x="Survived", hue="Pclass", ax=ax3)
    ax3.set_title("Survived vs Non-survived by Class")
    st.pyplot(fig3)

    st.subheader("Distribution of Age")
    fig4, ax4 = plt.subplots()
    titanic_data["Age"].plot.hist(ax=ax4)
    ax4.set_title("Distribution of Age")
    st.pyplot(fig4)

    st.subheader("Distribution of Fare")
    fig5, ax5 = plt.subplots()
    titanic_data["Fare"].plot.hist(ax=ax5)
    ax5.set_title("Distribution of Fare")
    st.pyplot(fig5)

    # Step 3: Data cleaning
    titanic_data.drop(columns=["zero"], axis=1, inplace=True, errors='ignore')

    # Handle categorical variables
    titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Pclass"], drop_first=True)

    titanic_data.drop(columns=["sibsp", "Parch", "Embarked"], axis=1, inplace=True)

    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]

    # Standardize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Step 4: Data Training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train)
   
    # Step 5: Testing dataset
    prediction = logistic_model.predict(x_test)

    # Step 6: Calculate Accuracy
    st.subheader("Classification report of Logistic Regression")
    st.text(classification_report(y_test, prediction))
    
    st.subheader("Confusion matrix of Logistic Regression")
    st.write(confusion_matrix(y_test, prediction))

    st.subheader("Accuracy of Logistic Regression")
    st.write(accuracy_score(y_test, prediction))

def main():
    MarvellousTitanicLogistic()

if __name__ == "__main__":
    main()
