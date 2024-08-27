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
    titanic_data = pd.read_csv("MarvellousTitanicDataset.csv")

    print("First five entries from loaded dataset: ")
    print(titanic_data.head()) 

    print("Number of passengers: " + str(len(titanic_data)))

    # Step 2: Analyze data
    plt.figure()
    sns.countplot(data=titanic_data, x="Survived").set_title("Survived vs Non-survived Passengers")
    plt.show()

    plt.figure()
    sns.countplot(data=titanic_data, x="Survived", hue="Sex").set_title("Survived vs Non-survived by Gender")
    plt.show()

    plt.figure()
    sns.countplot(data=titanic_data, x="Survived", hue="Pclass").set_title("Survived vs Non-survived by Class")
    plt.show()

    plt.figure()
    titanic_data["Age"].plot.hist().set_title("Distribution of Age")
    plt.show()

    plt.figure()
    titanic_data["Fare"].plot.hist().set_title("Distribution of Fare")
    plt.show()

    # Step 3: Data cleaning
    titanic_data.drop(columns=["zero"], axis=1, inplace=True)

    # Handle categorical variables
    titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Pclass"], drop_first=True)

    titanic_data.drop(columns=["sibsp", "Parch", "Embarked"], axis=1, inplace=True)

    # Ensure all column names are strings
    titanic_data.columns = titanic_data.columns.astype(str)

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
    print("Classification report of Logistic Regression:")
    print(classification_report(y_test, prediction))
    
    print("Confusion matrix of Logistic Regression:")
    print(confusion_matrix(y_test, prediction))

    print("Accuracy of Logistic Regression:")
    print(accuracy_score(y_test, prediction))

def main():
    print("Supervised machine learning: Logistic Regression on Titanic dataset")
    MarvellousTitanicLogistic()

if __name__ == "__main__":
    main()
