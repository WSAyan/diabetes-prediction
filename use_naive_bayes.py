import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pyfiglet

print("loading diabetes data set....")

df = pd.read_csv('diabetes.csv')

print("creating naive bayes model....")

features = df.drop('Outcome', axis=1).values
labels = df['Outcome'].values

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.25, random_state=42)

nb = GaussianNB()

nb.fit(feature_train, label_train)

print("naive bayes model created!")

print("running test...")

label_pred = nb.predict(feature_test)
cm = confusion_matrix(label_test, label_pred)
print((f"Confusion matix:\n  {cm}"))
accuracy = accuracy_score(label_test, label_pred)
print(f"Test accuracy: {(accuracy * 100):.2f}%")

print(pyfiglet.figlet_format("DIABETES PREDICTOR", font="small"))

def take_input():
    print("Enter the following values:")
    Pregnancies = int(input("Pregnancies: "))
    Glucose = float(input("Glucose: "))
    BloodPressure = float(input("BloodPressure: "))
    SkinThickness = float(input("SkinThickness: "))
    Insulin = float(input("Insulin: "))
    BMI = float(input("BMI: "))
    DiabetesPedigreeFunction = float(input("DiabetesPedigreeFunction: "))
    Age = int(input("Age: "))
    return np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)

user_input = take_input()

user_prediction = nb.predict(user_input)

if user_prediction[0] == 1:
    print("Diabetes Positive")
else:
    print("Diabetes Negative")
