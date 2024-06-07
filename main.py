# main.py

import pandas as pd
import numpy as np
import pyfiglet
from use_id3 import train_id3_model, predict_with_id3
from use_naive_bayes import train_nb_model, predict_with_nb
from use_knn import train_knn_model, predict_with_knn

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


df = pd.read_csv('diabetes.csv')
features = df.drop('Outcome', axis=1).values
labels = df['Outcome'].values

id3 = train_id3_model(features, labels)
nb = train_nb_model(features, labels)
knn = train_knn_model(features, labels)

print(pyfiglet.figlet_format("DIABETES PREDICTOR", font="small"))

while True:
    user_input = take_input()

    print("Using K-Nearest Neighbour:")
    user_prediction_knn = predict_with_knn(knn, user_input)
    if user_prediction_knn[0] == 1:
        print("Diabetes Positive")
    else:
        print("Diabetes Negative")
    
    print("Using Naive Bayes:")
    user_prediction_nb = predict_with_nb(nb, user_input)
    if user_prediction_nb[0] == 1:
        print("Diabetes Positive")
    else:
        print("Diabetes Negative")

    print("Using decesion tree(ID3):")
    user_prediction_id3 = predict_with_id3(id3, user_input)
    if user_prediction_id3[0] == 1:
        print("Diabetes Positive")
    else:
        print("Diabetes Negative")

    cont = input("Do you want to enter another set of values? (yes/no): ").strip().lower()
    if cont != 'yes':
        break
