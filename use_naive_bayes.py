import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pyfiglet

def train_nb_model(features, labels):
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

    return nb 

def predict_with_nb(model, user_input):
    return model.predict(user_input)
