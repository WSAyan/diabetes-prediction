import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def train_id3_model(features, labels):
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)

    id3.fit(feature_train, label_train)

    print("ID3 Decision Tree model created!")

    print("running test...")

    label_pred = id3.predict(feature_test)
    cm = confusion_matrix(label_test, label_pred)
    print(f"Confusion matrix:\n  {cm}")
    accuracy = accuracy_score(label_test, label_pred)
    print(f"Test accuracy: {(accuracy * 100):.2f}%")

    return id3

def predict_with_id3(model, user_input):
    return model.predict(user_input)