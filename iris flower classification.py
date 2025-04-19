# pandas ni import chestunnam - data ni table laga handle cheyyadaniki
import pandas as pd

# sklearn lo iris dataset ni load cheyyadam
from sklearn.datasets import load_iris

# data ni train and test ga divide cheyyadaniki
from sklearn.model_selection import train_test_split

# Random Forest Classifier ni import cheyyadam - ee algorithm ni training ki vadutam
from sklearn.ensemble import RandomForestClassifier

# model prediction accuracy calculate cheyyadaniki
from sklearn.metrics import accuracy_score


# Iris dataset ni load cheyyadam
iris = load_iris()

# Dataset lo features ni (X) dataframe ga convert cheyyadam (columns names kuda include chesam)
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Labels (target values) ni y lo store cheyyadam
y = pd.Series(iris.target)


# Data ni 80% training ki, 20% testing ki split cheyyadam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model ni create cheyyadam (RandomForestClassifier)
model = RandomForestClassifier()

# Training data mida model ni train cheyyadam
model.fit(X_train, y_train)


# Test data mida prediction cheyyadam
y_pred = model.predict(X_test)

# Actual values tho compare chesi accuracy calculate cheyyadam
accuracy = accuracy_score(y_test, y_pred)

# Model accuracy print cheyyadam
print("\nModel Accuracy:", accuracy)


# Features (sepal length, petal length etc.) yoka range display cheyyadam
print("\nIris Dataset - Feature Value Ranges (in cm):")
for feature in X.columns:
    print(f"{feature}: {X[feature].min()} to {X[feature].max()}")


# Class wise (setosa, versicolor, virginica) feature ranges print cheyyadam
print("\nApproximate Value Ranges by Class:")

print("\n1. Iris-setosa:")
print("   Sepal length: 4.3 – 5.8")
print("   Sepal width : 3.0 – 4.4")
print("   Petal length: 1.0 – 1.9")
print("   Petal width : 0.1 – 0.6")

print("\n2. Iris-versicolor:")
print("   Sepal length: 4.9 – 7.0")
print("   Sepal width : 2.0 – 3.4")
print("   Petal length: 3.0 – 5.1")
print("   Petal width : 1.0 – 1.8")

print("\n3. Iris-virginica:")
print("   Sepal length: 4.9 – 7.9")
print("   Sepal width : 2.2 – 3.8")
print("   Petal length: 4.5 – 6.9")
print("   Petal width : 1.4 – 2.5")


# User ni inputs adagatam (flower measurements)
print("\nEnter Iris flower measurements:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))


# Input values chala thakkuva leka ekkuva unna warning ivvadam
if any([
    sepal_length > 8 or sepal_length < 4,
    sepal_width > 5 or sepal_width < 2,
    petal_length > 7 or petal_length < 1,
    petal_width > 3 or petal_width < 0.1
]):
    print("\nInput values are too far from real data! Please enter realistic values.")


# Konni values normal range ki slight ga outside unte warnings ivvadam
if not (4.3 <= sepal_length <= 7.9):
    print("Warning: Sepal length is outside the normal range!")
if not (2.0 <= sepal_width <= 4.4):
    print("Warning: Sepal width is outside the normal range!")
if not (1.0 <= petal_length <= 6.9):
    print("Warning: Petal length is outside the normal range!")
if not (0.1 <= petal_width <= 2.5):
    print("Warning: Petal width is outside the normal range!")


# User ichina input mida prediction cheyyadam
sample = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(sample)

# Prediction result ni class name lo convert cheyyadam
predicted_class = iris.target_names[prediction][0]


# Final output: flower class + classifier name display cheyyadam
print("\nPredicted Iris flower class is:", predicted_class)
print("Classifier used:", type(model).__name__)  # e.g., RandomForestClassifier
