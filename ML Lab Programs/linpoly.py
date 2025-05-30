import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def plot_results(X_test, y_test, y_pred, xlabel, ylabel, title):
    plt.scatter(X_test, y_test, label="Actual", alpha=0.7)
    plt.scatter(X_test, y_pred, label="Predicted", alpha=0.7, color="red")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.show()

def evaluate_model(y_test, y_pred, title):
    print(f"\n{title}")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

def linear_regression_california():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data[["AveRooms"]], data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_results(X_test, y_test, y_pred, "AveRooms", "Median Home Value", "Linear Regression - California Housing")
    evaluate_model(y_test, y_pred, "California Housing Dataset")

def polynomial_regression_auto_mpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg", "cyl", "disp", "hp", "weight", "acc", "year", "origin"]
    df = pd.read_csv(url, sep='\s+', names=cols, na_values="?").dropna()
    X, y = df[["disp"]], df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_results(X_test, y_test, y_pred, "Displacement", "MPG", "Polynomial Regression - Auto MPG")
    evaluate_model(y_test, y_pred, "Auto MPG Dataset")

if __name__ == "__main__":
    print("Running Linear and Polynomial Regression Demos")
    linear_regression_california()
    polynomial_regression_auto_mpg()
