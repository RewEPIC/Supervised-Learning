from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd

# 1. Prepartion: Financial risk compliance dataset (No missing values)
data = pd.read_csv("./datasets/big4_financial_risk_compliance.csv")
target = 'Total_Revenue_Impact'
categorical_cols = ['Firm_Name', 'Industry_Affected', 'AI_Used_for_Auditing']
# Check and clean the dataset (drop missing values and duplicates)
print(data.isnull().sum())  # Check for missing values
data = data.dropna()  # Drop rows with missing values
data = data.drop_duplicates()  # Drop duplicate rows

# 2. Select the last 10 rows as the test set
training_data = data.head(len(data) - 10)
test_data = data.tail(10)

X_train = training_data.drop(target, axis=1)
Y_train = training_data[target]
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)

X_test = test_data.drop(target, axis=1)
Y_test = test_data[target]
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(Y_test, y_pred)
    return mae, mse, rmse, mape
# 3. Model Evaluation
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
linear_prediction = linear_model.predict(X_test)
linear_mae, linear_mse, linear_rmse, linear_mape = evaluate_model(linear_model, X_train, X_test, Y_train, Y_test)
# Ridge Regression
iterations = [100,500,1000]
alphas = [0.1, 1.0, 10]
results = []
ridge_predictions = []
for alpha in alphas:
    for max_iter in iterations:
        ridge_model = Ridge(alpha=alpha, solver='sag', max_iter=max_iter)
        ridge_model.fit(X_train, Y_train)
        mae, mse, rmse, mape = evaluate_model(ridge_model, X_train, X_test, Y_train, Y_test)
        y_pred = ridge_model.predict(X_test)
        results.append((alpha, max_iter, mae, mse, rmse, mape))
        ridge_predictions.append((alpha, max_iter, y_pred))

def perfomance_metrics():
    print("Linear Regression Metrics:")
    print(f"MAE: {linear_mae:.2f}")
    print(f"MSE: {linear_mse:.2f}")
    print(f"RMSE: {linear_rmse:.2f}")
    print(f"MAPE: {linear_mape:.2f}%")
    print("\nRidge Regression Metrics:")
    for alpha, max_iter, mae, mse, rmse, mape in results:
        print(f"\nAlpha: {alpha}, Max Iter: {max_iter}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
    print("\nTest Data Head:")
    print(test_data.head())

def linear_regression():
    plt.figure()
    plt.title("Linear Regression: Actual vs. Predicted")
    plt.scatter(Y_test, linear_model.predict(X_test), color='blue')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

def ridge_regression():
    for alpha, max_iter, ridge_pred in ridge_predictions:
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, ridge_pred, color='blue', alpha=0.5)
        plt.title(f"Ridge Regression: Actual vs. Predicted (alpha={alpha}, max_iter={max_iter})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.show()

def line_plot():
    plt.figure(figsize=(12, 6))
    plt.scatter(Y_test.index, Y_test.values, color='black', marker='o', s=50, label='Actual Points')  # Actual points
    plt.plot(Y_test.index, linear_prediction, label='Linear Regression', color='blue')
    for alpha in alphas:
        # Find the prediction with max_iter=500 for this alpha
        pred_for_alpha = next((pred for a, m, pred in ridge_predictions if a == alpha and m == 500), None)
        if pred_for_alpha is not None:
            plt.plot(Y_test.index, pred_for_alpha, label=f'Ridge (alpha={alpha}, iter=500)', linestyle='--')
    
    plt.title("Comparison of Actual, Linear, and Ridge Predictions Across Alpha Values")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend()
    plt.show()
line_plot()
# 4. Visualize the results
# plt.scatter(Y_test, linear_model.predict(X_test))
# plt.title("Actual vs. Predicted (Linear Regression)")
# plt.ylabel("Predicted")
# plt.xlabel("Actual")
# plt.show()

# Select a ridge model for visualization (e.g., alpha=1.0, iter=500)
# selected_ridge = Ridge(alpha=1.0, solver='sag', max_iter=500)
# selected_ridge.fit(X_train, Y_train)
# plt.scatter(Y_test, selected_ridge.predict(X_test))
# plt.title("Actual vs. Predicted (Ridge, alpha=1.0, iter=500)")
# plt.ylabel("Predicted")
# plt.xlabel("Actual")
# plt.show()

# plt.plot(Y_test.values, label='Actual')
# plt.plot(linear_model.predict(X_test), label='Linear Regression')
# plt.plot(selected_ridge.predict(X_test), label='Ridge (alpha=1.0, iter=500)')
# plt.legend()
# plt.title("Comparison of Predictions")
# plt.show()