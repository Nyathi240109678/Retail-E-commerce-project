
# RETAIL SALES , we are using(MULTIPLE LINEAR REGRESSION)
# 60% TRAINING / 40% TESTING


# 1. IMPORT LIBRARIES


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



# 2. LOAD DATASET


# Get the folder where this Python program is saved
folder = os.path.dirname(os.path.abspath(__file__))

# Loading the CSV
df = pd.read_csv("Retail_sales_dataset.csv")

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

print("File found: Retail_sales_dataset.csv")
print("Location:", os.path.join(folder, "Retail_sales_dataset.csv"))

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# 3. BASIC INFORMATION


print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())



# 4. CONVERT DATE COLUMN


df["transaction_date"] = pd.to_datetime(
    df["transaction_date"],
    errors="coerce"
)

print("\nTransaction date converted to:")
print(df["transaction_date"].dtype)



# 5. CREATE DATE FEATURES
# Converting the date into numerical features so that the Linear Regression model,
# can use the information from the transaction date.


df["year"] = df["transaction_date"].dt.year
df["month"] = df["transaction_date"].dt.month
df["day"] = df["transaction_date"].dt.day
df["day_of_week"] = df["transaction_date"].dt.dayofweek


print("\nDate features created:")
print("year")
print("month")
print("day")
print("day_of_week")



# 6. SELECT FEATURES
# we wont use gross_sales and discount_amount because they are calculated from sales_amount-related data.

numeric_features = [
    "quantity",
    "unit_price",
    "discount_pct",
    "year",
    "month",
    "day",
    "day_of_week"
]

categorical_features = [
    "sales_channel",
    "payment_method",
    "region"
]



# 7. CREATE X AND y

X = df[numeric_features + categorical_features]

y = df["sales_amount"]


print("\nFeatures before encoding:")
print(X.columns.tolist())

print("\nTarget:")
print("sales_amount")



# 8. ONE-HOT ENCODING
# We convert categorical text values into numerical 0 and 1 values.


X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)


print("\nFeatures after One-Hot Encoding:")
print(X.columns.tolist())

print("\nNumber of features:", len(X.columns))



# 9. CHECK FOR MISSING VALUES
# we check for missing values on the dataset

print("\nMissing values in features:")
print(X.isnull().sum().sum())

print("Missing values in target:")
print(y.isnull().sum())



# 10. REMOVE ROWS WITH MISSING VALUES
# We remove rows with missing values as a safety step after date conversion.


valid_rows = X.notnull().all(axis=1) & y.notnull()

X = X[valid_rows]
y = y[valid_rows]



# 11. SPLIT DATA
# We split the data into 60% training data and 40% testing data.
# The training data is used to train the model, while the testing data is used to evaluate it.


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42
)


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Total records:", len(X))

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training percentage:",
    round(len(X_train) / len(X) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2),
    "%"
)



# 12. CREATE LINEAR REGRESSION MODEL
# We create the Linear Regression model that will learn from the training data.

model = LinearRegression()



# 13. TRAIN MODEL
# we are training the model

print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training complete.")



# 14. MAKE PREDICTIONS
#we are testing the model

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("Predictions complete.")



# 15. EVALUATE MODEL
# We evaluate the model using R2 Score, MAE, and RMSE.

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("R2 Score :", round(r2, 6))
print("MAE      :", round(mae, 6))
print("RMSE     :", round(rmse, 6))



# 16. SHOW ACTUAL VS PREDICTED VALUES
# We print the first 10 predictions and actual value to compare and see if iyasebenza

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10))



# 17. ACTUAL VS PREDICTED GRAPH
# We compare the actual and predicted sales on the graph.

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3,
    color="blue"
)

# Perfect prediction line

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.title("Actual vs Predicted Sales Amount")

plt.xlabel("Actual Sales Amount")

plt.ylabel("Predicted Sales Amount")

plt.grid(True)

plt.show()



# 18. MODEL COEFFICIENTS
#We show how each feature affects the predicted sales amount.

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients["Absolute Value"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute Value",
    ascending=False
)


print("\n" + "=" * 60)
print("MODEL COEFFICIENTS")
print("=" * 60)

print(coefficients.to_string(index=False))



# 19. INTERCEPT


print("\nModel Intercept:")
print(model.intercept_)



# 20. FINAL SUMMARY
# We summarize the model, data split, features, target, and final results.


print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Model: Multiple Linear Regression")
print("Training data: 60%")
print("Testing data: 40%")

print("\nFeatures used:")

for feature in numeric_features:
    print("-", feature)

for feature in categorical_features:
    print("-", feature)

print("\nTarget: sales_amount")

print("\nFinal results:")
print("R2 Score :", round(r2, 6))
print("MAE      :", round(mae, 6))
print("RMSE     :", round(rmse, 6))

print("\nProgram finished successfully.")




# 21. SAVE TRAINED MODEL


model_path = os.path.join(folder, "sales_model.pkl")
features_path = os.path.join(folder, "model_features.pkl")

joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), features_path)

print("\n" + "=" * 60)
print("MODEL FILES SAVED SUCCESSFULLY")
print("=" * 60)

print("Model saved to:")
print(model_path)

print("\nFeatures saved to:")
print(features_path)

print("\nProgram finished successfully.")
