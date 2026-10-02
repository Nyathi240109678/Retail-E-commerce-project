
# RETAIL SALES - SALES PREDICTION TOOL
# Uses the trained Multiple Linear Regression model


import pandas as pd
import joblib
import os



# 1. LOAD SAVED MODEL


folder = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(folder, "sales_model.pkl")
features_path = os.path.join(folder, "model_features.pkl")


print("=" * 60)
print("RETAIL SALES PREDICTION TOOL")
print("=" * 60)


# Check that the saved files exist

if not os.path.exists(model_path):
    print("ERROR: sales_model.pkl was not found.")
    print("Make sure it is in the same folder as this program.")
    input("Press Enter to close...")
    exit()

if not os.path.exists(features_path):
    print("ERROR: model_features.pkl was not found.")
    print("Make sure it is in the same folder as this program.")
    input("Press Enter to close...")
    exit()


# Load model and feature names

model = joblib.load(model_path)
model_features = joblib.load(features_path)


print("\nSaved model loaded successfully.")



# 2. GET USER INPUT


print("\nEnter the information for the new transaction.")
print("-" * 60)


# Quantity

quantity = float(
    input("Quantity: ")
)


# Unit price

unit_price = float(
    input("Unit price: ")
)


# Discount percentage

discount_pct = float(
    input("Discount percentage (e.g. 10 for 10%): ")
)


# Transaction date

transaction_date = input(
    "Transaction date (YYYY-MM-DD): "
)


# Convert date

transaction_date = pd.to_datetime(
    transaction_date,
    errors="coerce"
)


if pd.isna(transaction_date):
    print("\nERROR: Invalid date.")
    input("Press Enter to close...")
    exit()



# 3. EXTRACT DATE FEATURES


year = transaction_date.year
month = transaction_date.month
day = transaction_date.day
day_of_week = transaction_date.dayofweek



# 4. CATEGORICAL INPUTS


print("\nSales Channel")
print("1. In-Store")
print("2. Online")
print("3. Mobile App")

channel_choice = input("Choose 1, 2, or 3: ")


if channel_choice == "1":
    sales_channel = "In-Store"
elif channel_choice == "2":
    sales_channel = "Online"
elif channel_choice == "3":
    sales_channel = "Mobile App"
else:
    print("ERROR: Invalid sales channel.")
    input("Press Enter to close...")
    exit()



# PAYMENT METHOD


print("\nPayment Method")
print("1. Cash")
print("2. Credit Card")
print("3. Debit Card")
print("4. Gift Card")
print("5. PayPal")

payment_choice = input("Choose 1, 2, 3, 4, or 5: ")


payment_methods = {
    "1": "Cash",
    "2": "Credit Card",
    "3": "Debit Card",
    "4": "Gift Card",
    "5": "PayPal"
}


if payment_choice not in payment_methods:
    print("ERROR: Invalid payment method.")
    input("Press Enter to close...")
    exit()


payment_method = payment_methods[payment_choice]



# REGION


print("\nRegion")
print("1. Central")
print("2. East")
print("3. North")
print("4. South")
print("5. West")

region_choice = input("Choose 1, 2, 3, 4, or 5: ")


regions = {
    "1": "Central",
    "2": "East",
    "3": "North",
    "4": "South",
    "5": "West"
}


if region_choice not in regions:
    print("ERROR: Invalid region.")
    input("Press Enter to close...")
    exit()


region = regions[region_choice]



# 5. CREATE INPUT DATAFRAME


new_data = pd.DataFrame({
    "quantity": [quantity],
    "unit_price": [unit_price],
    "discount_pct": [discount_pct],
    "year": [year],
    "month": [month],
    "day": [day],
    "day_of_week": [day_of_week],
    "sales_channel": [sales_channel],
    "payment_method": [payment_method],
    "region": [region]
})



# 6. ONE-HOT ENCODING


new_data = pd.get_dummies(
    new_data,
    columns=[
        "sales_channel",
        "payment_method",
        "region"
    ],
    drop_first=True,
    dtype=int
)



# 7. MATCH MODEL FEATURES
# We make sure the new data has the same columns as the trained model so it can predict correctly.

new_data = new_data.reindex(
    columns=model_features,
    fill_value=0
)



# 8. MAKE PREDICTION


prediction = model.predict(new_data)


predicted_sales = prediction[0]



# 9. DISPLAY RESULT


print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("Quantity:", quantity)
print("Unit price:", unit_price)
print("Discount:", discount_pct, "%")
print("Transaction date:", transaction_date.date())
print("Sales channel:", sales_channel)
print("Payment method:", payment_method)
print("Region:", region)

print("-" * 60)

print(
    "PREDICTED SALES AMOUNT: R",
    round(predicted_sales, 2)
)

print("=" * 60)



# 10. FINISH
#Press enter to close.

input("\nPress Enter to close...")
