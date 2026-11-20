# Name : Saksham Kaushik
# Rref No: 2026B15410084
# Git Repo : https://github.com/sakshamkaushi/Saksham-kaushik-HCL-project

import pandas as pd
import numpy as np

# 1. Load and inspect dataset
df = pd.read_csv("08_food_delivery.csv")

print("FIRST FIVE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

# 2. Missing delivery times
print("\nMISSING DELIVERY TIMES")
print(df[df["Delivery_Time_Minutes"].isnull()])

# 3. Check duplicate Order_ID
print("\nDUPLICATE ORDERS")
print(df["Order_ID"].duplicated().sum())

# Remove duplicate orders
df = df.drop_duplicates(subset="Order_ID", keep="first").copy()

# 4. Total orders and status percentages
print("\nTOTAL ORDERS")
print(df["Order_ID"].nunique())

print("\nORDER STATUS PERCENTAGES")
print(df["Order_Status"].value_counts(normalize=True) * 100)

# 5. Average delivery time by city (completed orders)
delivered = df[df["Order_Status"] == "Delivered"].copy()

print("\nAVERAGE DELIVERY TIME BY CITY")
print(delivered.groupby("City")["Delivery_Time_Minutes"].mean())

# 6. Top five restaurants by total order value
print("\nTOP FIVE RESTAURANTS")
print(df.groupby("Restaurant")["Order_Value"]
      .sum().sort_values(ascending=False).head(5))

# 7. Average order value by restaurant
print("\nAVERAGE ORDER VALUE BY RESTAURANT")
print(df.groupby("Restaurant")["Order_Value"].mean()
      .sort_values(ascending=False))

# 8. Peak ordering hours
df["Order_Hour"] = pd.to_datetime(
    df["Order_Time"], format="%H:%M:%S"
).dt.hour

print("\nORDERS BY HOUR")
print(df["Order_Hour"].value_counts().sort_index())

print("\nPEAK ORDERING HOUR")
print(df["Order_Hour"].value_counts().idxmax())

# 9. Delivery time by order status
print("\nDELIVERY TIME BY STATUS")
print(df.groupby("Order_Status")["Delivery_Time_Minutes"].mean())

# 10. Cancellation and delay rates by city
df["Cancelled"] = df["Order_Status"] == "Cancelled"

df["Delayed"] = df["Delivery_Time_Minutes"] > 45

cancel_rate = df.groupby("City")["Cancelled"].mean() * 100

valid = df[df["Delivery_Time_Minutes"].notna()]

delay_rate = valid.groupby("City")["Delayed"].mean() * 100

print("\nCANCELLATION RATE BY CITY")
print(cancel_rate.round(2))

print("\nDELAY RATE BY CITY")
print(delay_rate.round(2))

# 11. NumPy median and standard deviation
times = df["Delivery_Time_Minutes"].dropna().to_numpy()

print("\nMEDIAN DELIVERY TIME")
print(np.median(times))

print("\nSTANDARD DEVIATION")
print(np.std(times))

# 12. IQR outlier detection
Q1 = np.percentile(times, 25)
Q3 = np.percentile(times, 75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Delivery_Time_Minutes"] < lower_limit) |
    (df["Delivery_Time_Minutes"] > upper_limit)
]

print("\nIQR OUTLIERS")
print(outliers[[
    "Order_ID", "City", "Restaurant", "Delivery_Time_Minutes"
]])

# 13. Delivery-time categories
conditions = [
    df["Delivery_Time_Minutes"] <= 30,
    df["Delivery_Time_Minutes"].between(30, 45, inclusive="right"),
    df["Delivery_Time_Minutes"] > 45
]

choices = ["Fast", "Standard", "Delayed"]

df["Delivery_Band"] = np.select(
    conditions,
    choices,
    default="Not Available"
)

print("\nDELIVERY BANDS")
print(df["Delivery_Band"].value_counts())

# 14. Revenue by city and payment mode
print("\nREVENUE BY CITY AND PAYMENT MODE")

revenue = pd.pivot_table(
    df,
    values="Order_Value",
    index="City",
    columns="Payment_Mode",
    aggfunc="sum",
    fill_value=0
)

print(revenue)

# 15. City performance table
print("\nCITY PERFORMANCE")

city_performance = df.groupby("City").agg(
    Total_Orders=("Order_ID", "nunique"),
    Total_Order_Value=("Order_Value", "sum"),
    Average_Order_Value=("Order_Value", "mean")
)

city_performance["Average_Delivery_Time"] = (
    delivered.groupby("City")["Delivery_Time_Minutes"].mean()
)

print(city_performance.round(2))

print("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY")

