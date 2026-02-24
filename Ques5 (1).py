import pandas as pd
import numpy as np

customers = pd.read_csv("Customers.csv")
sales = pd.read_excel("Sales.xlsx")
support = pd.read_csv("Support.csv")

print("Shapes:", customers.shape, sales.shape, support.shape)
print("Missing values:\n",
      customers.isnull().sum(), "\n",
      sales.isnull().sum(), "\n",
      support.isnull().sum())


sales["Revenue"] = sales["Quantity"] * sales["Price"]
sales["DiscountedPrice"] = sales["Price"].to_numpy() * 0.9  # 10% discount (broadcasting)


customers["Signup Date"] = pd.to_datetime(customers["Signup Date"], errors="coerce")
sales["Order Date"] = pd.to_datetime(sales["Order Date"], errors="coerce")

jan_orders_2025 = sales[
    (sales["Order Date"].dt.year == 2025) &
    (sales["Order Date"].dt.month == 1)
]
sales_first10 = sales.iloc[:10]

north_customers = customers[customers["Region"] == "North"]
high_value_orders = sales[sales["Revenue"] > 10000]

customers_sorted = customers.sort_values(by="Signup Date")
sales_sorted = sales.sort_values(by="Revenue", ascending=False)


sales_with_region = sales.merge(customers[["Customer ID", "Region"]], on="Customer ID", how="left")
avg_revenue_by_region = sales_with_region.groupby("Region")["Revenue"].mean().reset_index()

avg_resolution_by_issue = support.groupby("Issue Type")["Resolution Time"].mean().reset_index()


customers["Age"] = customers["Age"].fillna(customers["Age"].median())


customers = customers.rename(columns={"Customer ID": "CustomerID"})
sales = sales.rename(columns={"Customer ID": "CustomerID"})
support = support.rename(columns={"Customer ID": "CustomerID"})


merged = customers.merge(sales, on="CustomerID", how="left") \
                  .merge(support, on="CustomerID", how="left")


clv = merged.groupby("CustomerID")["Revenue"].sum().rename("CLV").reset_index()
merged = merged.merge(clv, on="CustomerID", how="left")


avg_res_time = merged.groupby("CustomerID")["Resolution Time"].mean().rename("AvgResolutionTime").reset_index()
merged = merged.merge(avg_res_time, on="CustomerID", how="left")

merged.to_csv("Cleaned_Data.csv", index=False)

print("✅ Pipeline complete. Cleaned_Data.csv generated.")
