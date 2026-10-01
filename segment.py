import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/customers.csv")
features = ["Age", "AnnualIncome", "SpendingScore"]

X = StandardScaler().fit_transform(df[features])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

print(df.to_string(index=False))
print("\nCluster counts:")
print(df["Cluster"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
plt.scatter(df["AnnualIncome"], df["SpendingScore"], c=df["Cluster"])
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig("customer_segments.png", dpi=150)
plt.show()

df.to_csv("segmented_customers.csv", index=False)
