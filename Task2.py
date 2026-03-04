import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset from CSV file
df = pd.read_csv(r"D:\mall.csv")

# Display first few rows
print(df.head())

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Print clustered data
print(df)

# Plot clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
