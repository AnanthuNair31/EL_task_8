import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


df = pd.read_csv('Mall_Customers.csv')
df_head = df.head()
print(df_head , df.info() )
print(df.isnull().sum())


features = df.drop(columns = ['CustomerID' , 'Gender'])
pca = PCA(n_components = 2)
pca_output = pca.fit_transform(features)
print(pca_output)

plt.figure(figsize = (8,8))
plt.scatter(pca_output[:,0], pca_output[:,1], c = 'red', edgecolor = 'skyblue')
plt.title('PCA 2D visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

x = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters =5, random_state = 42)
df['cluster'] = kmeans.fit_predict(x)
print(df[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'cluster']].head())
print(df['cluster'].value_counts())

inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(x)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize = (8,8))
plt.plot(k_range, inertia_values , marker = 'o', linestyle = '--', color = 'blue')
plt.title('Elbow Method for K')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

plt.figure(figsize = (8,8))
scatter = plt.scatter(pca_output[:,0], pca_output[:,1], c = df['cluster'], cmap = 'tab10', edgecolor = 'red')
plt.title('PCA  and k-Means Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.colorbar(scatter, label= 'cluster')
plt.show()

score = silhouette_score(x, df['cluster'])
print(f"Silhouette Score for K=5: {score:.3f}")