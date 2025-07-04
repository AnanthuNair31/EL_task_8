# EL_task_8

## Overview
This task applies K-Means clustering to the Mall Customers dataset to segment customers based on their age, annual income, and spending score

## 1. Data Loading & Exploration
Reads the dataset and prints the first few rows, info, and checks for missing values.
Ensures the data is clean and ready for analysis.

## 2. PCA for Visualization
Drops  columns (CustomerID, Gender). - CustomerID:It's just a unique identifier for each customer, like a serial number.
                                     - Gender: in K-Means, which is distance-based, treating Gender as a number may mislead the algorithm
Applies PCA to reduce features to 2D for visualization.
Plots a scatter plot to visualize the distribution of customers.

## 3. K-Means Clustering
Selects Age, Annual Income (k$), and Spending Score (1-100) as features.
Applies K-Means clustering (default K=5).
Assigns each customer a cluster label and prints sample assignments and cluster sizes.

## 4. Elbow Method
Runs K-Means for K=1 to 10.
Plots inertia (within-cluster sum of squares) vs. K to help decide the optimal number of clusters.

## 5. Cluster Visualization
Plots the PCA-reduced data, color-coded by cluster assignment.

## 6. Clustering Evaluation
Calculates and prints the Silhouette Score to assess cluster quality.

## Output
PCA 2D visualization: See the distribution of customers.
Elbow plot: Helps you select the best K.
Clustered scatter plot: Visualizes how customers are grouped.
Silhouette Score: Quantitative measure of clustering quality. 
   -Silhouette Score for K=5: 0.358
