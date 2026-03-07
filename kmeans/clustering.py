from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import joblib

def run_elbow_method(X_train_scaled, k_range=range(1, 11)):
    """Chạy vòng lặp K để tính WCSS."""
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_train_scaled)
        wcss.append(kmeans.inertia_)
    return wcss

def train_kmeans(X_train_scaled, k=6):
    """Huấn luyện mô hình K-Means trên tập Train."""
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    model.fit(X_train_scaled)
    labels = model.labels_
    centers = model.cluster_centers_
    joblib.dump(model, 'kmeans_aqi_model.pkl')
    return model, labels,centers
