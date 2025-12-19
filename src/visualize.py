import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/metrics.csv")

# Select features
X = df[["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_Traffic"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load trained model
model = joblib.load("model.pkl")

# Predict anomalies
predictions = model.predict(X_scaled)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[predictions == 1, 0],
    X_pca[predictions == 1, 1],
    c="blue",
    label="Normal",
    alpha=0.6
)

plt.scatter(
    X_pca[predictions == -1, 0],
    X_pca[predictions == -1, 1],
    c="red",
    label="Anomaly",
    alpha=0.8
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Anomaly Detection Visualization (Isolation Forest)")
plt.legend()
plt.show()

