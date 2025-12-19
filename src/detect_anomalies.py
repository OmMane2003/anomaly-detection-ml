import joblib
from preprocess import preprocess_data

# Load & preprocess data
df, X_scaled = preprocess_data()

# Load trained model
model = joblib.load("model.pkl")

# Predict anomalies
prediction = model.predict(X_scaled)

# Label results
df["Anomaly"] = ["Anomaly" if x == -1 else "Normal" for x in prediction]

# Show anomalies
anomalies = df[df["Anomaly"] == "Anomaly"]

print("Detected anomalies:")
print(anomalies.head(10))
print(f"\nTotal anomalies detected: {len(anomalies)}")
