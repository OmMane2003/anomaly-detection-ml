import joblib
from sklearn.ensemble import IsolationForest
from preprocess import preprocess_data

# Load & preprocess data
df, X_scaled = preprocess_data()

# Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
