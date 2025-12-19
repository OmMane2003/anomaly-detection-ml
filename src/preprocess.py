import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path="data/metrics.csv"):
    """
    Loads data, selects features, and applies scaling.
    Returns scaled features and original dataframe.
    """
    df = pd.read_csv(csv_path)

    features = ["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_Traffic"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled
