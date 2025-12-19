import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 500

cpu = np.random.normal(40, 5, n_samples)
memory = np.random.normal(60, 8, n_samples)
disk = np.random.normal(100, 15, n_samples)
network = np.random.normal(200, 30, n_samples)

anomaly_indices = np.random.choice(n_samples, 25, replace=False)

cpu[anomaly_indices] += np.random.normal(40, 10, 25)
memory[anomaly_indices] += np.random.normal(30, 8, 25)
disk[anomaly_indices] += np.random.normal(80, 20, 25)
network[anomaly_indices] += np.random.normal(150, 30, 25)

df = pd.DataFrame({
    "CPU_Usage": cpu,
    "Memory_Usage": memory,
    "Disk_IO": disk,
    "Network_Traffic": network
})

df.to_csv("data/metrics.csv", index=False)

print("Dataset generated successfully!")
