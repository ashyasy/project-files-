
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = "output.csv"
data = pd.read_csv(file_path)

# Ensure data is sorted by n and K
data = data.sort_values(by=["n", "K"])

# Group data by n and find the optimal K for each n
optimal_k_values = []
for n, group in data.groupby("n"):
    # Find the K corresponding to the minimum average sorting time
    optimal_k = group.loc[group["AverageTime"].idxmin()]["K"]
    optimal_k_values.append((n, optimal_k))

# Convert to DataFrame for easier plotting
optimal_k_df = pd.DataFrame(optimal_k_values, columns=["n", "OptimalK"])

# Plot Optimal K vs. n
plt.figure(figsize=(10, 6))
plt.plot(optimal_k_df["n"], optimal_k_df["OptimalK"], marker='o', linestyle='-', label="Optimal K")
plt.xlabel("Array Length (n)")
plt.ylabel("Optimal Hybrid Sort Threshold (K)")
plt.title("Optimal K vs. Array Length (n)")
plt.grid(True)
plt.legend()
plt.savefig("optimal_k_vs_n.png", dpi=300)
plt.show()
