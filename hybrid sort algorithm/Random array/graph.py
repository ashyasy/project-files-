import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "C:\\Users\\ashso\\source\\repos\\algosc++\\algosc++\\output.csv"c  # this needs to be chaned to your dirctory

data = pd.read_csv(file_path)

# Sort unique `n` values
unique_n_values = sorted(data["n"].unique().astype(int))

# Set up the main figure for average sorting time
plt.figure(figsize=(10, 6))

# Plot average time for each `n`
for n in unique_n_values:
    subset = data[data["n"] == n]
    plt.plot(subset["K"], subset["AverageTime"], label=f"n = {n}")

# Add labels, title, and legend
plt.xlabel("Hybrid Sort Threshold (K)")
plt.ylabel("Average Sorting Time (seconds)")
plt.title("Hybrid Sort Performance: Average Sorting Time vs Threshold (K)")
plt.legend()
plt.grid(True)

# Set limits to zoom in
plt.ylim(0, 0.01)  # Adjust to the range you want for y-axis
plt.xlim(0, 100)   # Adjust to focus on specific `K` range if needed

# Save and show the plot
plt.savefig("hybrid_sort_zoomed_in.png", dpi=300)
plt.show()




file_path = "C:\\Users\\ashso\\source\\repos\\algosc++\\algosc++\\output.csv"