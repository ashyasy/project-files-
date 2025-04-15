import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the collected data
data = pd.read_csv('poker_data.csv')

# Display basic statistics
print(data.describe())

# Visualize action distribution
sns.countplot(x='action', data=data)
plt.title('Action Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
