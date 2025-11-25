import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('iris_results.csv')

# Ensure correct data types
data['Max Depth'] = data['Max Depth'].astype(int)
data['Selection Criteria'] = data['Selection Criteria'].astype(int)
data['Accuracy(%)'] = data['Accuracy(%)'].astype(float)
data['treeSize'] = data['treeSize'].astype(int)

# Group by Max Depth and Selection Criteria
grouped = data.groupby(['Max Depth', 'Selection Criteria'])['Accuracy(%)'].mean().unstack()

# Tree size data
tree_size_data = data[['Max Depth', 'treeSize']].drop_duplicates().sort_values('Max Depth')

# Criteria name mapping
criteria_names = {0: 'NG', 1: 'NGR', 2: 'NWIG'}
colors = {0: '#3B82F6', 1: '#EF4444', 2: '#10B981'}

# Plot accuracy vs max depth
plt.figure(figsize=(8, 6))
for criterion in grouped.columns:
    plt.plot(grouped.index, grouped[criterion], marker='o', color=colors[criterion],
             label=criteria_names[criterion], linewidth=2)

plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy (%)')
plt.title('Average Accuracy vs. Max Tree Depth for Different Criteria')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Selection Criteria')
plt.ylim(60, 100)
plt.savefig('accuracy_vs_depth.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot tree size vs max depth
plt.figure(figsize=(8, 6))
plt.plot(tree_size_data['Max Depth'], tree_size_data['treeSize'], marker='o', color='#3B82F6',
         label='Tree Size', linewidth=2)
plt.xlabel('Max Tree Depth')
plt.ylabel('Number of Nodes')
plt.title('Number of Nodes vs. Max Tree Depth')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('nodes_vs_depth.png', dpi=300, bbox_inches='tight')
plt.show()
