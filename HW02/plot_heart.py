import matplotlib.pyplot as plt

# Depth values
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Corresponding accuracies for each depth
accuracies = [0.5534, 0.6505, 0.7476, 0.7476, 0.7476, 0.7573, 0.7282, 0.7282, 0.6990, 0.6990]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(depths, accuracies, marker='o', linestyle='-', color='b')

# Add labels and title
plt.title('Decision Tree Accuracy vs Depth')
plt.xlabel('Depth')
plt.ylabel('Accuracy')

# Display grid
plt.grid(True)

# Show the plot

plt.savefig("./my_plot_heart.png", format='png')
plt.show()
