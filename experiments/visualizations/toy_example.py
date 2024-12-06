import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors
from sklearn.neural_network  import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = sklearn.datasets.make_moons(n_samples=1000, noise=0.27)

data_X, data_y = data

classifiers = []
for i in range(5, 20, 2):
    classifier = MLPClassifier(hidden_layer_sizes=(i, i), max_iter=1000)
    classifier.fit(data_X , data[1])
    classifiers.append(classifier)  


original = [0.0, 0.0]
base = [0.5, 0.35]
counterfactuals = [[0.45, 0.5], [0.5, 0.6], [0.59, 0.65], [0.69, 0.67]]#, [0.3, 0.7], [0.45, 0.7], [0.5, 0.8], [0.72, 0.84]]
deltas = [0.6, 0.7, 0.8, 0.9]

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

palette = sns.color_palette("rocket_r", len(counterfactuals))

for i, cf in enumerate(counterfactuals):
    plt.scatter(cf[0], cf[1], label=f"delta: {deltas[i]}", marker='o', s=100, color=palette[i], alpha=1)
    
plt.scatter(original[0], original[1], label='original x', color='green', marker='^', s=100, alpha=1)
plt.scatter(base[0], base[1], label='base CFE', color='green', marker='s', s=100, alpha=1)

plt.plot([original[0], base[0]], [original[1], base[1]], color='green', linestyle='--', alpha=0.6)

for i, cf in enumerate(counterfactuals):
    plt.plot([cf[0], base[0]], [cf[1], base[1]], color=palette[i], linestyle='--', alpha=0.6)
    

# Plot the decision boundary
x_min, x_max = data[0][:, 0].min() - 0.1, data[0][:, 0].max() + 0.1
y_min, y_max = data[0][:, 1].min() - 0.1, data[0][:, 1].max() + 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[::, 1]
Z = classifiers[0].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

for i in range(1, len(classifiers)):
    Z += classifiers[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
Z = Z / len(classifiers)

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')

# plt.xlabel('x1')
# plt.ylabel('x2')

plt.xlim(-0.5, 1.8)
plt.ylim(-0.5, 1.8)

plt.legend(ncols=3, fontsize=8, bbox_to_anchor=(0.85, 0.94), labelspacing=1)

# Turn off the grid, ticks, and lines
plt.grid(False)
plt.tick_params(left=False, bottom=False)
plt.xticks([])
plt.yticks([])

# remove the frame
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.savefig('images/paper/thumbnail.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

