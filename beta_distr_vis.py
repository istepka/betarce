import scipy.stats as stats
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

beta_settings = [
    (0.5, 0.5),
    (1.0, 1.0),
    (10, 0.5),
    (0.5, 10),
]

# Generate x values
x = np.linspace(0, 1, 100)

# Plot beta distributions
fig, ax = plt.subplots(figsize=(7, 4))

lines = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'm']

for a, b in beta_settings:
    y = stats.beta.pdf(x, a, b) / len(x)
    ax.plot(x, y, label=f'a={a}, b={b}', linestyle=lines.pop(0), color=colors.pop(0))
    
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')
ax.set_title('Beta Distribution')
# plt.yticks([])

# plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig('images/betarob/beta_distr.png', dpi=300, bbox_inches='tight')

plt.show()

