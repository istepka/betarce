import scipy.stats as stats
confidence = 0.95
a = 48
b = 1

lb, rb = stats.beta.interval(confidence, a, b)

print(f'Parameters a={a}, b={b} with confidence {confidence} gives the interval: [{lb:.3f}, {rb:.3f}]')

print(stats.beta.ppf(0.025, a, b))

import matplotlib.pyplot as plt
import numpy as np

k = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
conf = [0.7,  0.8,  0.9, 0.95, 0.99]

maxdeltas = []

for c in conf:
    deltas = []
    for i in k:
        lb, rb = stats.beta.interval(c, 0.5 + i, 0.5)
        deltas.append(lb)
    maxdeltas.append(deltas)
    
    
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.8, len(conf)))
for i, c in enumerate(conf):
    ax.plot(k, maxdeltas[i], label=f'Confidence: {c}', color=colors[i])
    
ax.set_xlabel('Number of estimators k')
ax.set_ylabel('Max achievable delta')
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])

plt.grid(axis='y')
plt.title('Max achievable delta-Robustness for different alpha-confidence levels')
plt.legend()
plt.tight_layout()

plt.savefig('images/betarob/max_delta_analysis.png', dpi=300, bbox_inches='tight')

plt.show()