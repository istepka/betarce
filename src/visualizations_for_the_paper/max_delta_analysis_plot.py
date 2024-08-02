import scipy.stats as stats
import pandas as pd

confidence = 0.95
a = 1
b = 1

print(stats.beta.ppf(0.025, a, b))

import matplotlib.pyplot as plt
import numpy as np

k = [1, 2] + [i for i in range(4, 128, 4)]
conf = [0.7,  0.8,  0.9, 0.95, 0.975, 0.99, 0.9999]

maxdeltas = []

for c in conf:
    deltas = []
    for i in k:
        lb = stats.beta.ppf(1 - c, a + i, b)
        deltas.append(lb)
    maxdeltas.append(deltas)
    
    
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.8, len(conf)))
for i, c in enumerate(conf):
    ax.plot(k, maxdeltas[i], label=f'Confidence: {c}', color=colors[i])
    
ax.set_xlabel('Number of estimators k')
ax.set_ylabel('Max achievable delta')
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])

# plt.grid(axis='y')
plt.grid()
plt.title('Max achievable delta-Robustness for different alpha-confidence levels')
plt.legend()
plt.tight_layout()

plt.savefig('images/betarob/max_delta_analysis.png', dpi=300, bbox_inches='tight')

plt.show()



k = [1, 2] + [i for i in range(4, 128, 8)]
conf = [0.7,  0.8,  0.9, 0.95, 0.975, 0.99]

maxdeltas = []

for c in conf:
    deltas = []
    for i in k:
        lb, rb = stats.beta.interval(c, a + i, b)
        deltas.append(lb)
    maxdeltas.append(deltas)

# Create also latex table for the max deltas for the different confidence levels
df = pd.DataFrame(maxdeltas, index=conf, columns=k)
# Index name
df.index.name = 'Confidence'
# Create latex table
df.T.to_latex('images/betarob/max_delta_analysis.tex', float_format="%.3f", label='tab:maxdelta', caption='Max achievable delta-Robustness for different alpha-confidence levels')

print(df)