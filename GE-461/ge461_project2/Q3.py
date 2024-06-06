from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

labels = np.loadtxt("labels.txt")
features = np.loadtxt("digits.txt")

tsne = TSNE(n_components=2, perplexity=40, n_iter=4000, random_state=310)
X1_tsne = tsne.fit_transform(features)

plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', 10))
plt.title('t-SNE Mapping for 4000 Iterations, Perplexity = 40')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()