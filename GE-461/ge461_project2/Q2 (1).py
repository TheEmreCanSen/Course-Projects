import numpy as np

labels = np.loadtxt("labels.txt")
digits = np.loadtxt("digits.txt")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

dims = range(1, 201, 10)
train_accs = []
test_accs = []

for dim in dims:
    iso = Isomap(n_components=dim)
    embedding = iso.fit_transform(digits)
    
    X_train, X_test, y_train, y_test = train_test_split(embedding, labels, test_size=0.5)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train.T)
    
    train_acc = 1 - accuracy_score(y_train.T, clf.predict(X_train))
    test_acc = 1 - accuracy_score(y_test.T, clf.predict(X_test))
    train_accs.append(train_acc)
    test_accs.append(test_acc)

plt.plot(dims, train_accs, label='Training error')
plt.plot(dims, test_accs, label='Test error')
plt.xlabel('Dimension')
plt.ylabel('Classification error')
plt.legend()
plt.show()

