import threading
import time
import numpy as np

running = True

def fc():
    while running:
        print("Loading...")
        time.sleep(0.5)

thrd = threading.Thread(target=fc)

thrd.start()

time.sleep(5)

running = False

thrd.join()

print("Done")

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# vytvoření náhodných dat pro příklad
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
                            n_clusters_per_class=1, random_state=42)

# inicializace KNN modelu a fitování dat
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# vytvoření mřížky bodů pro vizualizaci
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# výsledky vizualizace
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# vizualizace trénovacích dat
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('KNN Classification')
plt.show()
