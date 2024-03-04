from collections import Counter  
from sklearn.datasets import fetch_openml  
from sklearn.preprocessing import scale  
from imblearn.under_sampling import CondensedNearestNeighbour
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn

def generate_random_data(n, d, num_classes):
    X = 100 * np.random.rand(n, d)
    y = np.ndarray((n, 1))
    points_per_class = n // num_classes
    for i in range(num_classes):
        y[i * points_per_class : (i+1) * points_per_class] = i
        # y[i] = np.random.choice(list(range(num_classes)))
    # X[n-1] = X[n-1] * 0 + 1
    # y[n-1] = num_classes
    X, y = sklearn.utils.shuffle(X, y)
    return X, y.squeeze()

def find_nearest_prototype(x, prototypes):
    """
    Find the nearest prototype to the element x from the set of prototypes.
    """
    distances = [np.linalg.norm(x - prototype) for prototype in prototypes]
    if not distances:
        return None
    return np.argmin(distances)

def iterative_prototype_scan(X, labels):
    """
    Iteratively scan elements in X and update the set U of prototypes.
    """
    U = []  # Set of prototypes
    U_labels = []  # Labels corresponding to prototypes
    
    # Initial scan to find the nearest prototype for each element in X

    while True:
        # Find the first element x whose nearest prototype has a different label
        idx_to_remove = None
        for i in range(X.shape[0]):
            nearest_idx = find_nearest_prototype(X[i], U)
            if len(U) == 0 or labels[nearest_idx] != labels[i]:    
                # Add x to U and remove it from X
                U.append(X[i])
                U_labels.append(labels[i])
                idx_to_remove = i
                break

        if idx_to_remove is not None:
            # Remove the element from X
            X = np.delete(X, idx_to_remove, axis=0)
            labels = np.delete(labels, idx_to_remove, axis=0)

        else:
            # No more prototypes added, break the loop
            break

    return np.array(U), np.array(U_labels)


X, y = generate_random_data(200, 200000, 2)

print(X)
print('Original dataset shape %s' % Counter(y))  
cnn = CondensedNearestNeighbour(sampling_strategy='all')  
X_res, y_res = cnn.fit_resample(X, y)  
# cnn = CondensedNearestNeighbour(sampling_strategy='auto')  
# X_res, y_res = cnn.fit_resample(X_res, y_res) 
print('Resampled dataset shape %s' % Counter(y_res))
print(f"Original training set size: {y.shape[0]}. Compressed set size: {y_res.shape[0]}")
# X_res, y_res = iterative_prototype_scan(X, y)
# print('Resampled dataset shape: %s' % Counter(y_res))
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_res, y_res)
print(f"Correctly predicted: {neigh.score(X, y) * X.shape[0]}; Parameters: {y_res.shape[0]}")
print(f"Bits per parameter: {neigh.score(X, y) * X.shape[0] / y_res.shape[0]}")
# print(f"Original training set size: {y.shape[0]}. Compressed set size: {y_res.shape[0]}")
