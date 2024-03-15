from collections import Counter  
from sklearn.datasets import fetch_openml  
from sklearn.preprocessing import scale  
from imblearn.under_sampling import CondensedNearestNeighbour
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=FutureWarning)
# np.random.seed(7272)

def generate_random_data(n, d, num_classes):
    X = 100 * np.random.rand(n, d)
    y = np.ndarray((n, 1), dtype=int)
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

def original_condense(X, labels):
    """
    Implementation of the following:
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7c3771fd6829630cf450af853df728ecd8da4ab2
    """
    labels = labels.reshape(-1, 1)
    store = []
    grabbag = []
    store.append(0)
    for i in range(1, X.shape[0]):
        X_store = points_from_indices(X, store)
        X_i = points_from_indices(X, i)
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X_store, labels[store, :])
        if neigh.score(X_i, labels[i, :]) > 0:
            grabbag.append(i)
        else:
            store.append(i)
    while True:
        original_length = len(grabbag)
        for pt in grabbag:
            grabbag.remove(pt)
            X_store = points_from_indices(X, store)
            X_pt = points_from_indices(X, pt)
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X_store, labels[store, :])
            if neigh.score(X_pt, labels[pt, :]) == 1:
                grabbag.append(pt)
            else:
                store.append(pt)
        if original_length - len(grabbag) == 0:
            break
    return X[store, :], labels[store, :].squeeze()
    
def points_from_indices(X, indices):
    if type(indices) == int or len(indices) == 1:
        return X[indices, :].reshape(1, -1)
    else:
        return X[indices, :]


def old_experiments():

    # print(X)
    print('Original dataset shape %s' % Counter(y))  
    # cnn = CondensedNearestNeighbour(sampling_strategy='all')  
    # X_res, y_res = cnn.fit_resample(X, y)  
    # cnn = CondensedNearestNeighbour(sampling_strategy='auto')  
    # X_res, y_res = cnn.fit_resample(X_res, y_res) 
    # print('Resampled dataset shape %s' % Counter(y_res))
    # print(f"Original training set size: {y.shape[0]}. Compressed set size: {y_res.shape[0]}")
    # X_res, y_res = iterative_prototype_scan(X, y)
    # print('Resampled dataset shape: %s' % Counter(y_res))


    X_res, y_res = original_condense(X, y)
    X_res, y_res = X_res[:, :], y_res[:]
    print('Resampled dataset shape: %s' % Counter(y_res))
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_res, y_res)
    num_correct = neigh.score(X, y) * X.shape[0]
    print(f"Correctly predicted: {num_correct}; Parameters: {y_res.shape[0]}")
    print(f"Bits per parameter: {num_correct / y_res.shape[0]}")

    # X_res, y_res = X[:n//2, :], y[:n//2]
    # print('Resampled dataset shape: %s' % Counter(y_res))
    # neigh = KNeighborsClassifier(n_neighbors=1)
    # neigh.fit(X_res, y_res)
    # num_correct = neigh.score(X, y) * X.shape[0]
    # print(f"Correctly predicted: {num_correct}; Parameters: {y_res.shape[0]}")
    # print(f"Bits per parameter: {num_correct / y_res.shape[0]}")


def generate_table(X, y):
    table = [(sum(X[i]), y[i]) for i in range(X.shape[0])]
    table = sorted(table, key=lambda x: x[0])
    return table

def og_algo_8(X, y, num_classes=2):
    table = generate_table(X, y)
    thresholds = 0
    c = 0
    for i in range(len(table)):
        if not table[i][1] == c:
            c = table[i][1]
            thresholds += 1
    minthreshs = np.log(thresholds) / np.log(num_classes)
    mec = minthreshs * (X.shape[1] + 1) + minthreshs + 1
    return thresholds, minthreshs, mec

def binary_optimal_table(table):
    accuracies = []
    for i in range(3, len(table) - 3):
        below = table[:i]
        above = table[i:]
        zero_below = len(list(filter(lambda x: x[1] == 0, below)))
        one_above = len(list(filter(lambda x: x[1] == 1, above)))
        accuracies.append((max(zero_below + one_above, len(table) - (zero_below + one_above)), i))
    return max(accuracies, key=lambda x: x[0])

results = {}
for n_over_d in np.linspace(1, 2.5, num=10):
# for k in range(1, 10, 2):
    n = 300
    # d = int(n / n_over_d)
    # d = i * 10
    d = 3000
    c = 2
    avg = 0
    trial = []
    for _ in range(5):
    
        X, y = generate_random_data(n, d, c)
        # print(y)
        # clf = sklearn.svm.LinearSVC(C=1.0, loss="hinge")
        # clf.fit(X, y)
        # score = clf.score(X, y) * X.shape[0]

        X_res, y_res = original_condense(X, y)
        points_to_keep = int(n//n_over_d)
        X_res, y_res = X_res[:points_to_keep, :], y_res[:points_to_keep]
        print(len(y_res) / X.shape[0])

        neigh = KNeighborsClassifier(n_neighbors=1)
        # neigh.fit(X_res, y_res)

        neigh.fit(X_res, y_res)

        score = neigh.score(X, y) * n
        print(f"SVM Score: {score}")
        if score / n >= 0.98:
            score = n
        # thresh, minthreshs, mec = og_algo_8(X, y)
        # print(f"SVM Try: {score / thresh}")
        # print(f"Num thresholds: {thresh, minthreshs}")

        trial.append(score / n)

        # memorized = binary_optimal_table(generate_table(X, y))
        # print("Points memorized: ", memorized[0])
        # print(f"i: {memorized[1]}")

        
        # print(f"Points per threshold: {X.shape[0] / thresh}")
    avg = sum(trial) / len(trial)
    percent_correct = sum([1 for t in trial if t == 1]) / len(trial)
    results[n_over_d] = percent_correct
results = list(results.items())
dimensions = [r[0] for r in results]
mem_ratios = [r[1] for r in results]
plt.plot(dimensions, mem_ratios, 'go--')
# plt.title(f'Accuracy vs. Tree Depth for Decision Tree')
# plt.xlabel('Number of Nodes / Number of If/Else Clauses')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.show()
plt.savefig(f"mec.png")

# print(f"Ratio: {thresh / memorized[0]}")

# neigh = KNeighborsClassifier(n_neighbors=1)
# neigh.fit(X, y)
# num_correct = neigh.score(X, y) * X.shape[0]
# print(f"Num correct: {num_correct}")
# # print(n / thresh)