from collections import Counter  
from sklearn.datasets import fetch_openml  
from sklearn.preprocessing import scale  
from imblearn.under_sampling import CondensedNearestNeighbour
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import matplotlib.pyplot as plt
import pdb

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")

def generate_random_data(n, d, num_classes):
    X = 100 * np.random.rand(n, d)
    y = np.zeros((n, 1), dtype=int)
    points_per_class = n // num_classes
    for i in range(num_classes):
        y[i * points_per_class : (i+1) * points_per_class] = i
    X, y = sklearn.utils.shuffle(X, y)
    return X, y.squeeze()

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
        neigh.fit(X_store, labels[store].flatten())
        if neigh.score(X_i, labels[i].flatten()) == 0:
            store.append(i)
        else:
            grabbag.append(i)
    
    while True:
        to_remove = []
        for pt in grabbag:
            X_store = points_from_indices(X, store)
            X_pt = points_from_indices(X, pt)
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X_store, labels[store].flatten())
            if neigh.score(X_pt, labels[pt].flatten()) == 0:
                to_remove.append(pt)
            else:
                pass
        if len(to_remove) == 0:
            break
        else:
            # print(f"loop; {to_remove}")
            for pt in to_remove:
                grabbag.remove(pt)
        # if points_added == 0:
        #     break
    
    # neigh = KNeighborsClassifier(n_neighbors=1)
    # neigh.fit(X[store, :], labels[store, :])
    # print(f"Condense: {neigh.score(X[grabbag, :], labels[grabbag, :])}")
    # y_pred = neigh.predict(X[grabbag, :])
    # Find misclassified points
    # misclassified_indices = np.where(y_pred != labels[grabbag, :].flatten())[0]
    # print(f"Misclassifications: {misclassified_indices}")
    # print(f"Grabbag indices: {[grabbag[int(i)] for i in list(misclassified_indices)]}")
    # print(y_pred)
    # print(labels[grabbag, :].flatten())
    return X[store], labels[store].flatten()

def border_ratio(x, x_label, X, labels):
    """
    Calculate the border ratio for a given data point x in the dataset X with labels.
    """
    # Find indices of examples with different and same labels
    diff_label_indices = np.where(labels != x_label)[0]
    same_label_indices = np.where(labels == x_label)[0]

    # Calculate distance to the nearest example with a different label
    closest_external = np.argmin(np.linalg.norm(X[diff_label_indices] - x, axis=1))
    closest_external = X[diff_label_indices][closest_external]
    closest_to_closest_external = np.argmin(np.linalg.norm(X[same_label_indices] - closest_external, axis=1))
    closest_to_closest_external = X[same_label_indices][closest_to_closest_external]
    ratio = np.linalg.norm(closest_to_closest_external - closest_external) / np.linalg.norm(closest_external - x)
    assert 0 <= ratio <= 1, f"Ratio: {ratio}"
    return ratio


def new_condense(X, labels):
    store = list(range(X.shape[0]))
    while True:
        points_changed = 0
        for i in range(len(store) - 1):
            without_i = store[:i] + store[i+1:]
            pt = store[i]            
            X_store = points_from_indices(X, without_i)
            X_pt = points_from_indices(X, pt)
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X_store, labels[store].flatten())
            if neigh.score(X, labels.flatten()) == 1:
                store.remove(pt)

            without = np.delete(store, i, axis=0)
            without_labels = np.delete(labels, i, axis=0)
            neigh.fit(without, without_labels)
            if neigh.score(X, labels) == 1:
                store = without



def points_from_indices(X, indices):
    if type(indices) == int or len(indices) == 1:
        return X[indices, :].reshape(1, -1)
    else:
        return X[indices, :]


for c in range(2, 6):
    results = {}
    print(f"C: {c}")
    for d in range(2, 10, 2):
        print(f"D: {d}")
        trials = []
        for t in range(100):
            # print(f"trial: {t}")
            n = 2**d
            X, y = generate_random_data(n, d, c)

            # border_ratios = np.array([border_ratio(X[i], y[i], X, y) for i in range(X.shape[0])])
            # # Sort X based on the calculated border ratios
            # sorted_indices = np.flipud(np.argsort(border_ratios))
            # X, y = X[sorted_indices], y[sorted_indices]

            X_res, y_res = original_condense(X, y)
            trials.append(X.shape[0] / X_res.shape[0])
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X_res, y_res)
            s = neigh.score(X, y)
            # if s != 1:
            #     print(f"Warning: s {s}, n {n}, d {d}, c {c}")
        # print(trials)
        results[d] = sum(trials) / len(trials)
    
    item_results = list(results.items())
    dimensions = [r[0] for r in item_results]
    mem_ratios = [r[1] for r in item_results]
    # plt.clf()
    # plt.xlabel("Dimension of data")
    # plt.ylabel("Original set size / minimum to memorize size")
    # plt.plot(dimensions, mem_ratios, 'go--')
    # plt.savefig(f"knn_{c}_classes.png")

    print(f"Number of classes: {c} \
          \nFull set size / minimum set size for d=2: {results[2]} \
          \nFull set size / minimum set size for d=4: {results[4]} \
          \nFull set size / minimum set size for d=8: {results[8]}")