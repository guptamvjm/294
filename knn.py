import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import matplotlib.pyplot as plt

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
        if neigh.score(X_i, labels[i].flatten()) == 1:
            grabbag.append(i)
        else:
            store.append(i)
    while True:
        to_remove = []
        for pt in grabbag:
            X_store = points_from_indices(X, store)
            X_pt = points_from_indices(X, pt)
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X_store, labels[store].flatten())
            if neigh.score(X_pt, labels[pt].flatten()) == 1:
                to_remove.append(pt)
            else:
                # store.append(pt)
                pass
        if len(to_remove) == 0:
            break
        else:
            for pt in to_remove:
                grabbag.remove(pt)
    
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
                
    prunedX, prunedLabels = prune_many(X, labels, store)
    return prunedX, prunedLabels
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

def prune_one(X, labels, store, original_score):
    for i in range(len(store) - 1):
        without_i = store[:i] + store[i+1:]
        pt = store[i]            
        X_store = points_from_indices(X, without_i)
        X_pt = points_from_indices(X, pt)
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X_store, labels[without_i].flatten())
        if neigh.score(X, labels.flatten()) == original_score:
            store.remove(pt)
            return store
    return None

def prune_many(X, labels, store):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X[store], labels[store].flatten())
    original_score = neigh.score(X, labels.flatten())
    while True:
        maybe_store = prune_one(X, labels, store, original_score)
        if maybe_store is None:
            return X[store], labels[store].flatten()
        else:
            store = maybe_store




def points_from_indices(X, indices):
    if type(indices) == int or len(indices) == 1:
        return X[indices, :].reshape(1, -1)
    else:
        return X[indices, :]


for c in range(2, 6):
    results = {}
    print(f"C: {c}")
    # for d in range(16, 256, 16):
    for d_exp in range(3, 9):
        d = 2 ** d_exp
        print(f"D: {d}")
        # if 2**(int(d//8)) // d < 3:
        #     continue
        trials = []
        for t in range(20):
            print(f"trial: {t}")
            # n = 2**(int(d//8))
            n = d * 10
            X, y = generate_random_data(n, d, c)

            border_ratios = np.array([border_ratio(X[i], y[i], X, y) for i in range(X.shape[0])])
            # Sort X based on the calculated border ratios
            sorted_indices = np.flipud(np.argsort(border_ratios))
            X, y = X[sorted_indices], y[sorted_indices]

            X_res, y_res = original_condense(X, y)
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X_res, y_res)
            s = neigh.score(X, y)
            trials.append(X.shape[0] * s / X_res.shape[0])
            
            if s != 1:
                print(f"Warning: s {s}, n {n}, d {d}, c {c}")
        # print(trials)
        results[d] = sum(trials) / len(trials)
        print(f"Compression: {results[d]}")
    item_results = list(results.items())
    dimensions = [r[0] for r in item_results]
    mem_ratios = [r[1] for r in item_results]
    # plt.clf()
    # plt.xlabel("Dimension of data")
    # plt.ylabel("Original set size / minimum to memorize size")
    # plt.plot(dimensions, mem_ratios, 'go--')
    # plt.savefig(f"knn_{c}_classes.png")
    print(results)
    # print(f"Number of classes: {c} \
    #       \nFull set size / minimum set size for d=2: {results[2]} \
    #       \nFull set size / minimum set size for d=4: {results[4]} \
    #       \nFull set size / minimum set size for d=8: {results[8]} \
    #       \nFull set size / minimum set size for d=10: {results[10]}")