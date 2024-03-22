import numpy as np
import copy
from collections import Counter
import sklearn

def generate_table(X, y):
    table = [[sum(X[i]), y[i]] for i in range(X.shape[0])]
    table = sorted(table, key=lambda x: x[0])
    return table

def thresholds_to_mec(num_thresholds, d):
    minthreshs = np.log(num_thresholds) / np.log(2) # change of base
    mec = minthreshs * (d + 1) + minthreshs
    return mec

def count_thresholds(table):
    thresholds = 0
    c = table[0][1]
    for i in range(len(table)):
        if not table[i][1] == c:
            c = table[i][1]
            thresholds += 1
    return thresholds

def og_algo_8(X, y):
    table = generate_table(X, y)
    thresholds = count_thresholds(table)
    return thresholds_to_mec(thresholds, X.shape[1])

def expanded_algo_8(X, y):
    count = Counter(y)
    num_classes = len(count)
    table = generate_table(X, y)
    thresholds = 0
    for c in count.keys():
        new_table = copy.deepcopy(table)
        for i in range(len(new_table)):
            if new_table[i][1] != c:
                new_table[i][1] = num_classes + 1
        thresholds += count_thresholds(new_table)
    return thresholds_to_mec(thresholds, X.shape[1])

def generate_random_data(n, d, num_classes):
    X = 100 * np.random.rand(n, d)
    y = np.zeros((n, 1), dtype=int)
    points_per_class = n // num_classes
    for i in range(num_classes):
        y[i * points_per_class : (i+1) * points_per_class] = i
    X, y = sklearn.utils.shuffle(X, y)
    return X, y.squeeze()

X, y = generate_random_data(100, 10, 2)
print(f"Algorithm 8 on random data, 2 classes: {expanded_algo_8(X, y)}")
X, y = generate_random_data(100, 10, 3)
print(f"Algorithm 8 on random data, 3 classes: {expanded_algo_8(X, y)}")
X, y = generate_random_data(100, 10, 100)
print(f"Algorithm 8 on random data, regression: {expanded_algo_8(X, y)}")