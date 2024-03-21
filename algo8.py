import numpy as np
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