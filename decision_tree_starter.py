# You may want to install "gprof2dot"
import io
from collections import Counter, defaultdict

import numpy as np
import scipy.io
import scipy.stats
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import random
import pydot
import matplotlib.pyplot as plt
import pandas as pd

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, feat_amt=-1, pos="s"):
        self.max_depth = max_depth
        self.features = feature_labels
        self.feat_amt = feat_amt
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.position = pos

    
    @staticmethod
    def entropy(x_idxs, y):
        ent = 0
        classes = defaultdict(lambda: 0)
        for i in x_idxs:
            classes[y[i]] += 1
        total_items = x_idxs.shape[0]
        for c in classes:
            p_c = classes[c] / total_items
            ent += p_c * np.log2(p_c)
        return -ent
    
    @staticmethod
    def information_gain(X, y, thresh):
        # TODO: implement information gain function
        left_idxs = (X <= thresh).nonzero()[0]
        right_idxs = (X > thresh).nonzero()[0]
        left_entropy = __class__.entropy(left_idxs, y)
        right_entropy = __class__.entropy(right_idxs, y)
        before_entropy = __class__.entropy(np.arange(X.shape[0]), y)
        H_after = left_idxs.shape[0] * left_entropy + right_idxs.shape[0] * right_entropy
        H_after = H_after / (left_idxs.shape[0] + right_idxs.shape[0])
        return before_entropy - H_after
    

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO: implement gini impurity function
        pass

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):

        if self.feat_amt == -1:
            feat_subset = list(range(X.shape[1]))
            X_bag = X
        else:
            shuff = random.sample(list(range(X.shape[1])), X.shape[1])
            feat_subset = shuff[:self.feat_amt]
            X_bag = X[:,feat_subset]
            # print(f"Feature amount: {self.feat_amt}")
            # print(shuff)
            # print(X_bag.shape)

        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X_bag[:, i]) + eps, np.max(X_bag[:, i]) - eps, num=10)
                for i in range(X_bag.shape[1])
            ])
            for i in range(X_bag.shape[1]):
                gains.append([self.information_gain(X_bag[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            self.split_idx = feat_subset[self.split_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, pos=self.position + "l")
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, pos=self.position + "r")
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y, keepdims=True).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y, keepdims=True).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())
        
    def node_name(self):
        pos = f"Position: {self.position}\n"
        if self.max_depth == 0:
            return pos + "Prediction: %s ---- Amount of Points: %s" % (self.pred, self.labels.size)
        return pos + f"{self.features[self.split_idx]} < {self.thresh}"


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200, subset_scale=1):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.subset_scale = subset_scale
        self.decision_trees = [
            #sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            DecisionTree(**self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        i = 0
        for t in self.decision_trees:
            i += 1
            if i % 5 == 0:
                print(f"Fitting tree #{i}")
            subsample, subsample_labels = self.gen_subset(X, y)
            t.fit(subsample, subsample_labels)
    
    def gen_subset(self, X, y):
        num_samples = int(X.shape[0] * self.subset_scale)
        idxs = np.random.randint(0, high=X.shape[0], size=num_samples)
        #print(idxs)
        return X[idxs], y[idxs]

    def predict(self, X):
        yhat = []
        predictions = []
        for t in self.decision_trees:
            predictions.append(t.predict(X))
        predictions = np.array(predictions)
        predictions = predictions.T
        #print(f"Predictions matrix: {predictions}")
        for i in range(X.shape[0]):
            points_pred = predictions[i,:]
            avg = sum(points_pred) / self.n
            #if avg != 0 and avg != 1:
                #print(f"Average for point {i}: {avg}")
            yhat.append(scipy.stats.mode(points_pred, keepdims=True)[0][0])
        return yhat

class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=0):
        if params is None:
            params = {}
        self.params = params
        self.m = m
        self.n = n
        params["feat_amt"] = self.m
        if "subset_scale" in params:
            s = params["subset_scale"]
            params.pop("subset_scale")
        else:
            s = 1
        super().__init__(params=params, n=self.n, subset_scale=s)

class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO: implement function
        return self

    def predict(self, X):
        # TODO: implement function
        pass

def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == ''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == '-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode
            # if type(mode) != float or type(mode) != int:
            #     mode = mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y, scoring="accuracy"))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)

def shuffle_data(t_data, t_labels, data_name):
    indices = list(range(t_data.shape[0]))
    random.shuffle(indices)
    shuffled_labels = []
    shuffled_data = []
    while indices:
        idx = indices.pop()
        shuffled_labels.append(t_labels[idx])
        shuffled_data.append(t_data[idx])
    shuffled_labels = np.array(shuffled_labels)
    shuffled_data = np.array(shuffled_data)
    print(f"{data_name} shuffled! Data shape: {shuffled_data.shape}")
    # print(shuffled_labels[1:10])
    return shuffled_data, shuffled_labels

def partition(t_data, t_labels):
    validation_size = int(0.2 * t_data.shape[0])
    return t_data[:validation_size], t_labels[:validation_size], t_data[validation_size:], t_labels[validation_size:]

def shuffle_and_partition(t_data, t_labels, data_name):
    shuffled_data, labels = shuffle_data(t_data, t_labels, data_name)
    return partition(shuffled_data, labels)

def make_single_line_plot(xs, ys):
    plt.plot(xs, ys)
    plt.title(f'Validation Accuracy vs. Tree Depth for Decision Tree')
    plt.xlabel('Number of Nodes / Number of If/Else Clauses')
    plt.ylabel('Validation Accuracy')
    # plt.show()
    plt.savefig("sweep.png")

def depth_sweep(train_data, train_labels, valid_data, valid_labels):
    depths = []
    accuracies = []
    node_counts = []
    for i in range(1, 41):
        depths.append(i)
        dt = DecisionTree(max_depth=i, feature_labels=features)
        dt.fit(train_data, train_labels)
        train_pred = dt.predict(train_data)
        train_acc = sklearn.metrics.accuracy_score(train_pred, train_labels)
        valid_pred = dt.predict(valid_data)
        valid_acc = sklearn.metrics.accuracy_score(valid_pred, valid_labels)
        print(f"Tree Depth: {i}")
        print(f"Training accuracy: {train_acc}")
        print(f"Validation accuracy: {valid_acc}")
        accuracies.append(valid_acc)
        node_counts.append(count_nodes(dt))
    make_single_line_plot(node_counts, accuracies)

def results_to_csv(y_test, name):
    y_test = np.array(y_test)
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(f'{name}_submission.csv', index_label='Id')
    print("CSV Saved!")

def visualize(dt, dataset):
    graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="white")
    nodes = []
    populate_graph(dt, graph, nodes)
    last_name =  nodes[-1]
    graph.write_png(f"{dataset}_output.png")
    #plt.imshow(im)

def populate_graph(dt, graph, nodes):
    this_name = dt.node_name()
    right = dt.right
    left = dt.left
    if right != None and left != None:
        name = left.node_name()
        graph.add_node(pydot.Node(name, label=name, shape="box"))
        graph.add_edge(pydot.Edge(this_name, name, color="blue"))
        nodes.append(name)
        name = right.node_name()
        graph.add_node(pydot.Node(name, label=name, shape="box"))
        graph.add_edge(pydot.Edge(this_name, name, color="blue"))
        nodes.append(name)
        populate_graph(left, graph, nodes)
        populate_graph(right, graph, nodes)


def setup(dataset):
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = '/home/pingpong-michael/code/294/dataset/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None, encoding=None)
        path_test = '/home/pingpong-michael/code/294/dataset/titanic/titanic_test_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None, encoding=None)
        y = data[1:, -1]  # label = survived
        class_names = ["Died", "Survived"]
        labeled_idx = np.where(y != '')[0]

        y = np.array(y[labeled_idx])
        y = y.astype(float).astype(int)


        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, :-1], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, :-1]) + onehot_features
        sklearn_params = {
        "max_depth": 5,
        #"random_state": 6,
        "min_samples_leaf": 10,
        }
        bagging_params = {
            "max_depth": 8,
        }
        rforest_params = {
            "max_depth": 8,
            "subset_scale": 1
        }

    elif dataset == "spam":
        sklearn_params = {
        "max_depth": 5,
        #"random_state": 6,
        "min_samples_leaf": 10,
        }
        dtree_params = {
            "max_depth": 8,
        }
        bagging_params = {
            "max_depth": 8,
        }
        rforest_params = {
            "max_depth": 8,
            "subset_scale": 1
        }
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = '/home/pingpong-michael/code/294/dataset/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    return X, y, Z, sklearn_params, bagging_params, rforest_params, features, class_names    

def count_nodes(root: DecisionTree):
    if root is None:
        return 0
    
    return 1 + count_nodes(root.left) + count_nodes(root.right)

if __name__ == "__main__":
    random.seed(727272)
    np.random.seed(6312)
    launch = {"basic": False, "sklearn": False, "bagged": False, "forest": False, "sweep": True}
    dataset = "spam"

    X, y, Z, sklearn_params, bagging_params, rforest_params, features, class_names = setup(dataset)
    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)
    valid_data, valid_labels, train_data, train_labels = shuffle_and_partition(X, y, dataset)
    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)
    # Basic decision tree
    if launch["basic"]:
        print("\n\nPart (a-b): simplified decision tree")
        dt = DecisionTree(max_depth=8, feature_labels=features)
        dt.fit(train_data, train_labels)
        out = io.StringIO()
        train_pred = dt.predict(train_data)
        train_acc = sklearn.metrics.accuracy_score(train_pred, train_labels)
        valid_pred = dt.predict(valid_data)
        valid_acc = sklearn.metrics.accuracy_score(valid_pred, valid_labels)
        
        print(f"Training accuracy: {train_acc}")
        print(f"Validation accuracy: {valid_acc}")
        print(f"Number of nodes: {count_nodes(dt)}")

    if launch["sklearn"]:
        print("\n\nPart (c): sklearn's decision tree")
        clf = sklearn.tree.DecisionTreeClassifier(criterion="entropy", **sklearn_params)
        clf.fit(X, y)
        evaluate(clf)
        out = io.StringIO()
        # You may want to install "gprof2dot"
        sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
        graph = pydot.graph_from_dot_data(out.getvalue())
        pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)


    if launch["bagged"]:
        print("\n\nPart (d): bagged trees")
        dt = BaggedTrees(params=bagging_params, n=60, subset_scale=1)
        dt.fit(train_data, train_labels)
        train_pred = dt.predict(train_data)
        train_acc = sklearn.metrics.accuracy_score(train_pred, train_labels)
        valid_pred = dt.predict(valid_data)
        for t in dt.decision_trees:
            th_pred = t.predict(valid_data)
            th_acc = sklearn.metrics.accuracy_score(th_pred, valid_labels)
            print(f"Validation accuracy for tree: {th_acc}")
        valid_acc = sklearn.metrics.accuracy_score(valid_pred, valid_labels)
        #print("Predictions", dt.predict(Z)[:10])
        print(f"Training accuracy: {train_acc}")
        print(f"Validation accuracy: {valid_acc}")

    if launch["forest"]:
        print("\n\nPart (e): random forest")
        dt = RandomForest(params=rforest_params, n=100, m=int(np.sqrt(train_data.shape[1])))
        dt.fit(train_data, train_labels)
        train_pred = dt.predict(train_data)
        train_acc = sklearn.metrics.accuracy_score(train_pred, train_labels)
        valid_pred = dt.predict(valid_data)
        valid_acc = sklearn.metrics.accuracy_score(valid_pred, valid_labels)
        #print("Predictions", dt.predict(Z)[:10])
        print(f"Training accuracy: {train_acc}")
        print(f"Validation accuracy: {valid_acc}")
        test_pred = dt.predict(Z)
        results_to_csv(test_pred, dataset)

    if launch["sweep"]:
        depth_sweep(train_data, train_labels, valid_data, valid_labels)

