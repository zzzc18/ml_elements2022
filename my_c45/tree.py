import numpy as np
import math
from multiprocessing import Pool
import random


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTree():
    def __init__(self, criterion, min_samples_split=2, max_depth=2):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # self.continuous_max_split = 3
        self.continuous_max_split = 20

    def build_tree(self, dataset, curr_depth=1):
        ''' recursive function to build the tree '''

        num_samples, num_features = np.shape(dataset)
        num_features -= 1

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(
                dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(
                    best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(
                    best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(dataset[:, -1])
        # return leaf node
        return Node(value=leaf_value)

    def get_possible_thresholds(self, feature_values: np.ndarray):
        possible_thresholds = np.unique(feature_values)
        possible_thresholds = (
            possible_thresholds[1:]+possible_thresholds[:-1])/2
        length_possible_thresholds = len(possible_thresholds)
        if length_possible_thresholds > self.continuous_max_split:
            tmp = np.linspace(0, length_possible_thresholds-1,
                              self.continuous_max_split)
            tmp = tmp.round().astype(np.int32)
            possible_thresholds = possible_thresholds[tmp]

        return possible_thresholds

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        overall_best_split = {}
        overall_max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = self.get_possible_thresholds(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                curr_info_gain = self.split(
                    dataset, feature_index, threshold)
                # check if childs are not null
                if curr_info_gain > overall_max_info_gain:
                    overall_best_split["feature_index"] = feature_index
                    overall_best_split["threshold"] = threshold
                    overall_best_split["info_gain"] = curr_info_gain
                    overall_max_info_gain = curr_info_gain

        dataset_left = np.array(
            [row for row in dataset if row[overall_best_split["feature_index"]] <= overall_best_split["threshold"]])
        dataset_right = np.array(
            [row for row in dataset if row[overall_best_split["feature_index"]] > overall_best_split["threshold"]])
        overall_best_split["dataset_left"] = dataset_left
        overall_best_split["dataset_right"] = dataset_right
        return overall_best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        left_y = []
        right_y = []
        for row_index in range(dataset.shape[0]):
            if dataset[row_index, feature_index] <= threshold:
                left_y.append(dataset[row_index, -1])
            else:
                right_y.append(dataset[row_index, -1])
        left_y = np.array(left_y)
        right_y = np.array(right_y)

        # check if childs are not null
        if len(left_y) > 0 and len(right_y) > 0:
            y = dataset[:, -1]
            # compute information gain
            curr_info_gain = self.information_gain(
                y, left_y, right_y, self.criterion)
            return curr_info_gain
        else:
            return -float("inf")

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(
                parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        if mode == "entropy":
            gain = self.entropy(
                parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        if mode == "C4.5":
            gain = self.entropy(
                parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
            gain = gain/self.prob_entropy(np.array([weight_l, weight_r]))
        return gain

    def prob_entropy(self, p: np.ndarray):
        entropy = -(p*np.log2(p)).sum()
        return entropy

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=",
                  tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''

        Y = np.expand_dims(Y, axis=-1)
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
