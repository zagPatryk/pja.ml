# Author: Dominik Deja
# First created: 20.06.2021
# This is a stack-based implementation of a decision tree

from __future__ import annotations
from typing import Any, NoReturn
import numpy as np


def attrgetter(obj: object, name: str, value: Any = None) -> Any:
    """
    Returns value from nested objects/chained attributes (basically, getattr() on steroids)
    :param obj: Primary object
    :param name: Path to an attribute (dot separated)
    :param value: Default value returned if a function fails to find the requested attribute value
    :return:
    """
    for attribute in name.split('.'):
        obj = getattr(obj, attribute, value)
    return obj


def attrsetter(obj: object, name: str, value: Any) -> NoReturn:
    """
    Sets the value of an attribute of a (nested) object (basically, setattr() on steroids)
    :param obj: Primary object
    :param name: Path to an attribute (dot separated)
    :param value: Value to be set
    """
    pre, _, post = name.rpartition('.')
    setattr(attrgetter(obj, pre) if pre else obj, post, value)


class Node:
    """
    Building block of each tree
    May contain children nodes (left and right) or be a final node (called "leaf")
    """
    def __init__(self, data: np.ndarray = None, target: np.ndarray = None, left: Node = None, right: Node = None,
                 curr_depth: int = None, variable: int = None, threshold: float = None, leaf_value: np.ndarray = None) -> NoReturn:
        """
        :param data: NxM (where N denotes #observations and M denotes #variables) numpy array containing independent variables
        :param target: numpy vector containing dependent variable
        :param left: (if exists) node containing observations smaller than a given threshold at a given variable
        :param right: (if exists) node containing observations bigger than a given threshold at a given variable
        :param curr_depth: number of parent nodes directly above the current node
        :param variable: variable used to split data
        :param threshold: threshold at which data was split
        :param leaf_value: (if a node is a leaf, i.e. a final node with no children) the most frequent value(s) of a
                           dependent variable in a given node
        """
        self.data = data
        self.target = target
        self.left = left
        self.right = right
        self.curr_depth = curr_depth
        self.variable = variable
        self.threshold = threshold
        self.leaf_value = leaf_value

    def __str__(self) -> str:
        return f'This node is at level: {self.curr_depth}'


class DecisionTree:
    """
    Classification decision tree
    Data must be provided on instance creation, then fit() can be used to fit the tree to the data and
    predict() to predict classes of the new data samples
    """
    def __init__(self, max_depth: int = 3) -> NoReturn:
        """
        :param max_depth: maximum depth of a tree
        """
        self.max_depth = max_depth
        self.root = Node()
        self.fitted_depth = 0

    def __str__(self) -> str:
        if self.fitted_depth == 0:
            return 'This tree is still a sapling. There\'s nothing to show'
        else:
            s = f'Tree with max fitted depth of {self.fitted_depth}:\n'
            s += f'root ::: Split at variable {self.root.variable} at {self.root.threshold}\n'
            stack = ['left', 'right']
            while stack:
                name = stack.pop()
                curr_depth = name.count('.') + 2
                if attrgetter(self.root, f'{name}.variable'):
                    s += f"{' ' * curr_depth}{name} ::: Split at variable {attrgetter(self.root, f'{name}.variable')} at " \
                         f"{attrgetter(self.root, f'{name}.threshold')} (" \
                         f"{np.unique(attrgetter(self.root, f'{name}.target'), return_counts=True)[1]})\n"
                else:
                    s += f"{' ' * curr_depth}{name} ({np.unique(attrgetter(self.root, f'{name}.target'), return_counts=True)[1]}" \
                         f" and the leaf value is {attrgetter(self.root, f'{name}.leaf_value')})\n"
                if attrgetter(self.root, f'{name}.left'):
                    stack.append(f'{name}.left')
                if attrgetter(self.root, f'{name}.right'):
                    stack.append(f'{name}.right')
            return s

    @staticmethod
    def entropy(x: np.ndarray) -> float:
        """
        Entropy, as defined in information theory (https://en.wikipedia.org/wiki/Entropy_(information_theory))
        :param x: vector of real values
        :return: entropy of a vector x
        """
        if x.size == 0:
            return 0
        else:
            counts = np.unique(x, return_counts=True)[1]
            norm_counts = counts / counts.sum()
            return -(norm_counts * np.log(norm_counts)).sum()

    def information_gain(self, parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
        """
        Information gain, as defined on Wikipedia (https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
        :param parent: float vector
        :param left_child: float vector
        :param right_child: float vector
        :return: float denoting information gain for a given split (parent into its children)
        """
        return self.entropy(parent) - (left_child.size / parent.size * self.entropy(left_child) +
                                       right_child.size / parent.size * self.entropy(right_child))

    @staticmethod
    def moving_average(x: np.ndarray, w: int) -> np.ndarray:
        """
        Moving average of vector x with w-wide window
        :param x: float vector
        :param w: width of the moving window
        :return: float vector with averaged values
        """
        return np.convolve(x, np.ones(w), 'valid') / w

    def find_best_split(self, data: np.ndarray, target: np.ndarray) -> dict:
        """
        Searches for the best split on a given data with respect to the dependent variable (using information gain criterion)
        :param data: NxM (where N denotes #observations and M denotes #variables) numpy array containing independent variables
        :param target: numpy vector containing dependent variable
        :return: dictionary with best split variable, threshold and gain
        """
        best_split = {'variable': None,
                      'threshold': None,
                      'gain': -1}
        if np.unique(target).size == 1:
            return best_split
        for variable in range(data.shape[1]):
            indices = data[:, variable].argsort()
            # Threshold is set to be a point in between two values (in a monotonically increasing set of unique values)
            thresholds = self.moving_average(data[indices, variable], 2)
            for threshold in thresholds:
                left_indices = data[:, variable] < threshold  # TODO: Clean it, if possible, as it adds unnecessary complexity
                gain = self.information_gain(target, target[left_indices], target[np.invert(left_indices)])
                if gain > best_split['gain']:
                    best_split['variable'] = variable
                    best_split['threshold'] = threshold
                    best_split['gain'] = gain
        return best_split


    def fit(self, data: np.ndarray = None, target: np.ndarray = None) -> NoReturn:
        """
        Grows a binary classification tree using greedy approach and information gain criterion
        :param data: NxM (where N denotes #observations and M denotes #variables) numpy array containing independent variables
        :param target: numpy vector containing dependent variable
        """
        best_split = self.find_best_split(data, target)
        left_indices = data[:, best_split['variable']] < best_split['threshold']
        self.root.variable = best_split['variable']
        self.root.threshold = best_split['threshold']
        attrsetter(self.root, 'left', Node(data=data[left_indices, :],
                                           target=target[left_indices],
                                           curr_depth=1))
        attrsetter(self.root, 'right', Node(data=data[np.invert(left_indices), :],
                                            target=target[np.invert(left_indices)],
                                            curr_depth=1))
        self.fitted_depth = 1
        stack = ['left', 'right']
        while stack:
            name = stack.pop()
            curr_depth = name.count('.') + 2
            data = attrgetter(self.root, f'{name}.data')
            target = attrgetter(self.root, f'{name}.target')

            best_split = self.find_best_split(data=data, target=target)
            if curr_depth <= self.max_depth and np.unique(target).size > 1 and best_split['gain'] > 0:
                if self.fitted_depth < curr_depth:
                    self.fitted_depth = curr_depth
                left_indices = data[:, best_split['variable']] <= best_split['threshold']
                attrsetter(self.root, f'{name}.variable', best_split['variable'])
                attrsetter(self.root, f'{name}.threshold', best_split['threshold'])
                attrsetter(self.root, f'{name}.left', Node(data=data[left_indices, :],
                                                           target=target[left_indices],
                                                           curr_depth=curr_depth))
                attrsetter(self.root, f'{name}.right', Node(data=data[np.invert(left_indices), :],
                                                            target=target[np.invert(left_indices)],
                                                            curr_depth=curr_depth))
                stack.append(f'{name}.left')
                stack.append(f'{name}.right')
            else:
                target_values, target_counts = np.unique(target, return_counts=True)
                attrsetter(self.root, f'{name}.leaf_value', target_values[target_counts == target_counts.max()])

    def get_prediction(self, x: np.ndarray, name: str = '') -> np.ndarray:
        """
        Returns predicted class(es) for a given observation (numpy vector)
        :param x: float vector
        :param name: path to an attribute (dot separated)
        :return: vector with predicted class(es)
        """
        if attrgetter(self.root, f'{name}.leaf_value') is not None:
            return attrgetter(self.root, f'{name}.leaf_value')[0]
        if name == '':   # TODO: Check if it's possible to get rid of default case
            if x[attrgetter(self.root, f'variable')] < attrgetter(self.root, f'threshold'):
                return self.get_prediction(x, name='left')
            else:
                return self.get_prediction(x, name='right')
        else:
            if x[attrgetter(self.root, f'{name}.variable')] < attrgetter(self.root, f'{name}.threshold'):
                return self.get_prediction(x, name=f'{name}.left')
            else:
                return self.get_prediction(x, name=f'{name}.right')

    def predict(self, new_data: np.ndarray) -> list:
        """
        Returns predicted classes for given observations
        :param new_data: NxM (where N denotes #observations and M denotes #variables) numpy array
        :return: list with predicted classes
        """
        return [self.get_prediction(x) for x in new_data]

    # def fit(self):  # Would be great, but not working due to the lack of explicit pointers in Python
    #     stack = [self.root]
    #     depth, max_depth = 0, 3
    #     while stack:
    #         node = stack.pop()  # I would like to pass it by reference
    #         node = Node()
    #         if split > 0:  # split_yes (dependent on depth and max_depth)
    #             node.left = None  # So that I'm adding attribute left to self.root (and further expanding)
    #             node.right = None
    #             stack.append(node.left)
    #             stack.append(node.right)
    #             print(stack)
    #             split += -1
