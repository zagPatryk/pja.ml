from __future__ import annotations
from typing import Any, NoReturn
import numpy as np
import Common as cm
import Node as nd


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
        self.root = nd.Node()
        self.fitted_depth = 0

    def __str__(self) -> str:
        if self.fitted_depth == 0:
            return 'This tree is still a sapling. There\'s nothing to show'
        else:
            s = f'Tree with max fitted depth of {self.fitted_depth}:\n'
            s += f'root ::: Split at variable {self.root.variable} at {self.root.threshold}\n'
            s = self.prepString('right', s)
            s = self.prepString('left', s)
            return s

    def prepString(self, name, s):
        curr_depth = name.count('.') + 2
        if cm.attrgetter(self.root, f'{name}.variable'):
            s += f"{' ' * curr_depth}{name} ::: Split at variable {cm.attrgetter(self.root, f'{name}.variable')} at " \
                 f"{cm.attrgetter(self.root, f'{name}.threshold')} (" \
                 f"{np.unique(cm.attrgetter(self.root, f'{name}.target'), return_counts=True)[1]})\n"
        else:
            s += f"{' ' * curr_depth}{name} ({np.unique(cm.attrgetter(self.root, f'{name}.target'), return_counts=True)[1]}" \
                 f" and the leaf value is {cm.attrgetter(self.root, f'{name}.leaf_value')})\n"
        if cm.attrgetter(self.root, f'{name}.right'):
            s = self.prepString(f'{name}.right', s)
        if cm.attrgetter(self.root, f'{name}.left'):
            s = self.prepString(f'{name}.left', s)
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
                left_indices = data[:,
                               variable] < threshold  # TODO: Clean it, if possible, as it adds unnecessary complexity
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

        self.fitted_depth = 1

        left = nd.Node(name='left',
                       data=data[left_indices, :],
                       target=target[left_indices],
                       curr_depth=1)

        cm.attrsetter(self.root, 'left', left)
        left.grow_tree(self)

        right = nd.Node(name='right',
                        data=data[np.invert(left_indices), :],
                        target=target[np.invert(left_indices)],
                        curr_depth=1)

        cm.attrsetter(self.root, 'right', right)
        right.grow_tree(self)

    def get_prediction(self, x: np.ndarray, name: str = '') -> np.ndarray:
        """
        Returns predicted class(es) for a given observation (numpy vector)
        :param x: float vector
        :param name: path to an attribute (dot separated)
        :return: vector with predicted class(es)
        """
        if cm.attrgetter(self.root, f'{name}.leaf_value') is not None:
            return cm.attrgetter(self.root, f'{name}.leaf_value')[0]
        if name == '':  # TODO: Check if it's possible to get rid of default case
            if x[cm.attrgetter(self.root, f'variable')] < cm.attrgetter(self.root, f'threshold'):
                return self.get_prediction(x, name='left')
            else:
                return self.get_prediction(x, name='right')
        else:
            if x[cm.attrgetter(self.root, f'{name}.variable')] < cm.attrgetter(self.root, f'{name}.threshold'):
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

#%%
