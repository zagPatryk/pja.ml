from __future__ import annotations
from typing import Any, NoReturn
import numpy as np
import Common as cm


class Node:
    """
    Building block of each tree
    May contain children nodes (left and right) or be a final node (called "leaf")
    """

    def __init__(self, name=None, data: np.ndarray = None, target: np.ndarray = None, left: Node = None,
                 right: Node = None,
                 curr_depth: int = None, variable: int = None, threshold: float = None, leaf_value: np.ndarray = None,
                 ) -> NoReturn:
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
        self.name = name

    def __str__(self) -> str:
        return f'This node is at level: {self.curr_depth}'

    def grow_tree(self, tree):
        name = self.name
        curr_depth = name.count('.') + 2
        data = cm.attrgetter(tree.root, f'{name}.data')
        target = cm.attrgetter(tree.root, f'{name}.target')

        best_split = tree.find_best_split(data=data, target=target)
        if curr_depth <= tree.max_depth and np.unique(target).size > 1 and best_split['gain'] > 0:
            if tree.fitted_depth < curr_depth:
                tree.fitted_depth = curr_depth
            left_indices = data[:, best_split['variable']] <= best_split['threshold']
            cm.attrsetter(tree.root, f'{name}.variable', best_split['variable'])
            cm.attrsetter(tree.root, f'{name}.threshold', best_split['threshold'])

            left = Node(name=f'{name}.left',
                        data=data[left_indices, :],
                        target=target[left_indices],
                        curr_depth=curr_depth)

            cm.attrsetter(tree.root, f'{name}.left', left)

            right = Node(name=f'{name}.right',
                         data=data[np.invert(left_indices), :],
                         target=target[np.invert(left_indices)],
                         curr_depth=curr_depth)

            cm.attrsetter(tree.root, f'{name}.right', right)

            left.grow_tree(tree)
            right.grow_tree(tree)
        else:
            target_values, target_counts = np.unique(target, return_counts=True)
            cm.attrsetter(tree.root, f'{name}.leaf_value', target_values[target_counts == target_counts.max()])
