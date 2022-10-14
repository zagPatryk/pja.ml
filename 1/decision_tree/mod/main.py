from DecisionTree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
    tree = DecisionTree(max_depth=6)
    tree.fit(X_train, y_train)
    print(tree)
    print(f'Rough test: Prediction={tree.get_prediction(X_test[0, :])}, True Value={y_test[0]}')


if __name__ == '__main__':
    main()

#%%
