import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
import pandas
import numpy as np
import time
import graphviz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def run():
    time1 = time.time()
    dataset = pandas.read_csv("processed.csv")
    dataset = dataset.dropna()

    le = sklearn.preprocessing.LabelEncoder()
    encoded = dataset.apply(le.fit_transform)

    ds_size = dataset.keys().size - 1
    x = encoded.iloc[:, 0:ds_size].values
    y = encoded.iloc[:, ds_size].values
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=0)

    sc = sklearn.preprocessing.StandardScaler()

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    treeRegressor = tree.DecisionTreeRegressor()
    treeRegressor = treeRegressor.fit(x_train, y_train)

    # print(tree.plot_tree(treeRegressor))
    tree.plot_tree(treeRegressor)
    dot_data = tree.export_graphviz(treeRegressor, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("connecticut")

    y_prediction = treeRegressor.predict(x_test)

    time2 = time.time()
    print('It took %s seconds to load and train the data.' % '{0:,.2f}'.format((time2 - time1)))
    print('Mean Absolute Error:', '{0:,.2f}'.format(metrics.mean_absolute_error(y_test, y_prediction)))
    print('Mean Squared Error:', '{0:,.2f}'.format(metrics.mean_squared_error(y_test, y_prediction)))
    print('Maximum Error: ', '{0:,.2f}'.format(metrics.max_error(y_test, y_prediction)))
    print('Root Mean Squared Error:', '{0:,.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_prediction))))
    print(list(dataset))
    print(treeRegressor.feature_importances_)
