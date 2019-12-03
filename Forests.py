import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import copy
import time as t


def run():
    time1 = t.time()
    print('RF Reading...:')
    # Read
    dataset = pd.read_csv(f'processed.csv')

    # Fill NaN
    print('RF Dropping NaN values...')
    print('Initial size ', dataset.shape)
    dataset.dropna(inplace=True)
    print('Ending size ', dataset.shape)

    print('RF Encoding...:')
    # Encode
    le = LabelEncoder()
    encoded_dataset = dataset.apply(le.fit_transform)

    print('RF Setup data...:')
    # Setup data
    ds_size = dataset.keys().size - 1
    x = encoded_dataset.iloc[:10000, 0:ds_size].values
    y = encoded_dataset.iloc[:10000, ds_size].values

    err = 0
    kfold_count = 10
    best = 100000000000
    best_model = None
    ss = StratifiedKFold(n_splits=kfold_count, shuffle=True, random_state=42)
    for train_index, test_index in ss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print('RF Scaling...:')
        # Scale data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        print('RF Training...:')
        # Train
        regressor = RandomForestRegressor(n_estimators=10, max_depth=50, max_leaf_nodes=300, random_state=0)
        regressor.fit(x_train, y_train)
        y_prediction = regressor.predict(x_test)

        curr_err = np.sqrt(metrics.mean_absolute_error(y_test, y_prediction))
        err += curr_err
        print(curr_err)

        if best is None or best > curr_err:
            best_model = copy.deepcopy(regressor)

    print('Average Absolute Mean Error: ', err / kfold_count)

    x = sc.transform(x)
    y_prediction = best_model.predict(x)

    time2 = t.time()
    print("Forest took %s minutes to complete" % ((time2 - time1) / 60))
    print("Importances: ", best_model.feature_importances_)
    print('Mean Absolute Error On Whole Dataset:', metrics.median_absolute_error(y, y_prediction))
    plt.figure()
    plt.plot(y_prediction[0:30], label='pred')
    plt.plot(y_test[0:30], 'gd', label='test')
    plt.legend(loc="best")
    plt.show()
