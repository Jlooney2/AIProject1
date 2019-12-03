import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
    x = encoded_dataset.iloc[:, 0:ds_size].values
    y = encoded_dataset.iloc[:, ds_size].values

    err = 0
    kfold_count = 5
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
        regressor = RandomForestRegressor(n_estimators=100, max_depth=50, max_leaf_nodes=300, random_state=0)
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


def run_grid_search():
    grid_param = {
        'n_estimators': [2500],
        'bootstrap': [True],
        'max_depth': [75, 100, 125],
        'max_leaf_nodes': [65, 75, 85]
    }
    gd_sr = GridSearchCV(estimator=RandomForestRegressor(),
                         param_grid=grid_param,
                         scoring='neg_mean_absolute_error',
                         cv=5,
                         n_jobs=-1)

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
    x = encoded_dataset.iloc[:, 0:ds_size].values
    y = encoded_dataset.iloc[:, ds_size].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    print('RF Scaling...:')
    # Scale data
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    gd_sr.fit(x_train, y_train)

    best_parameters = gd_sr.best_params_
    print(best_parameters)

    best_grid = gd_sr.best_estimator_
    evaluate(best_grid, x_test, y_test)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy



