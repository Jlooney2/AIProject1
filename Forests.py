import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def run():
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
        regressor = RandomForestRegressor(n_estimators=100, max_depth=5, max_leaf_nodes=15, random_state=0)
        regressor.fit(x_train, y_train)
        y_prediction = regressor.predict(x_test)


        err += np.sqrt(metrics.mean_absolute_error(y_test, y_prediction))
        print(np.sqrt(metrics.mean_absolute_error(y_test, y_prediction)))
    print('Average Root Mean Squared Error: ', err/kfold_count)
