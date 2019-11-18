import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import time


def run():
    time1 = time.time()
    print('RF Reading...:')
    # Read
    dataset = pd.read_csv(f'processed.csv')

    ds_size = dataset.keys().size - 1

    print(ds_size)

    # Fill NaN
    print('RF Dropping NaN values...')
    # dataset.fillna(dataset.mean(), inplace=True)
    print('Initial size ', dataset.size)
    dataset.dropna(inplace=True)
    print('Ending size ', dataset.size)

    print('RF Encoding...:')
    # Encode
    le = LabelEncoder()
    encoded_dataset = dataset.apply(le.fit_transform)

    print('RF Setup data...:')
    # Setup data
    x = encoded_dataset.iloc[:, 0:ds_size].values
    y = encoded_dataset.iloc[:, ds_size].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

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

    time2 = time.time()
    print('It took %s seconds to load and train the data.' % (time2 - time1))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_prediction))
    print('Max: ', metrics.max_error(y_test, y_prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
    print(regressor.feature_importances_)

    plt.figure()
    plt.plot(y_prediction[0:30], label='pred')
    plt.plot(y_test[0:30], 'gd', label='test')
    plt.legend(loc="best")
    plt.show()

    # testdata = [[2009,'Hartford','31 WOODLAND ST UNIT 6D',46450.0,'Condo']]
    # print(metrics.mean_absolute_error(, y_prediction))
