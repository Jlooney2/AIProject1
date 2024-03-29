import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt


def run_kfold():
    dataset = pd.read_csv(f'processed.csv')
    dataset.dropna(inplace=True)
    le = LabelEncoder()
    encoded_dataset = dataset.apply(le.fit_transform)
    ds_size = dataset.keys().size - 1
    x = encoded_dataset.iloc[:, 0:ds_size].values
    y = encoded_dataset.iloc[:, ds_size].values
    average = 0
    kfold_count = 10
    ss = StratifiedKFold(n_splits=kfold_count, shuffle=False, random_state=0)
    for train_index, test_index in ss.split(x, y):
        print('new fold')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        regressor = linear_model.LinearRegression()
        regressor.fit(x_train, y_train)
        y_prediction = regressor.predict(x_test)
        average += metrics.mean_absolute_error(y_test, y_prediction)  
    print('Average mean absolute error: ' + str((average / kfold_count)))

def run():
    time1 = time.time()
    print('RF Reading...:')
    # Read
    dataset = pd.read_csv(f'processed.csv')

    # Drop NaN
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
    ds_size = dataset.keys().size - 1
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
    regressor = linear_model.LinearRegression()
    regressor.fit(x_train, y_train)

    y_prediction = regressor.predict(x_test)

    time2 = time.time()
    print('It took %s seconds to load and train the data.' % (time2 - time1))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_prediction))
    print('Max: ', metrics.max_error(y_test, y_prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
    print(f'R-2: {metrics.r2_score(y_test, y_prediction)}')
    print(f"Coefficients: {regressor.coef_}")

    plt.figure()
    plt.plot(y_prediction[0:30], label='pred')
    plt.plot(y_test[0:30], 'gd', label='test')
    plt.legend(loc="best")
    plt.show()
