import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd

data = pd.read_csv('processed.csv')
df = pd.DataFrame(data)

le = preprocessing.LabelEncoder()
le.fit(df['Town'])
list(le.classes_)
le.transform(df['Town'])
print(df.describe())
print(df)
#df['Town'] = df['Town'].astype(int)
#df['Address'] = data['Address'].astype(int)
#df['PropertyType'] = data['PropertyType'].astype(int)
#df['ResidentialType'] = data['ResidentialType'].astype(int)

#train = df[['ListYear','Town','Address','AssessedValue','PropertyType', 'ResidentialType']]
#label = df[['SaleAmount']]

#model = linear_model.LinearRegression()
#model.fit(train, label)

#instance_to_predict = np.array([2010, 'Andover', '126 SHODDY MILL ROAD', 100000, 'Residential', 'Single Family'])
#instance_to_predict = instance_to_predict.reshape(1,-1)
#prediction = model.predict(instance_to_predict)
#print(f"Predicted y value for x= {instance_to_predict} is {prediction}")