import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load Home Credit default risk data
train_data = pd.read_csv('./data/application_train.csv')
test_data = pd.read_csv('./data/application_test.csv')
train_data = train_data.fillna(0)

# 'TARGET' columns contains target values: either a 0 for the loan was repaid on time,
# or a 1 indicating the client had payment difficulties
# In plot we see the imbalaced class problem, more loans that were repaid on time than loans that were not repair

train_data['TARGET'].value_counts()
train_data['TARGET'].astype(int).plot.hist()

# Categorical data
train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# Accuracy of major class
train_data['TARGET'].value_counts()[0] / len(train_data['TARGET'])

# Missing values
miss_val_percent = 100 * train_data.isnull().sum() / len(train_data)
miss_val_percent = miss_val_percent[miss_val_percent > 0]
miss_val_percent.sort_values(inplace=True, ascending=False)

# Plot missing data in features
miss_val_percent.plot(kind='barh', figsize=(12,12)) # TODO Clean NaN

# Check categorical features
train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# Replace anomalous column
train_data['DAYS_EMPLOYED_ANOM'] = train_data['DAYS_EMPLOYED'] == 365243
train_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

test_data['DAYS_EMPLOYED_ANOM'] = test_data["DAYS_EMPLOYED"] == 365243
test_data["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


# Check correlations with target column
corr = train_data.corr()['TARGET'].sort_values(ascending=False)
train_data['DAYS_BIRTH'] = abs(train_data['DAYS_BIRTH'])
train_data['DAYS_BIRTH'].corr(train_data['DAYS_BIRTH'])

plt.figure(figsize=(10,8))
sns.kdeplot(train_data.loc[train_data['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'TARGET == 0')
sns.kdeplot(train_data.loc[train_data['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'TARGET == 1')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution of Ages')

ext_data = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
plt.figure(figsize=(10,8))
sns.heatmap(ext_data.corr(), annot=True)

