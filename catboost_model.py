import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

train_path = './input/train_sets/'
cd_file = './input/train.cd'


# train data
train_data = pd.read_csv('./data/application_train.csv')
test_data = pd.read_csv('./data/application_test.csv')


train_data.fillna(-999, inplace=True)
test_data.fillna(-999, inplace=True)


params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'use_best_model': True
}

X = train_data.drop('TARGET', axis=1)
y = train_data.TARGET
categorical_features_indices = np.where(X.dtypes != np.float)[0]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)

model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validate_pool)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True})

best_model = CatBoostClassifier(**best_model_params)
best_model.fit(train_pool, eval_set=validate_pool)


# Validation sets
def generate_train_sets(df):
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(kf.split(range(len(df)))):
        train = df.loc[train_index]
        valid = df.loc[valid_index]
        train.to_csv(train_path+f"/train_%d.csv"%i, index=False)
        valid.to_csv(train_path+f"/valid_%d.csv"%i, index=False)


def get_cat_column_indexes(df):
    indexes = []
    list_cols = list(df.select_dtypes('object').apply(pd.Series.nunique, axis=0).index)
    for col in list_cols:
        indexes.append(train_data.columns.get_loc(col))
    return indexes


def catboost_train():
    K = 10
    # test = np.zeros((len(test_data), 1))


    for i in range(K):
        train_df = pd.read_csv(train_path + f'train_%d.csv'%i)
        val_df = pd.read_csv(train_path + f'valid_%d.csv'%i)

        X_train = train_df.drop(['TARGET','SK_ID_CURR'], axis=1)
        y_train = train_df.TARGET
        X_val = val_df.drop(['TARGET','SK_ID_CURR'], axis=1)
        y_val = val_df.TARGET
        categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        val_pool = Pool(X_val, y_val, cat_features=categorical_features_indices)

        model = CatBoostClassifier(iterations=5000, learning_rate=0.05, eval_metric='AUC', depth=10, random_seed=42)
        model.fit(train_pool, eval_set=val_pool)


catboost_train()