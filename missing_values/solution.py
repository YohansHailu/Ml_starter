import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


X = pd.read_csv('./melb_data.csv')
y = X.Price
predictors = X.drop(['Price'], axis=1)
X = predictors.select_dtypes(exclude=['object'])
print(X.isnull().sum())

# split to training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, train_size=0.8, test_size=0.2)


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# get column names with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

reduced_X_train = train_X.drop(cols_with_missing, axis=1)
reduced_X_val = val_X.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_val, train_y, val_y))

# Imputation based on meadiean 
my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X)) 
imputed_X_val = pd.DataFrame(my_imputer.transform(val_X))

# Imputation removed column names; put them back
imputed_X_train.columns = train_X.columns
imputed_X_val.columns = val_X.columns


print("MAE from Approach 1 (imputing):")
print(score_dataset(imputed_X_train, imputed_X_val, train_y, val_y))



train_X_plus = train_X.copy()
val_X_plus = val_X.copy()

# using extension of imputation 
for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X[col].isnull()


my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus)) 
imputed_X_val_plus = pd.DataFrame(my_imputer.transform(val_X_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = train_X_plus.columns
imputed_X_val_plus.columns = val_X_plus.columns


print("MAE from Approach 1 (imputing with extension):")
print(score_dataset(imputed_X_train_plus, imputed_X_val_plus, train_y, val_y))



