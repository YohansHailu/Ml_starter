## print current folder
import os
print(os.getcwd())

## import panda_to_tf_dataset
import tensorflow_decision_forests as tfdf
import pandas as pd
import tensorflow as tf
import numpy as np

##
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

##
data_set = pd.read_csv("./Titanic/train.csv")
data_set.shape

##
data_set.head()

##
data_set.describe()

##
data_set.info()

##
data_set.isnull().sum()

##
type(data_set["Transported"].values)

##
data_set.drop(["PassengerId", "Name"], axis=1, inplace=True)
data_set.head()

##
data_set.isnull().sum().sort_values(ascending=False)

##
bool_cols = ['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in bool_cols:
    data_set[col].fillna(0, inplace=True)
    #data_set[col] = data_set[col].astype('bool')

data_set.isnull().sum().sort_values(ascending=False)


##
data_set['Transported'] = data_set['Transported'].astype('int')
data_set['VIP'] = data_set['VIP'].astype('int')
data_set['CryoSleep'] = data_set['CryoSleep'].astype('int')

##
data_set[["Deck", "Cabin_num", "Side"]] = data_set["Cabin"].str.split("/", expand=True)
##
try:
    data_set= data_set.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

##
valid_index = np.random.rand(len(data_set)) < 0.2
train_pd = data_set[~valid_index]
valid_pd = data_set[valid_index]

##
train = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd, label="Transported")
valid = tfdf.keras.pd_dataframe_to_tf_dataset(valid_pd, label="Transported")

## Model selection
tfdf.keras.get_all_models()

##

rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"])

## train the model
history = rf.fit(x=train)

## visuaize the model
import matplotlib.pyplot as plt
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.show()

##
inspector = rf.make_inspector()
inspector.evaluation()

##
evaluation = rf.evaluate(x=valid,return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")



## submission

test_df = pd.read_csv("./Titanic/test.csv")
PassengerId = test_df["PassengerId"]
test_df.head()

##

try:
    test_df.drop(["PassengerId", "Name"], axis=1, inplace=True)
except KeyError:
    print("Field does not exist")

##
bool_cols = ['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in bool_cols:
    test_df[col].fillna(0, inplace=True)

##

test_df['VIP'] = test_df['VIP'].astype('int')
test_df['CryoSleep'] = test_df['CryoSleep'].astype('int')

##
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
##
try:
    test_df = test_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

##
test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

##

##
predictions = rf.predict(test)
predictions = (predictions > 0.5).astype("bool")

##
predictions
##

output = pd.DataFrame({'PassengerId': PassengerId, 'Transported': predictions.flatten()})

output.head()

##

output.to_csv('tf_random_forest_submission.csv', index=False)
