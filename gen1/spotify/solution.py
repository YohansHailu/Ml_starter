import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./spotify.csv")


obj_cols = data.select_dtypes(include=["object"]).columns.tolist()
X = data.drop(["track_popularity"] + obj_cols, axis=1)
y = data["track_popularity"]

X.head()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

pipline = make_column_transformer((StandardScaler(), X_train.columns))
X_train = pipline.fit_transform(X_train)
X_valid = pipline.transform(X_valid)


model = keras.models.Sequential(
    [
        keras.layers.Dense(128, activation="relu", input_shape=[X_train.shape[1]]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ]
)

model.compile(loss="mean_squared_error", optimizer="adam")
model

history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)


y_pred = model.predict(X_valid)
y_valid = np.array(y_valid)
print(y_pred.shape)
print(y_valid.shape)
#print(np.array(y) - y_pred)

exit()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()
