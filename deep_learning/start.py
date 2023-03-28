import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector


# Load the data
fuel = pd.read_csv("fuel.csv")


x = fuel.copy()
# Remove target
y = x.pop("FE")

preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False), make_column_selector(dtype_include=object)),
)

x = preprocessor.fit_transform(x)
y = np.log(y)

input_shape = [x.shape[1]]

model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=input_shape),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(1),
    ]
)

model.compile(
        optimizer = "adam",
        loss='binary_crossentropy'
        metrics=['binary_accuracy'],
)

history = model.fit(
    x,
    y,
    batch_size=128,
    epochs=200,
)

y_pred = model.predict(x)
y_pred = np.exp(y_pred)





