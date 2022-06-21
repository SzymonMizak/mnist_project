#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


# In[10]:


mnist = fetch_openml("mnist_784", version = 1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int8)
X_prevalidsplit, X_test, y_prevalidsplit, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[11]:


X_train, X_valid, y_train, y_valid = train_test_split(X_prevalidsplit, y_prevalidsplit, random_state=42, test_size=0.2)


# In[89]:


y_train = tf.keras.utils.to_categorical(y_train)
y_valid = tf.keras.utils.to_categorical(y_valid)
y_test = tf.keras.utils.to_categorical(y_test)


# Define neural network model as object to use for hyperparameters tunning (N of hidden layers, N of neurons within layer, regularization, batch size)

# In[122]:


def nn_model(hidden_layer_sizes = (784,784,784), input_shape=784, dropout_rate = 0, batch_norm = False):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for hls in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(hls, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        if batch_norm == True:
            model.add(tf.keras.layers.BatchNormalization)
    model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
    return model

clasiffier = KerasClassifier(
    model = nn_model, 
    loss=["categorical_crossentropy"], 
    metrics=["accuracy"],
    optimizer="Adam", 
    optimizer__learning_rate = 0.001, 
    model__hidden_layer_sizes = (784,784,784), 
    model__dropout_rate = 0, 
    model_batch_norm = False
)


# Define early stopping callback for model

# In[118]:


stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5)


# Define hyperparameters space to explore in GridSearch

# In[124]:


nn_hyperparameters = {
    "optimizer__learning_rate": [0.001], 
    "model__hidden_layer_sizes": [(784,), (784, 784,), (784, 784, 784,)],
    "model__dropout_rate": [0, 0.1, 0.2], 
    "model__batch_norm": [False, True]
}


# In[125]:


grid_search_cv = GridSearchCV(clasiffier, nn_hyperparameters, cv = 5)


# In[126]:


grid_search_cv.fit(X_train, y_train, epochs = 100, validation_data = (X_valid, y_valid), callbacks = [stopping_callback])


# In[ ]:




