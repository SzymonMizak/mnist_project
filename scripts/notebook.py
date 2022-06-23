#!/usr/bin/env python
# coding: utf-8

# # MNIST dataset

# In[134]:


from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Fetch data and split into training-validation set and test set
mnist = fetch_openml("mnist_784", version = 1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int8) # converting target to numbers instead of character
X_prevalidsplit, X_test, y_prevalidsplit, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[3]:


# Split set into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_prevalidsplit, y_prevalidsplit, random_state=42, test_size=0.2)


# In[4]:


# Standardize Xs
scaler = StandardScaler()
X_train_tr = scaler.fit_transform(X_train)
X_valid_tr = scaler.transform(X_valid)
X_test_tr = scaler.transform(X_test)

# convert targets to one-hot vertors
y_train = tf.keras.utils.to_categorical(y_train)
y_valid = tf.keras.utils.to_categorical(y_valid)
y_test = tf.keras.utils.to_categorical(y_test)


# Define neural network model as object to use for hyperparameters tunning (N of hidden layers, N of neurons within layer, regularization, batch size)

# In[86]:


# define model generation function
def nn_model(hidden_layers, dropout_rate, batch_norm):
    # build model architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=784)) # input layer
    for l in hidden_layers:
        model.add(tf.keras.layers.Dense(l, activation="relu")) # hidden layers
        if batch_norm == True:
            model.add(tf.keras.layers.BatchNormalization) # optional batch normalization
    model.add(tf.keras.layers.Dropout(dropout_rate)) # dropout layer
    model.add(tf.keras.layers.Dense(10, activation="softmax")) # output layer
    return model

# define callback to implement early stopping in training. Also saving checkpoint at best point in training
stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# define wrapper object
dnn_classifier = KerasClassifier(
    model = nn_model, 
    loss=["categorical_crossentropy"], 
    metrics=["accuracy"],
    optimizer="Adam", 
    callbacks=[stopping_callback],
    optimizer__learning_rate = 0.001, 
    model__hidden_layers = (512,), 
    model__dropout_rate = 0.2, 
    model__batch_norm = False
)


# In[87]:


# define hyperparameters space for gridsearch
nn_hyperparameters = {
    "optimizer__learning_rate": [0.0001, 0.0003, 0.001, 0.003], 
    "model__hidden_layers": [(256,256),(256,256,256),(256,256,256,256), 
                             (512,512),(512,512,512),(512,512,512,512)],
    "model__dropout_rate": [0, 0.2], 
    "model__batch_norm": [False, True]
}

# define GridSearchCv object 
grid_search_cv = GridSearchCV(dnn_classifier, nn_hyperparameters, cv = 3, n_jobs=4)


# In[88]:


# Run grid search
grid_search_cv.fit(
    X_train_tr, 
    y_train,
    epochs = 50,
    validation_data = (X_valid_tr, y_valid), 
    callbacks = [stopping_callback]
)


# In[91]:


# See best parameters
grid_search_cv.best_params_


# In[101]:


# Refit model with best parameters
best_model = nn_model(hidden_layers=(512,512), dropout_rate=0.2, batch_norm=False)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=["categorical_crossentropy"], metrics=["accuracy"])
history = best_model.fit(    
    X_train_tr, 
    y_train,
    epochs = 50,
    validation_data = (X_valid_tr, y_valid), 
    callbacks = [stopping_callback])


# In[97]:


# save fitted model to a file 
best_model.save("models/dnn.h5")


# In[ ]:


# Load model if neccessary
#tf.keras.models.load_model("models/dnn.h5")


# In[144]:


# plot learning curves for metric and loss
fig, axes = plt.subplots(1, 2, figsize=(16,6))
axes[0].plot(history.history["accuracy"], color = "orange")
axes[0].plot(history.history["val_accuracy"])
axes[1].plot(history.history["loss"], color = "orange")
axes[1].plot(history.history["val_loss"])
axes[0].set_title('Accuracy')
axes[1].set_title('Loss')
plt.show()


# In[135]:


# create confusion matrix
predictions = best_model.predict(X_valid_tr)
conf_matrix = tf.math.confusion_matrix(labels=tf.argmax(y_valid, axis=1), predictions=tf.argmax(predictions, axis=1))


# In[149]:


# plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, linecolor="white", linewidths=2)


# In[143]:


# evaluate model on test set
test_results = best_model.evaluate(X_test_tr, y_test, verbose=0)

print("Testing accuracy of model is {} and loss is {}".format(test_results[1], test_results[0]))

