#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

import talos

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


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

# In[16]:


# helper function for tensorboard callback
root_logdir = os.path.join(os.curdir, "/Users/szymonmizak/machine_learning/mnist_project", "my_logs")
def get_run_logdir(): 
  run_id = time.strftime("run_%Y_%m%d-%H_%M_%S")
  return os.path.join(root_logdir, run_id)


# In[17]:


# define callback to use in model function
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True)

# define model generation function
def mnist_dense_model(x_tr, y_tr, x_val, y_val, params):
    # build model architecture
    model = Sequential()
    model.add(Input(shape=x_tr.shape[1])) 
    
    for l in params['hidden_layers']:
        model.add(tf.keras.layers.Dense(l, activation=params['activation']))
        if params['batch_normalization'] == True:
            model.add(BatchNormalization())

    model.add(Dropout(params['dropout_rate']))
    model.add(
        Dense(10, activation=params['last_activation'])
        ) 

    # compile model
    model.compile(optimizer = params['optimizer'](learning_rate = params['lr']), 
                  loss = params['losses'], 
                  metrics=['accuracy']) 
    # fit model
    history = model.fit(x_tr, y_tr, 
                        validation_data = [x_val, y_val], 
                        batch_size = params['batch_size'], 
                        callbacks = [early_stopping_cb, tf.keras.callbacks.TensorBoard(get_run_logdir())], 
                        epochs = params['epochs'],
                        verbose = 0
                        )
    return history, model

# Define hyperparameters grid
params_space = {
    'lr': [0.0001, 0.0003, 0.001, 0.003],
    'hidden_layers': [(256, 256), (256, 256, 256), (256, 256, 256, 256), 
                      (512, 512), (512, 512, 512), (512, 512, 512, 512)], 
    'shapes': ['brick'],     
    'batch_size': [8, 16, 32], 
    'epochs': [100], 
    'optimizer': [Adam], 
    'losses': ['categorical_crossentropy'], 
    'activation': ['relu'], 
    'last_activation': ['softmax'], 
    'dropout_rate': [0, .1, .2], 
    'batch_normalization': [True, False]
}


# In[18]:


# Run hyperparameters space scan using random search
scan_object = talos.Scan(X_train_tr, 
                         y_train, 
                         params_space, 
                         mnist_dense_model, 
                         "mnist", 
                         X_valid_tr, 
                         y_valid, 
                         print_params=True, 
                         fraction_limit=0.2)


# In[84]:


# Extract experiment data
exp_data = scan_object.data[['val_accuracy', 'val_loss', 'hidden_layers', 'batch_size', 
                             'lr', 'dropout_rate', 'batch_normalization']]


# In[85]:


# Inspect top 10 models
exp_data.sort_values('val_accuracy', ascending=False).head(10)


# Results indicate that best models almost always include batch normalization and use 3/4 layers of size 256/512. Larger batch sizes seems to perform better. Learning rate varies but none of top models uses lowest value (0.0001).

# In[81]:


# Inspect least 10 models
exp_data.sort_values('val_accuracy', ascending=True).head(10)


# Worst models also show consistent pattern with no batch normalization, low batch size and largest tested learning rate. 

# In[111]:


# Plot experiment results 
sns.set_style("ticks")
sns.set_palette("tab10")
fig = sns.FacetGrid(exp_data, col = 'batch_size', row='batch_normalization', height=6)
fig.map(sns.pointplot,'lr','val_accuracy')


# When batch normalization is included larger learning rates do not lead to divergence. When it is missing problem appear most significant when using small batch size. Results indicate it would be reasonalbe to investigate even larger batch sizes. However running another scan would take long time and probably simple convolutional network will work better anyway. 

# In[119]:


# Save best model to file
talos.Deploy(scan_object=scan_object, model_name='mnist_model', metric='val_accuracy')


# In[120]:


# load model from file
best_model = talos.Restore('mnist_model.zip')


# In[142]:


best_model.model.compile(optimizer=Adam(learning_rate=0.0030), 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
best_model.model.fit(X_train_tr, y_train, 
                     validation_data = [X_valid_tr, y_valid], 
                        batch_size = 32, 
                        callbacks = [early_stopping_cb, tf.keras.callbacks.TensorBoard(get_run_logdir())], 
                        epochs = 100,
                        verbose = 0
                        )


# In[159]:


# create confusion matrix
predictions = best_model.model.predict(X_test_tr)
conf_matrix = tf.math.confusion_matrix(labels=tf.argmax(y_test, axis=1), predictions=tf.argmax(predictions, axis=1))
# plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, linecolor="white", linewidths=2)


# Main type of errors are betweenp airs 5-3 and 9-4. 

# In[164]:


# evaluate model on test set
test_results = best_model.model.evaluate(X_test_tr, y_test, verbose=0)
print("Testing accuracy of model is {} and loss is {}".format(round(test_results[1], 3), round(test_results[0], 3)))

