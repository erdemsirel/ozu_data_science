# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd

import keras
from keras import models
from keras import layers
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score


# +
# accuracy, AUC and F-1
# f1_score(), accuracy_score(), auc(), average_precision_score()

def evaluate (y_true_train, y_pred_prob_train, y_true_test, y_pred_prob_test,  threshold=0.5):
    y_pred_train = pd.Series(np.reshape(y_pred_prob_train, (1,np.product(y_pred_prob_train.shape)))[0])
    y_pred_test = pd.Series(np.reshape(y_pred_prob_test, (1,np.product(y_pred_prob_test.shape)))[0])
    
    y_pred_train = y_pred_train.apply(lambda x: 1 if x >= threshold else 0 )
    y_pred_test = y_pred_test.apply(lambda x: 1 if x >= threshold else 0 )
    
    confusion_matrix = pd.crosstab(y_true_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'])
    print('Test Set Confusion Matrix')
    print(confusion_matrix)
    
    return pd.DataFrame.from_records([[f1_score(y_true_train, y_pred_train), 
                                       accuracy_score(y_true_train, y_pred_train), 
                                       roc_auc_score(y_true_train, y_pred_prob_train), 
                                       average_precision_score(y_true_train, y_pred_prob_train),
                                      (y_pred_train.sum() / y_pred_train.count())], 
                                      
                                      [f1_score(y_true_test, y_pred_test), 
                                       accuracy_score(y_true_test, y_pred_test), 
                                       roc_auc_score(y_true_test, y_pred_prob_test), 
                                       average_precision_score(y_true_test, y_pred_prob_test),
                                      (y_pred_test.sum() / y_pred_test.count())]], 
                                     index=['Train', 'Test'], 
                                     columns=['f1_score', 'accuracy_score', 'auc', 'average_precision_score', 'positive_ratio'])


# +
data = pd.read_csv('Call Details-Data.csv')

def prepare_data(data=data, scaler = StandardScaler()):
    data.drop('Phone Number', axis=1, inplace=True)

    data['Churn'] = data['Churn'].apply(lambda x: 1 if x else 0)

    x = data.drop('Churn', axis=1)
    y = data['Churn']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
    
    # Scale Train
    if scaler is not None:
        scaler_train = scaler
        scaler_train.fit(x_train)
        x_train_s = scaler_train.transform(x_train)
        x_train = pd.DataFrame(x_train_s, index=x_train.index, columns=x_train.columns)

        # Scale Test
        scaler_test = scaler
        scaler_test.fit(x_test)
        x_test_s = scaler_test.transform(x_test)
        x_test = pd.DataFrame(x_test_s, index=x_test.index, columns=x_test.columns)
    
    return x_train, x_test, y_train, y_test
    


# -

x_train, x_test, y_train, y_test = prepare_data(data=data, scaler=StandardScaler())

print('Positive Ratio', data.Churn.sum() / data.Churn.count())

# ---

# # Experiments

# ## Lightweight Model

# +
model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=10,  verbose=1, validation_split=0.2)

test_acc_score, test_f1_score = model.evaluate(x_test, y_test)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
model.summary()
# -

evaluate(y_train, y_train_pred, y_test, y_test_pred, threshold=0.05)

# ---

# +
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=10,  verbose=1, validation_split=0.2)

test_acc_score, test_f1_score = model.evaluate(x_test, y_test)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
model.summary()
# -

evaluate(y_train, y_train_pred, y_test, y_test_pred, threshold=0.21)

# ---

# +
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=10,  verbose=1, validation_split=0.2)

test_acc_score, test_f1_score = model.evaluate(x_test, y_test)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
model.summary()
# -

evaluate(y_train, y_train_pred, y_test, y_test_pred, threshold=0.175)

# ---

# +
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=10,  verbose=1, validation_split=0.2)

test_acc_score, test_f1_score = model.evaluate(x_test, y_test)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
model.summary()
# -

evaluate(y_train, y_train_pred, y_test, y_test_pred, threshold=0.04)

# ---

# +
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=10,  verbose=1, validation_split=0.2)

test_acc_score, test_f1_score = model.evaluate(x_test, y_test)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
model.summary()
# -

evaluate(y_train, y_train_pred, y_test, y_test_pred, threshold=0.04)
