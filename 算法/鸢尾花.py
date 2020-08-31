#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 11:08 上午
# @Author  : Leoning
# @File    : 鸢尾花.py

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from sklearn.datasets import load_iris

# reproducibility
seed = 13
np.random.seed(seed)

# load data
df = load_iris()
X = df.values[:, 0:4].astype(float)
Y = df.values[:, 4]

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)


# define a network
def baseline_model():
    model = Sequential()
    model.add(Dense(7, input_dim=4, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='mean_square_error', optimizer='sgd', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model(), epochs=20, batch_size=1, verbose=1)

# evaluate
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
print("Acucuray of cross validation, mean %.2f, std %.2f" % (result.mean(), result.std()))

# save model
estimator.fit(X, Y_onehot)
model_json = estimator.model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

estimator.model.save_weight("model.h5")
print("saved model to disk")

# load model and use it for prediction
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print("loaded model from disk")

predicted = loaded_model.predict(X)
print("predicted probability:" + str(predicted))

predicted_label = loaded_model.predict_classes()
print("predicted label :" + str(predicted_label))
