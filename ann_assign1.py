# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:56:52 2021

@author: asilp
"""

import pandas as pd
import numpy as np
startup=pd.read_csv("C:/Users/asilp/Desktop/AI/ANN_assign/startups.csv")
startup.dtypes
startup.columns
startup.describe()
startup['State']=pd.get_dummies(startup['State'],drop_first=True)
startup.dtypes
#Scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

startup['R&D Spend']=scaler.fit_transform(startup[['R&D Spend']])
startup['Administration']=scaler.fit_transform(startup[['Administration']])
startup['Marketing Spend']=scaler.fit_transform(startup[['Marketing Spend']])
startup['Profit']=scaler.fit_transform(startup[['Profit']])



startup.head()
x=startup.drop(['Profit'],axis=1)           
y=startup.iloc[:,[4]]

# splitting of data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
##model building
import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(4,input_shape=(4,),activation='relu'),
    keras.layers.Dense(2,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
##Fitting the model 
history=model.fit(x_train,y_train,validation_data=(x_test, y_test),epochs=100)
##training the model
train_mse=model.evaluate(x_train,y_train)
test_mse=model.evaluate(x_test,y_test)
print('Train : %3f,Test :%3f'%(train_mse[0],test_mse[0]))
#Graphical representation of metrics
import matplotlib.pyplot as plt

plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()