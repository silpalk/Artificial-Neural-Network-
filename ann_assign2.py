# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:30:53 2021

@author: asilp
"""

import pandas as pd
import numpy as np
forest_data=pd.read_csv("C:/Users/asilp/Desktop/AI/ANN_assign/fireforests.csv")
##exploiratory data analysis
forest_data.dtypes
forest_data.columns
forest_data=forest_data.drop(['month','day'],axis=1)
forest_data.describe()
##scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
forest_data['FFMC']=scaler.fit_transform(forest_data[['FFMC']])
forest_data['DMC']=scaler.fit_transform(forest_data[['DMC']])
forest_data['DC']=scaler.fit_transform(forest_data[['DC']])
forest_data['ISI']=scaler.fit_transform(forest_data[['ISI']])
forest_data['temp']=scaler.fit_transform(forest_data[['temp']])
forest_data['RH']=scaler.fit_transform(forest_data[['RH']])
forest_data['wind']=scaler.fit_transform(forest_data[['wind']])
forest_data['area']=scaler.fit_transform(forest_data[['area']])
forest_data.describe()


x=forest_data.drop(['area'],axis=1)
y=forest_data.iloc[:,[8]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

##model building
import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(27,input_shape=(27,),activation='relu'),
    keras.layers.Dense(17,activation='relu'),
    keras.layers.Dense(7,activation='relu'),
    keras.layers.Dense(3,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])    
    
#model evaluation
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

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


