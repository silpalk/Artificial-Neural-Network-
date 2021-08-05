# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:24:26 2021

@author: asilp
"""


import pandas as pd
import numpy as np
concrete=pd.read_csv("C:/Users/asilp/Desktop/AI/ANN_assign/concrete.csv")
##exploiratory data analysis
concrete.dtypes
concrete.columns
concrete.isnull().sum()

##scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
concrete['cement']=scaler.fit_transform(concrete[['cement']])
concrete['slag']=scaler.fit_transform(concrete[['slag']])
concrete['ash']=scaler.fit_transform(concrete[['ash']])
concrete['water']=scaler.fit_transform(concrete[['water']])
concrete['superplastic']=scaler.fit_transform(concrete[['superplastic']])
concrete['coarseagg']=scaler.fit_transform(concrete[['coarseagg']])
concrete['fineagg']=scaler.fit_transform(concrete[['fineagg']])
concrete['age']=scaler.fit_transform(concrete[['age']])
concrete['strength']=scaler.fit_transform(concrete[['strength']])


concrete.describe()


x=concrete.drop(['strength'],axis=1)
y=concrete.iloc[:,[8]]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

##model building
import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(8,input_shape=(8,),activation='relu'),
    keras.layers.Dense(3,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])    
    
#model evaluation
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
##training of model
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=250)
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



