# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:30:53 2021

@author: asilp
"""

import pandas as pd
import numpy as np
bank_data=pd.read_csv("C:/Users/asilp/Desktop/AI/ANN_assign/RPL.csv")
##exploiratory data analysis
bank_data.dtypes
bank_data.columns
bank_data.isnull().sum()
bank_data=bank_data.drop(['RowNumber', 'CustomerId', 'Surname','Geography'],axis=1)
bank_data['Gender']=pd.get_dummies(bank_data['Gender'], drop_first=True)
##scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
bank_data['Balance']=scaler.fit_transform(bank_data[['Balance']])
bank_data['EstimatedSalary']=scaler.fit_transform(bank_data[['EstimatedSalary']])

bank_data.describe()


x=bank_data.drop(['Exited'],axis=1)
y=bank_data.iloc[:,[9]]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

##model building
import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(9,input_shape=(9,),activation='relu'),
    keras.layers.Dense(4,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])    
    
#model evaluation
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
##training of model
model.fit(x_train,y_train,epochs=250)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set



