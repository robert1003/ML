import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from init import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
init('NN')

def create_model():
    model=Sequential()
    '''model.add(Dense(600,input_dim=600,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(400,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(300,activation='relu'))'''
#model.add(Dense(200,input_dim=200,activation='relu'))
#model.add(Dense(200,activation='relu'))
    model.add(Dense(150,input_dim=200,activation='relu'))
    model.add(BatchNormalization())
#model.add(Dropout(0.05))
    model.add(Dense(125,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100,activation='relu'))
    model.add(BatchNormalization())
#model.add(Dropout(0.05))
    model.add(Dense(75,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50,activation='relu'))
#model.add(Dropout(0.05))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(10,activation='relu'))
#model.add(Dropout(0.1))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
#activation:elu,selu,relu,tanh,linear,softmax
    model.compile(optimizer='RMSprop',loss='mape',metrics=['accuracy'])
#optimizer:adam,sgd,RMSprop,Nadam
#loss:mse,mae,mape,hinge
    return model

def shallow_model():
    model=Sequential()
    model.add(Dense(300,input_dim=200,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer='RMSprop',loss='mape',metrics=['accuracy'])
    return model

#sz=[10,30,100,250,1000]
sz=[100]*5
model=[]
for i in sz:
    model.append(KerasRegressor(build_fn=create_model,nb_epoch=10000,batch_size=i,verbose=0))
run(models=model,track=2,max_features=200,add_noise=True,standardize=True,normalization=False,gen=True)

