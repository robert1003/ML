import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from init import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
init('NN')

def create_model():
    model=Sequential()
    model.add(Dense(200,input_dim=200,activation='relu'))
    model.add(Dense(150,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
#activation:elu,selu,relu,tanh,linear,softmax
    model.compile(optimizer='RMSprop',loss='mape',metrics=['accuracy'])
#optimizer:adam,sgd,RMSprop,Nadam
#loss:mse,mae,mape,hinge
    return model

def blend_model():
    model=Sequential()
#model.add(BatchNormalization(input_shape=(13,)))
    model.add(GaussianNoise(0.08))
    model.add(BatchNormalization())
    model.add(Dense(8,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.13))
    model.add(BatchNormalization())
    model.add(Dense(7,activation='relu'))
    model.add(Dense(6,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer='RMSprop',loss='mape',metrics=['accuracy'])
    return model

sz=[100]
model=[]
for i in sz:
    model.append(KerasRegressor(build_fn=blend_model,epochs=2,batch_size=i,verbose=9))

run(models=model,yid=0,track=2,blending=True,standardize=False,normalization=False,gen=True)

