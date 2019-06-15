import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from init import *
from functools import partial
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import Callback,EarlyStopping
init('NN0')

def create_model(ns1,ns2,ns3):
    model=Sequential()
    model.add(GaussianNoise(ns1))
    model.add(BatchNormalization())
    model.add(Dense(180,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(ns2))
    model.add(BatchNormalization())
    model.add(Dense(140,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(ns3))
    model.add(BatchNormalization())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(75,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
#activation:elu,selu,relu,tanh,linear,softmax
    model.compile(optimizer='adam',loss='mape')
#optimizer:adam,sgd,RMSprop,Nadam
#loss:mse,mae,mape,hinge
    return model

def blend_model():
    model=Sequential()
    model.add(BatchNormalization(input_shape=(13,)))
    model.add(GaussianNoise(0.5))
    model.add(BatchNormalization())
    model.add(Dense(8,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.5))
    model.add(BatchNormalization())
    model.add(Dense(7,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.5))
    model.add(BatchNormalization())
    model.add(Dense(6,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.5))
    model.add(BatchNormalization())
    model.add(Dense(5,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer='RMSprop',loss='mape')
    return model

sz=[15000]
n1=[0.18,0.19,0.195,0.2]
n2=[0.08,0.09,0.095,0.1]
n3=[0.13,0.14,0.145,0.15]
model=[]
cb=Callback()

for i in sz:
    tmpf=partial(create_model,0.15,0.5,0.12)
    model.append(KerasRegressor(build_fn=tmpf,epochs=400,batch_size=i,verbose=0))

run(models=model,yid=0,track=2,blending=False,max_features=200,standardize=True,normalization=False,gen=True)

