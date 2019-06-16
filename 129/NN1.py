import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf
from keras import backend as K
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
K.set_session(sess)

from init import *
from functools import partial
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
init('NN1')

def create_model(ns1,ns2,ns3):
    model=Sequential()
#model.add(GaussianNoise(ns1))
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
    model.add(GaussianNoise(0.15))
    model.add(BatchNormalization())
    model.add(Dense(10,activation='relu'))
    model.add(GaussianNoise(0.2))
    model.add(BatchNormalization())
    model.add(Dense(7,activation='relu'))
#model.add(BatchNormalization())
#model.add(GaussianNoise(0.2))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer='adam',loss='mape')
    return model

sz=[15000]
model=[]

for i in sz:
    tmpf=partial(create_model,0.0,0.195,0.045)
    model.append(KerasRegressor(build_fn=tmpf,epochs=389,batch_size=i,verbose=0))

run(models=model,yid=1,track=2,max_features=200,standardize=True,normalization=False,gen=True)

