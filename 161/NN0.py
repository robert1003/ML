import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from keras import backend as K
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
K.set_session(sess)

from init import *
from functools import partial
from keras.models import Sequential
from keras.layers import Layer,Dense,Dropout,BatchNormalization,GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import Callback,EarlyStopping
init('NN0')

class RBFLayer(Layer):
    def __init__(self,units,gamma,**kwargs):
        super(RBFLayer,self).__init__(**kwargs)
        self.units=units
        self.gamma=K.cast_to_floatx(gamma)
    def build(self,input_shape):
        self.mu=self.add_weight(name='mu',shape=(int(input_shape[1]),self.units),initializer='uniform',trainable=True)
        super(RBFLayer,self).build(input_shape)
    def call(self,inputs):
        diff=K.expand_dims(inputs)-self.mu
        l2=K.sum(K.pow(diff,2),axis=1)
        res=K.exp(-1*self.gamma*l2)
        return res
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.units)

def RBFNetwork():
    model=Sequential()
    model.add(RBFLayer(7,0.1))
    model.add(RBFLayer(5,0.1))
    model.add(RBFLayer(3,0.1))
    model.add(RBFLayer(1,0.1))
    model.compile(optimizer='adam',loss='mape')
    return model

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
#model.add(Dropout(0.15))
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
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
#model.add(GaussianNoise(0.03))
#model.add(BatchNormalization())
    model.add(Dense(7,activation='relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
#model.add(GaussianNoise(0.1))
#model.add(BatchNormalization())
#model.add(Dense(6,activation='relu'))
    model.add(Dense(5,activation='relu'))
#model.add(Dense(3,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer='adam',loss='mape')
    return model

sz=[15000]
n1=[0.18,0.19,0.195,0.2]
n2=[0.08,0.09,0.095,0.1]
n3=[0.13,0.14,0.145,0.15]
earr=[700]
cb=Callback()

for i in sz:
    for e in earr:
        tmpf=partial(create_model,0.15,0.5,0.12)
        model=(KerasRegressor(build_fn=blend_model,epochs=e,batch_size=i,verbose=1))
        run(models=model,yid=0,track=2,blending=True,add_noise=False,max_features=200,standardize=False,normalization=False,gen=True)

