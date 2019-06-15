import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from init import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
init('NN2')

def create_model():
    model=Sequential()
    model.add(GaussianNoise(0.2))
    model.add(BatchNormalization())
    model.add(Dense(180,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization())
    model.add(Dense(140,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.15))
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
    model.compile(optimizer='adam',loss='mape',metrics=['accuracy'])
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
    model.compile(optimizer='RMSprop',loss='mape',metrics=['accuracy'])
    return model

sz=[15000]
model=[]
for i in sz:
    model.append(KerasRegressor(build_fn=create_model,epochs=500,batch_size=i,verbose=0))

run(models=model,yid=2,track=2,blending=False,max_features=200,standardize=True,normalization=False,gen=True)

