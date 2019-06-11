from init import *
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
init('NN')

def create_model():
    model=Sequential()
    model.add(Dense(200,input_dim=200,activation='relu'))
    model.add(Dense(175,activation='relu'))
    model.add(Dense(150,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(125,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(75,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='relu'))
#activation:elu,selu,relu,tanh,linear,softmax
    model.compile(optimizer='RMSprop',loss='mape',metrics=['accuracy'])
#optimizer:adam,sgd,RMSprop,Nadam
#loss:mse,mae,mape,hinge
    return model

sz=[10,30,100,250,1000]
model=[]
for i in sz:
    model.append(KerasRegressor(build_fn=create_model,nb_epoch=10000,batch_size=i,verbose=0))
run(models=model,yid=0,track=2,max_features=200,standardize=True,gen=True)

