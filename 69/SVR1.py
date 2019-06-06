from init import *
init('SVR1')

from sklearn.svm import SVR

Ca=[100,300,1000,2500,10000]
eps=[0.0001,0.001,0.01]
model=[]
for c in Ca:
    for e in eps:
        model.append(SVR(C=c,epsilon=e))
run(models=model,yid=1,validation=3,cross_val=True,max_features=500)
