from init import *
init('SVR2')

from sklearn.svm import SVR

Ca=[10,25,100]
eps=[0.0001,0.001,0.01]
model=[]
for c in Ca:
    for e in eps:
        model.append(SVR(C=c,epsilon=e))
run(models=model,yid=2,validation=3,cross_val=True,max_features=500)
