from init import *
init('SVR1')

from sklearn.svm import SVR

Ca=[10,30,100]
eps=[0.001,0.01,0.1]
model=[]
for c in Ca:
    for e in eps:
        model.append(SVR(C=c,epsilon=e))
run(models=model,yid=1,validation=3,cross_val=True,max_features=500)
