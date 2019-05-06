from init import *
init('SVR0')

from sklearn.svm import SVR

Ca=[10,30,100]
eps=[0.00001,0.0002,0.001]
model=[]
for c in Ca:
    for e in eps:
        model.append(SVR(C=c,epsilon=e))
run(models=model,yid=0,validation=3,cross_val=True,max_features=500)
