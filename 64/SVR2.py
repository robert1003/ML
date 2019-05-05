from init import *
init('SVR2')

from sklearn.svm import SVR

Ca=[0.1,1,10]
eps=[0.02,0.1]
model=[]
for c in Ca:
    for e in eps:
        model.append(SVR(C=c,epsilon=e))
run(models=model,yid=2,validation=3,cross_val=True,max_features=500)
