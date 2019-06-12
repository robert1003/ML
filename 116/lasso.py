from init import *
from sklearn.linear_model import Lasso
init('lasso')

aarr=[0.01,0.1,1.,5.]
model=[]
for i in aarr:
    model.append(Lasso(alpha=i,normalize=False))
    model.append(Lasso(alpha=i,normalize=True))
run(models=model,standardize=True)
