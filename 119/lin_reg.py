from init import *
init('lin_reg')

from numpy import *
class linear_ridge_regression:
    def __init__(self,lamb=1.):
        self.w=None
        self.lamb=lamb
    def fit(self,x,y):
        H=matmul(transpose(x),x)
        H=matmul(linalg.pinv(H+self.lamb*identity(len(H))),transpose(x))
        self.w=matmul(H,y)
    def predict(self,x):
        return matmul(x,self.w)

larr=[0,0.0001,0.001,0.01,0.1,1,10]
model=[]
for l in larr:
    model.append(linear_ridge_regression(lamb=l))
run(models=model,max_features=200,validation=5,cross_val=True)
