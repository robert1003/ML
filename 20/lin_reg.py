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

model=[]
i=0.00001
while i <= 100:
    model.append(linear_ridge_regression(lamb=i))
    model.append(linear_ridge_regression(lamb=i*5))
    i*=10

run(models=model,validation=5)
