from init import *
init('gd2')

class gd:
    def __init__(self,lamb=0.5,iter=100,decrease_rate=0.01):
        self.w=None
        self.lamb=lamb
        self.iter=iter
        self.decrease_rate=decrease_rate
    def ndf(self,x,y):
        tmp=matmul(x,self.w)
        res=array([0.]*len(self.w))
        for i in range(len(x)):
            res+=sign(y[i]-tmp[i])*x[i]
        return res/len(x)
    def fit(self,x,y):
        self.w=array([0.]*len(x[0]))
        lamb=self.lamb
        for i in range(self.iter):
#            print(i,lamb,average(self.w),flush=True)
            self.w+=self.ndf(x,y)*lamb
            lamb*=(1-self.decrease_rate)
    def predict(self,x):
        return matmul(x,self.w)

model=gd(lamb=0.00065,iter=20000,decrease_rate=0.00008)
run(models=model,yid=2)
