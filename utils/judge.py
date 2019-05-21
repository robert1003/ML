from numpy import *
import sys
import csv
f=sys.argv[1]
ans=load('/tmp2/b07902139/data/Y_val.npz')['arr_0'].astype(float)
y=[]
w=[200.,1.,300.]
with open(f,newline='') as csvf:
    r=csv.reader(csvf)
    for i in r:
        y.append(array(i))
y=array(y).astype(float)
N=len(ans)
y-=ans
y=abs(y)
print('track 1 score =',inner(sum(y,axis=0),w)/N)
y/=ans
print('track 2 score =',sum(y)/N)
