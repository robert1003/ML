import sys
from numpy import *

fname=None
folder='/tmp2/b07902139/'
weight=[200.,1.,300.]
score1,score2=0.,0.
x,y,y_train_res,x_test,y_test=[],[],[],[],[]

def init(_fname='tmp'):
    global fname
    fname=_fname

def output():
    if len(y_test)!=3 or len(y_test[0])!=2500 or len(y_test[1])!=2500 or len(y_test[2])!=2500:
        print('err',flush=True)
        print(len(y_test),len(y_test[0]),len(y_test[1]),len(y_test[2]),flush=True)
        exit(0)
    f=open(folder+'answer/'+fname+'.csv','w')
    for i in range(2500):
        for j in range(3):
            print('%.10f' % y_test[j][i],end='',file=f)
            if j==2:
                print(file=f)
            else:
                print(',',end='',file=f)
    f.flush()
    f=open(folder+'answer/'+fname+'_train.csv','w')
    for i in range(47500):
        for j in range(3):
            print('%.10f' % y_train_res[j][i],end='',file=f)
            if j==2:
                print(file=f)
            else:
                print(',',end='',file=f)
    f.flush()

def evaluate(i,y,ans):
    global score1,score2
    res=abs(array(y)-array(ans))
    tmp1,tmp2=0.,0.
    tmp1=res.sum()*weight[i]
    for j in range(len(y)):
        tmp2+=res[j]/ans[j]
    score1+=tmp1
    score2+=tmp2
    print('done',i,'score =',tmp1/47500,tmp2/47500,flush=True)

def run(models=None,yid=None,gen=True):

    if models is None:
        print('error:model undefined')
        exit(0)
    
    print('start',flush=True)
    global x,y,y_train_res,x_test,y_test
    x=load(folder+'data/X_train.npz')['arr_0']
    y=transpose(load(folder+'data/Y_train.npz')['arr_0'])
    x_test=load(folder+'data/X_test.npz')['arr_0']
    y_test=[]
    print('finish reading data',flush=True)
    
    if yid==None:
        yid=range(3)
    elif type(yid)==int:
        yid=[yid]
    else:
        yid=list(yid)
    for i in range(3):
        if not i in yid:
            if gen:
                y_test.append([0]*len(x_test))
                y_train_res.append([0]*len(x))
            continue
        print('training',i,flush=True)
        x_train,y_train=x,y[i]
        models.fit(x_train,y_train)
        if gen:
            y_test.append(models.predict(x_test))
            y_train_res.append(models.predict(x_train))
        evaluate(i,y_train_res[-1],y_train)
    if gen:
        output()
    print('done','score =',score1/47500,score2/47500,flush=True)
