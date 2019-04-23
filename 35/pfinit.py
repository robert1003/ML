import sys
from numpy import *

fname=None
folder='/tmp2/b07902139/'
weight=[200.,1.,300.]
score1,score2=0.,0.
x,y,y_train_res,x_test,y_test,feature=[],[],[],[],[],[]

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
    f=open(folder+'answer/'+fname+'_feature.csv','w')
    for i in range(10000):
        for j in range(3):
            print('%.10f' % feature[j][i],end='',file=f)
            if j==2:
                print(file=f)
            else:
                print(',',end='',file=f)
    f.flush()

def evaluate(i,y,ans):
    res=abs(array(y)-array(ans))
    tmp1,tmp2=0.,0.
    tmp1=res.sum()*weight[i]
    for j in range(len(y)):
        tmp2+=res[j]/ans[j]
    return tmp1/47500,tmp2/47500

def run(models=None,yid=None,gen=True,track=1):

    if models is None:
        print('error:model undefined')
        exit(0)
    
    print('start',flush=True)
    global x,y,y_train_res,x_test,y_test,feature,score1,score2
    x=load(folder+'data/X_train.npz')['arr_0']
    y=transpose(load(folder+'data/Y_train.npz')['arr_0'])
    x_test=load(folder+'data/X_test.npz')['arr_0']
    y_test=[[0.]*2500 for i in range(3)]
    y_train_res=[[0.]*47500 for i in range(3)]
    feature=[[0.]*10000 for i in range(3)]
    print('finish reading data',flush=True)
    
    if yid==None:
        yid=range(3)
    elif type(yid)==int:
        yid=[yid]
    else:
        yid=list(yid)
    if type(models)!=list:
        models=[models]
    for i in range(3):
        if not i in yid:
            continue
        print('training y',i,flush=True)
        mn1,mn2,mnj=1e9,1e9,-1
        for j in range(len(models)):
            print('training y',i,'model',j,flush=True)
            model=models[j]
            x_train,y_train=x,y[i]
            model.fit(x_train,y_train)
            s1,s2=evaluate(i,model.predict(x_train),y_train)
            if (track==1 and s1<mn1) or (track==2 and s2<mn2):
                mn1,mn2,mnj=s1,s2,j
                if gen:
                    y_test[i]=model.predict(x_test).copy()
                    y_train_res[i]=model.predict(x_train).copy()
                    feature[i]=model.feature_importances_
            print('done',i,j,',score =',s1,s2,flush=True)
        score1+=mn1
        score2+=mn2
        print('done y',i,',best model =',mnj,',score = ',mn1,mn2,flush=True)
    if gen:
        output()
    print('done','score =',score1,score2,flush=True)
