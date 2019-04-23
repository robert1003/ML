import sys
from numpy import *

fname=None
folder='/tmp2/b07902139/'
weight=[200.,1.,300.]
score1,score2=0.,0.
x,y,y_res,x_test,y_test=[],[],[],[],[]
x_val,y_val,ind=[],[],[]
sz=0

def init(_fname='tmp'):
    global fname
    fname=_fname

def readdata():
    global x,y,x_test
    x=load(folder+'data/X_train.npz')['arr_0']
    y=load(folder+'data/Y_train.npz')['arr_0']
    x_test=load(folder+'data/X_test.npz')['arr_0']
    print('finish reading data',flush=True)

def init_validation(validation,cross_val):
    global x,y,x_val,y_val,ind,sz
    tmp=random.permutation(len(x))
    for i in tmp:
        x_val.append(x[i])
        y_val.append(y[i])
    y_val=transpose(y_val)
    sz=len(x)//validation
    if cross_val:
        for i in range(0,len(x),sz):
            ind.append((i,min(i+sz,len(x))))
    else:
        ind.append((0,sz))
        ind.append((sz,len(x)))

def output():
    f=open(folder+'answer/'+fname+'.csv','w')
    for i in range(len(y_test[0])):
        for j in range(len(y_test)):
            print('%.10f' % y_test[j][i],end='',file=f)
            if j==2:
                print(file=f)
            else:
                print(',',end='',file=f)
    f.flush()
    f=open(folder+'answer/'+fname+'_train.csv','w')
    for i in range(len(y[0])):
        for j in range(len(y)):
            print('%.10f' % y_res[j][i],end='',file=f)
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
    return tmp1,tmp2

def run(models=None,yid=None,gen=True,track=1,validation=None,cross_val=False):

    if models is None:
        print('error:model undefined')
        exit(0)
    
    print('start',flush=True)
    readdata()
    global x,y,y_res,x_test,y_test,x_val,y_val,ind,score1,score2
    y_test=[[0.]*len(x_test) for i in range(3)]
    y_res=[[0.]*len(x) for i in range(3)]

    if not validation is None:
        init_validation(validation,cross_val)

    y=transpose(y)
    if yid==None:
        yid=range(len(y))
    elif type(yid)==int:
        yid=[yid]
    else:
        yid=list(yid)
    if type(models)!=list:
        models=[models]

    for i in range(len(y)):
        if not i in yid:
            continue
        print('training y',i,flush=True)
        mn1,mn2,mn1,mn2,mnj=1e9,1e9,1e9,1e9,-1
        s1,s2,v1,v2=0.,0.,0.,0.
        for j in range(len(models)):
            print('training y',i,'model',j,flush=True)
            model=models[j]
            for k in range(max(1,len(ind))):
                x_train,y_train=[],[]
                x_valid,y_valid=[],[]
                if validation is None:
                    x_train,y_train=x,y[i]
                else:
                    for cur in range(len(ind)):
                        for pos in range(ind[cur][0],ind[cur][1]):
                            if cur!=k:
                                x_train.append(x_val[pos])
                                y_train.append(y_val[i][pos])
                            else:
                                x_valid.append(x_val[pos])
                                y_valid.append(y_val[i][pos])
                        
                model.fit(x_train,y_train)
                tmp=evaluate(i,model.predict(x_train),y_train)
                s1+=tmp[0]
                s2+=tmp[1]
                if not validation is None:
                    tmp=evaluate(i,model.predict(x_valid),y_valid)
                    v1+=tmp[0]
                    v2+=tmp[1]
                if not cross_val:
                    break

            if validation is None:
                s1/=len(x)
                s2/=len(x)
                if (track==1 and s1<mn1) or (track==2 and s2<mn2):
                    mn1,mn2,mnj=s1,s2,j
                    if gen:
                        y_test[i]=model.predict(x_test).copy()
                        y_res[i]=model.predict(x_train).copy()
            else:
                if cross_val:
                    s1/=len(x)*(len(ind)-1)
                    s2/=len(x)*(len(ind)-1)
                    v1/=len(x)
                    v2/=len(x)
                else:
                    sz=len(x)//validation
                    s1/=len(x)-sz
                    s2/=len(x)-sz
                    v1/=sz
                    v2/=sz
                if (track==1 and v1<mn1) or (track==2 and v2<mn2):
                    mn1,mn2,mnj=v1,v2,j
            print('done',i,j,',score =',s1,s2,end='',flush=True)
            if not validation is None:
                print(',validation score =',v1,v2,flush=True)
            else:
                print(flush=True)

        if not validation is None:
            x_train,y_train=x,y[i]
            model=models[mnj]
            model.fit(x_train,y_train)
            mn1,mn2=evaluate(i,model.predict(x_train),y_train)
            mn1/=len(x)
            mn2/=len(x)
            if gen:
                y_test[i]=model.predict(x_test).copy()
                y_res[i]=model.predict(x_train).copy()
        score1+=mn1
        score2+=mn2
                
        print('done y',i,',best model =',mnj,',score =',mn1,mn2,flush=True)
    if gen:
        output()
    print('done','score =',score1,score2,flush=True)
