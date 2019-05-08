import sys
from numpy import *
import csv

fname=None
folder='/tmp2/b07902139/'
weight=[200.,1.,300.]
score1,score2,test_score1,test_score2=0.,0.,0.,0.
x,y,y_res,x_test,y_test,x_intest,y_intest_res,y_intest=[],[],[],[],[],[],[],[]
x_val,y_val,ind,f_ind=[],[],[],[]
sz=0

def init(_fname='tmp'):
    global fname
    fname=_fname

def readdata():
    global x,y,x_test,x_intest,y_intest
    x=load(folder+'data/X_train.npz')['arr_0']
    y=load(folder+'data/Y_train.npz')['arr_0']
    x_test=load(folder+'data/X_test.npz')['arr_0']
    x_intest=load(folder+'data/X_val.npz')['arr_0']
    y_intest=load(folder+'data/Y_val.npz')['arr_0']
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

def init_feature(max_features):
    global x,x_test,x_intest,f_ind
    features=[]
    with open('/tmp2/b07902139/data/feature.csv',newline='') as cf:
        features=list(csv.reader(cf))
    if type(max_features)==float:
        max_features=int(len(x[0])*max_features)
    for i in range(3):
        ind=[i for i in range(len(x[0]))]
        ind.sort(reverse=True,key=lambda x:features[x][i])
        f_ind.append(ind[:max_features])

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
    f=open(folder+'answer/'+fname+'_val.csv','w')
    for i in range(len(y_intest[0])):
        for j in range(len(y_intest)):
            print('%.10f' % y_intest_res[j][i],end='',file=f)
            if j==2:
                print(file=f)
            else:
                print(',',end='',file=f)
    f.flush()

def evaluate(i,y_cur,ans):
    res=abs(array(y_cur)-array(ans))
    tmp1,tmp2=0.,0.
    tmp1=res.sum()*weight[i]
    for j in range(len(y_cur)):
        tmp2+=res[j]/ans[j]
    return tmp1,tmp2

def run(models=None,yid=None,gen=True,track=1,validation=None,cross_val=False,add_const=False,max_features=None):

    if models is None:
        print('error:model undefined')
        exit(0)
    
    print('start',flush=True)
    readdata()
    global x,y,y_res,x_test,y_test,x_val,y_val,x_intest,y_intest_res,y_intest,ind,score1,score2,test_score1,test_score2,f_ind
    y_test=[[0.]*len(x_test) for i in range(3)]
    y_res=[[0.]*len(x) for i in range(3)]
    y_intest_res=[[0.]*len(x_intest) for i in range(3)]

    if not max_features is None:
        init_feature(max_features)

    if add_const:
        for i in range(len(x)):
            append(x[i],1)
        for i in range(len(x_test)):
            append(x_test[i],1)

    if not validation is None:
        init_validation(validation,cross_val)

    y=transpose(y)
    y_intest=transpose(y_intest)
    if yid==None:
        yid=range(len(y))
    elif type(yid)==int:
        yid=[yid]
    else:
        yid=list(yid)
    if type(models)!=list:
        models=[models]

    x,x_val,x_test,x_intest=array(x),array(x_val),array(x_test),array(x_intest)
    y,y_val,y_intest=array(y),array(y_val),array(y_intest)

    for i in range(len(y)):
        if not i in yid:
            continue
        print('training y',i,flush=True)
        mn1,mn2,mn1,mn2,mnj=1e9,1e9,1e9,1e9,-1
        s1,s2,v1,v2=0.,0.,0.,0.
        
        x_cur,x_val_cur,x_test_cur,x_intest_cur=[],[],[],[]
        if max_features is None:
            x_cur,x_val_cur,x_test_cur,x_intest_cur=x,x_val,x_test,x_intest
        else:
            x_cur,x_test_cur,x_intest_cur=x[:,f_ind[i]],x_test[:,f_ind[i]],x_intest[:,f_ind[i]]
            if not validation is None:
                x_val_cur=array(x_val[:,f_ind[i]])
        x_cur,x_val_cur,x_test_cur,x_intest_cur=array(x_cur),array(x_val_cur),array(x_test_cur),array(x_intest_cur)

        for j in range(len(models)):
            print('training y',i,'model',j,flush=True)
            model=models[j]

            for k in range(max(1,len(ind))):
                x_train,y_train=[],[]
                x_valid,y_valid=[],[]
                if validation is None:
                    x_train,y_train=x_cur,y[i]
                else:
                    for cur in range(len(ind)):
                        for pos in range(ind[cur][0],ind[cur][1]):
                            if cur!=k:
                                x_train.append(x_val_cur[pos])
                                y_train.append(y_val[i][pos])
                            else:
                                x_valid.append(x_val_cur[pos])
                                y_valid.append(y_val[i][pos])
                x_train=array(x_train)
                x_valid=array(x_valid)
                y_train=array(y_train)
                y_valid=array(y_valid)

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
                        y_test[i]=model.predict(x_test_cur).copy()
                        y_res[i]=model.predict(x_train).copy()
                        y_intest_res[i]=model.predict(x_intest_cur).copy()
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
            x_train,y_train=x_cur,y[i]
            model=models[mnj]
            model.fit(x_train,y_train)
            mn1,mn2=evaluate(i,model.predict(x_train),y_train)
            mn1/=len(x)
            mn2/=len(x)
            if gen:
                y_test[i]=model.predict(x_test_cur).copy()
                y_res[i]=model.predict(x_train).copy()
                y_intest_res[i]=model.predict(x_intest_cur).copy()
        score1+=mn1
        score2+=mn2
        ts1,ts2=evaluate(i,y_intest_res[i],y_intest[i])
        ts1/=len(x_intest)
        ts2/=len(x_intest)
        test_score1+=ts1
        test_score2+=ts2

        print('done y',i,',best model =',mnj,',score =',mn1,mn2,',test score =',ts1,ts2,flush=True)
    if gen:
        output()
    print('done','score =',score1,score2,',test score =',test_score1,test_score2,flush=True)
