import sys
from numpy import *
from sklearn.preprocessing import scale,normalize
import csv

fname=None
folder='/tmp2/b07902139/'
weight=[300.,1.,200.]
score1,score2,test_score1,test_score2=0.,0.,0.,0.
x,y,y_res,x_test,y_test,x_intest,y_intest_res,y_intest=[],[],[],[],[],[],[],[]
xs,x_tests,x_intests,x_vals=[ [] for i in range(3)],[ [] for i in range(3)],[ [] for i in range(3)],[ [] for i in range(3)]
x_val,y_val,ind=[],[],[]
sz=0

def init(_fname='tmp'):
    global fname
    fname=_fname

def readdata(train_all,small,add_noise,blending):
    global x,y,x_test,x_intest,y_intest,xs,x_tests,x_intests
    if blending:
        xs[0]=load(folder+'blend/X_train0.npz')['arr_0']
        xs[1]=load(folder+'blend/X_train1.npz')['arr_0']
        xs[2]=load(folder+'blend/X_train2.npz')['arr_0']
        x_tests[0]=load(folder+'blend/X_test0.npz')['arr_0']
        x_tests[1]=load(folder+'blend/X_test1.npz')['arr_0']
        x_tests[2]=load(folder+'blend/X_test2.npz')['arr_0']
        x_intests[0]=load(folder+'blend/X_val0.npz')['arr_0']
        x_intests[1]=load(folder+'blend/X_val1.npz')['arr_0']
        x_intests[2]=load(folder+'blend/X_val2.npz')['arr_0']
        x=[0]*len(xs[0])
        x_test=[0]*len(x_tests[0])
        x_intest=[0]*len(x_intests[0])
        y=load(folder+'data/Y_train.npz')['arr_0']
        y_intest=load(folder+'data/Y_val.npz')['arr_0']
        print('finish reading data',flush=True)
        return
    elif train_all:
        if small:
            if add_noise:
                xs[0]=load(folder+'data/X_all_noise0.npz')['arr_0']
                xs[1]=load(folder+'data/X_all_noise1.npz')['arr_0']
                xs[2]=load(folder+'data/X_all_noise2.npz')['arr_0']
            else:
                xs[0]=load(folder+'data/X_all_small0.npz')['arr_0']
                xs[1]=load(folder+'data/X_all_small1.npz')['arr_0']
                xs[2]=load(folder+'data/X_all_small2.npz')['arr_0']
            x=[0]*len(xs[0])
            y=load(folder+'data/Y_all.npz')['arr_0']
        else:
            x=load(folder+'data/X_all.npz')['arr_0']
            y=load(folder+'data/Y_all.npz')['arr_0']
    else:
        if small:
            if add_noise:
                xs[0]=load(folder+'data/X_train_noise0.npz')['arr_0']
                xs[1]=load(folder+'data/X_train_noise1.npz')['arr_0']
                xs[2]=load(folder+'data/X_train_noise2.npz')['arr_0']
            else:
                xs[0]=load(folder+'data/X_train_small0.npz')['arr_0']
                xs[1]=load(folder+'data/X_train_small1.npz')['arr_0']
                xs[2]=load(folder+'data/X_train_small2.npz')['arr_0']
            x=[0]*len(xs[0])
            y=load(folder+'data/Y_train.npz')['arr_0']
        else:
            x=load(folder+'data/X_train.npz')['arr_0']
            y=load(folder+'data/Y_train.npz')['arr_0']
    if small:
        x_tests[0]=load(folder+'data/X_test_small0.npz')['arr_0']
        x_tests[1]=load(folder+'data/X_test_small1.npz')['arr_0']
        x_tests[2]=load(folder+'data/X_test_small2.npz')['arr_0']
        x_test=[0]*len(x_tests[0])
        x_intests[0]=load(folder+'data/X_val_small0.npz')['arr_0']
        x_intests[1]=load(folder+'data/X_val_small1.npz')['arr_0']
        x_intests[2]=load(folder+'data/X_val_small2.npz')['arr_0']
        x_intest=[0]*len(x_intests[0])
    else:
        x_test=load(folder+'data/X_test.npz')['arr_0']
        x_intest=load(folder+'data/X_val.npz')['arr_0']
    y_intest=load(folder+'data/Y_val.npz')['arr_0']
    print('finish reading data',flush=True)

def init_validation(validation,cross_val,max_features,blending):
    global x,y,x_val,y_val,ind,sz,x_vals
    tmp=random.permutation(len(x))
    if not blending and max_features is None:
        for i in tmp:
            x_val.append(x[i])
            y_val.append(y[i])
    else:
        for i in range(3):
            for j in tmp:
                x_vals[i].append(xs[i][j])
        for i in tmp:
            y_val.append(y[i])
    y_val=transpose(y_val)
    sz=len(x)//validation
    if cross_val:
        for i in range(0,len(x),sz):
            ind.append((i,min(i+sz,len(x))))
    else:
        ind.append((0,sz))
        ind.append((sz,len(x)))

def init_feature(max_features,small):
    global x,x_test,x_intest,xs,x_tests,x_intests
    features=[]
    if small:
        for i in range(3):
            xs[i]=list(map(list,xs[i]))
            for j in range(len(xs[i])):
                xs[i][j]=xs[i][j][:max_features]
            xs[i]=array(xs[i])
            x_tests[i]=list(map(list,x_tests[i]))
            for j in range(len(x_tests[i])):
                x_tests[i][j]=x_tests[i][j][:max_features]
            x_tests[i]=array(x_tests[i])
            x_intests[i]=list(map(list,x_intests[i]))
            for j in range(len(x_intests[i])):
                x_intests[i][j]=x_intests[i][j][:max_features]
            x_intests[i]=array(x_intests[i])
    else:
        with open('/tmp2/b07902139/data/feature.csv',newline='') as cf:
            features=list(csv.reader(cf))
        for i in features:
            for j in range(len(i)):
                i[j]=float(i[j])
        for i in range(3):
            ind=[i for i in range(len(x[0]))]
            ind.sort(reverse=True,key=lambda x:features[x][i])
            ind=ind[:max_features]
            for arr in x:
                tmp=[]
                for j in ind:
                    tmp.append(arr[j])
                xs[i].append(tmp)
            for arr in x_test:
                tmp=[]
                for j in ind:
                    tmp.append(arr[j])
                x_tests[i].append(tmp)
            for arr in x_intest:
                tmp=[]
                for j in ind:
                    tmp.append(arr[j])
                x_intests[i].append(tmp)

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
    if i==-1:
        tmp1,tmp2=0.,0.
        for j in range(3):
            res1,res2=evaluate(j,y_cur[j],ans[j])
            tmp1+=res1
            tmp2+=res2
        return tmp1,tmp2
    res=abs(array(y_cur)-array(ans))
    tmp1,tmp2=0.,0.
    tmp1=res.sum()*weight[i]
    for j in range(len(y_cur)):
        tmp2+=res[j]/ans[j]
    return tmp1,tmp2

def run(models=None,yid=None,gen=True,track=1,seed=1126,validation=None,cross_val=False,add_const=False,max_features=None,add_noise=False,train_all=False,standardize=False,normalization=False,iteration=1,blending=False,callback=False):

    if models is None:
        print('error:model undefined')
        exit(0)
    
    print('start',flush=True)
    global x,y,y_res,x_test,y_test,x_val,y_val,x_intest,y_intest_res,y_intest,ind,score1,score2,test_score1,test_score2,xs,x_tests,x_intests
    if not seed is None:
        random.seed(seed)
    small=(max_features!=None and max_features<=200)
    readdata(train_all,small,add_noise,blending)
    y_test=[[0.]*len(x_test) for i in range(3)]
    y_res=[[0.]*len(x) for i in range(3)]
    y_intest_res=[[0.]*len(x_intest) for i in range(3)]

    if not blending and not max_features is None:
        init_feature(max_features,small)

    if add_const:
        if max_features is None:
            x=concatenate((x,[[1] for i in range(len(x))]),axis=1)
            x_test=concatenate((x_test,[[1] for i in range(len(x_test))]),axis=1)
            x_intest=concatenate((x_intest,[[1] for i in range(len(x_intest))]),axis=1)
        else:
            xs[0]=concatenate((xs[0],[[1] for i in range(len(xs[0]))]),axis=1)
            xs[1]=concatenate((xs[1],[[1] for i in range(len(xs[1]))]),axis=1)
            xs[2]=concatenate((xs[2],[[1] for i in range(len(xs[2]))]),axis=1)
            x_tests[0]=concatenate((x_tests[0],[[1] for i in range(len(x_tests[0]))]),axis=1)
            x_tests[1]=concatenate((x_tests[1],[[1] for i in range(len(x_tests[1]))]),axis=1)
            x_tests[2]=concatenate((x_tests[2],[[1] for i in range(len(x_tests[2]))]),axis=1)
            x_intests[0]=concatenate((x_intests[0],[[1] for i in range(len(x_intests[0]))]),axis=1)
            x_intests[1]=concatenate((x_intests[1],[[1] for i in range(len(x_intests[1]))]),axis=1)
            x_intests[2]=concatenate((x_intests[2],[[1] for i in range(len(x_intests[2]))]),axis=1)

    if not validation is None:
        init_validation(validation,cross_val,max_features,blending)

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

    for i in range(-1,len(y)):
        if not i in yid:
            continue
        print('training y',i,flush=True)
        mn1,mn2,mn1,mn2,mnj=1e9,1e9,1e9,1e9,-1
        s1,s2,v1,v2=0.,0.,0.,0.
        
        x_cur,x_val_cur,x_test_cur,x_intest_cur=[],[],[],[]

        if not blending and max_features is None:
            x_cur,x_val_cur,x_test_cur,x_intest_cur=x,x_val,x_test,x_intest
        else:
            if i>=0:
                x_cur,x_test_cur,x_intest_cur=xs[i],x_tests[i],x_intests[i]
            else:
                x_cur=concatenate((xs[0],xs[1],xs[2]),axis=1)
                x_test_cur=concatenate((x_tests[0],x_tests[1],x_tests[2]),axis=1)
                x_intest_cur=concatenate((x_intests[0],x_intests[1],x_intests[2]),axis=1)
            if not validation is None:
                if i>=0:
                    x_val_cur=array(x_vals[i])
                else:
                    x_val_cur=array(concatenate((x_vals[0],x_vals[1],x_vals[2]),axis=1))
        x_cur,x_val_cur,x_test_cur,x_intest_cur=array(x_cur),array(x_val_cur),array(x_test_cur),array(x_intest_cur)
        
        if standardize or normalization:
            sz=[len(x_cur),len(x_val_cur),len(x_test_cur),len(x_intest_cur)]
            if validation is None:
                tmp=concatenate((x_cur,x_test_cur,x_intest_cur))
            else:
                tmp=concatenate((x_cur,x_val_cur,x_test_cur,x_intest_cur))
            if standardize:
                tmp=scale(tmp)
            else:
                tmp=normalize(tmp)
            x_cur=tmp[0:sz[0]]
            if not validation is None:
                x_val_cur=tmp[sz[0]:sz[0]+sz[1]]
            x_test_cur=tmp[sz[0]+sz[1]:sz[0]+sz[1]+sz[2]]
            x_intest_cur=tmp[sz[0]+sz[1]+sz[2]:sz[0]+sz[1]+sz[2]+sz[3]]

        for j in range(len(models)):
            print('training y',i,'model',j,flush=True)
            model=models[j]
            s1,s2,v1,v2=0.,0.,0.,0.
            
            for tt in range(iteration):
                cs1,cs2,cv1,cv2=0.,0.,0.,0.
                for k in range(max(1,len(ind))):
                    x_train,y_train=[],[]
                    x_valid,y_valid=[],[]
                    if validation is None:
                        x_train,y_train=x_cur,y[i] if i>=0 else transpose(y)
                    else:
                        for cur in range(len(ind)):
                            for pos in range(ind[cur][0],ind[cur][1]):
                                if cur!=k:
                                    x_train.append(x_val_cur[pos])
                                    y_train.append(y_val[i][pos] if i>=0 else y_val[:][pos])
                                else:
                                    x_valid.append(x_val_cur[pos])
                                    y_valid.append(y_val[i][pos] if i>=0 else y_val[:][pos])
                    x_train=array(x_train)
                    x_valid=array(x_valid)
                    y_train=array(y_train)
                    y_valid=array(y_valid)

                    model.fit(x_train,y_train)
                    if i==-1:
                        tmp=evaluate(i,transpose(model.predict(x_train)),transpose(y_train))
                    else:
                        tmp=evaluate(i,model.predict(x_train),y_train)
                    cs1+=tmp[0]
                    cs2+=tmp[1]
                    if not validation is None:
                        if i==-1:
                            tmp=evaluate(i,transpose(model.predict(x_valid)),transpose(y_valid))
                        else:
                            tmp=evaluate(i,model.predict(x_valid),y_valid)
                        cv1+=tmp[0]
                        cv2+=tmp[1]
                    if not cross_val:
                        break
                s1+=cs1
                s2+=cs2
                if not validation is None:
                    v1+=cv1
                    v2+=cv2

            if validation is None:
                s1/=len(x)
                s2/=len(x)
                if (track==1 and s1<mn1) or (track==2 and s2<mn2):
                    mn1,mn2,mnj=s1,s2,j
                    if i>=0:
                        y_test[i]=model.predict(x_test_cur).copy()
                        y_res[i]=model.predict(x_train).copy()
                        y_intest_res[i]=model.predict(x_intest_cur).copy()
                    else:
                        y_test=transpose(model.predict(x_test_cur).copy())
                        y_res=transpose(model.predict(x_train).copy())
                        y_intest_res=transpose(model.predict(x_intest_cur).copy())
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
            x_train,y_train=x_cur,y[i] if i>=0 else transpose(y)
            model=models[mnj]
            model.fit(x_train,y_train)
            if i>=0:
                mn1,mn2=evaluate(i,model.predict(x_train),y_train)
            else:
                mn1,mn2=evaluate(i,transpose(model.predict(x_train)),transpose(y_train))
            mn1/=len(x)
            mn2/=len(x)
            if i>=0:
                y_test[i]=model.predict(x_test_cur).copy()
                y_res[i]=model.predict(x_train).copy()
                y_intest_res[i]=model.predict(x_intest_cur).copy()
            else:
                y_test=transpose(model.predict(x_test_cur).copy())
                y_res=transpose(model.predict(x_train).copy())
                y_intest_res=transpose(model.predict(x_intest_cur).copy())
            
        score1+=mn1
        score2+=mn2
        ts1,ts2=evaluate(i,y_intest_res[i] if i>=0 else y_intest_res,y_intest[i] if i>=0 else y_intest)
        ts1/=len(x_intest)
        ts2/=len(x_intest)
        test_score1+=ts1
        test_score2+=ts2

        print('done y',i,',best model =',mnj,',score =',mn1,mn2,',test score =',ts1,ts2,flush=True)

    if gen:
        output()
    print('done','score =',score1,score2,',test score =',test_score1,test_score2,flush=True)
