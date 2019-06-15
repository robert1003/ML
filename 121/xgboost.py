from init import *
init('xgboost')

from xgboost import XGBRegressor
larr=[0.1]
darr=[4,6,8]
narr=[250]
sarr=[0.75]
#barr=['gbtree','dart']
#tarr=['approx','hist']
model=[]
for l in larr:
    for d in darr:
        for n in narr:
            for s in sarr:
                model.append(XGBRegressor(learning_rate=l,max_depth=d,n_estimators=n,n_jobs=-1,subsample=s))
run(models=model,max_features=100,track=1,validation=5,cross_val=True)
