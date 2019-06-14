from init import *
init('xgboost')

from xgboost import XGBRegressor
larr=[0.1]
darr=[8]
narr=[500]
sarr=[0.9]
model=[]
for l in larr:
    for d in darr:
        for n in narr:
            for s in sarr:
                model.append(XGBRegressor(learning_rate=l,max_depth=d,n_estimators=n,subsample=s,n_jobs=-1))
#model=XGBRegressor(learning_rate=0.1,max_depth=6,n_estimators=100,n_jobs=-1,subsample=0.75)
run(models=model,track=1,validation=5,cross_val=True,max_features=200)
