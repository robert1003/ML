from pfinit import *
init('xgboost')

from xgboost import XGBRegressor
model=XGBRegressor(learning_rate=0.1,max_depth=6,n_estimators=1000,n_jobs=-1,subsample=0.75,tree_method='hist')
run(models=model)
