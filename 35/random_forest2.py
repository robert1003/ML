from pfinit import *
init('random_forest2')

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_features='sqrt',n_estimators=100,criterion='mae',n_jobs=16,oob_score=True)
run(models=model,yid=2)
