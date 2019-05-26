from init import *
init('KNN')

from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor(n_neighbors=3,weights='uniform',algorithm='kd_tree',leaf_size=500,n_jobs=-1)
run(models=model)
