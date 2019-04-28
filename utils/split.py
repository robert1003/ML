from numpy import *
random.seed(1126)
a=random.permutation(47500)
X=load('X_train_old.npz')['arr_0']
Y=load('Y_train_old.npz')['arr_0']
X_train,Y_train=X[a[2500:]],Y[a[2500:]]
X_val,Y_val=X[a[:2500]],Y[a[:2500]]
#print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)
savez('X_train.npz',X_train)
savez('Y_train.npz',Y_train)
savez('X_val.npz',X_val)
savez('Y_val.npz',Y_val)
