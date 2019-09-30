from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

import barebones_nnet

boston=load_boston()

clf=LinearRegression().fit(boston.data,boston.target)
#print(clf.score(boston.data,boston.target))

nnet=barebones_nnet.NeuralNetwork(datapipe='default',data=boston.data,target=boston.target, standardize=True)
nnet.set_optimization(optimizer='sgd',loss_func='msre') # add parameter variable for rmpsprop
nnet.addLayer(depth=13,activation='relu')
nnet.addLayer(depth=6,activation='relu')

test.init_weights()

for epoch in range(1000):
    nnet.sgd(lr=0.001,batch=30,n_iter=1)
    print(nnet.r2score())
