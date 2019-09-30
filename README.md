# nnet-from-scratch
Barebones neural network made from scratch

This is for-practice implementation of regression multilayer perceptron.


## How to use:

Check example.py




## To-Do:
Neural net is using ordinary Stochastic Gradient Descent which makes optimization for multi-layer networks slow due to their tendency of getting stuck into local optima or pathological curves. Next iteration should focus on improving training speed.

* Add binary crossentropy loss function for classification
* Implement RMPSprop or Adam optimizer
* Write math heavy code segment in C and parallelize
