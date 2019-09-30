import numpy as np

class InputLayer(): # planned feature is to make this layer communicate with SQL directly and handle drawing samples from data source
    
    def standardize(self): # subtract mean and divide by standard deviation, save parameters for reconstructing raw scores again
        self.dataMean=np.mean(self.data,axis=0)
        self.dataStd=np.std(self.data,axis=0)

        self.targetMean=np.mean(self.target,axis=0)
        self.targetStd=np.std(self.target,axis=0)

        self.data=(self.data-self.dataMean)/self.dataStd
        self.target=(self.target-self.targetMean)/self.targetStd

    
    def init_mysql(self,**kwargs): #open SQL connection, find row IDs to build batch lists and load batches to input layer as needed
        raise Exception("MySQL interface not yet implemented")

    def init_xml(self,**kwargs): #read XML file, find row IDs to build batch lists and load batches to input layer as needed
        raise Exception("Reading from XML not yet implemented")

    def init_json(self,**kwargs): #read JSON file, find row IDs to build batch lists and load batches to input layer as needed
        raise Exception("Reading from JSON not yet implemented")

    def init_esearch(self,**kwargs): #send requests to ElasticSearch service and read response to find row IDs and build batch lists and load batches to input layer as needed
        raise Exception("ElasticSearch interface not yet implemented")

    def read_default(self,**kwargs): # pass data and target parameters
        # data=predictor array, target=criterium array, set standardize=True
        try:
            assert isinstance(kwargs['data'],np.ndarray),"Data array is not a NumPy array"
            assert isinstance(kwargs['target'],np.ndarray),"Target array is not a NumPy array"
            assert kwargs['target'].shape[0]==kwargs['data'].shape[0],"Data and target arrays don't have same number of entries"
        
        except KeyError:
            raise Exception("Data or target array not specified")
        try:
            assert kwargs['target'].shape[1]==1,"Target array has more than 1 dimension or is of (n,) dimension"
            self.target=kwargs['target']
        except IndexError:
            self.target=kwargs['target'].reshape(-1,1)
            
        try: # reshape data array to (n,1) if it is in (n,) form
            self.data=kwargs['data']
            self.data.shape[1]
        except IndexError:
            self.data=self.data.reshape(-1,1)

        try:
            if kwargs['standardize']==True:
                self.standardize()
        except KeyError:
            pass

        self.pipeline='default'
    
    def init_default(self,**kwargs): # set input arrays by calling read_default
        self.read_default(**kwargs)
        
    def replace_default(self,**kwargs): # change input arrays by calling read_default
        self.read_default(**kwargs)
        
    def __init__(self, **kwargs): # pick input source (array, SQL interface)
        switcher={'default':self.init_default,'mysql':self.init_mysql,'xml':self.init_xml,'json':self.init_json,'elastic_search':self.init_esearch}
        switcher[kwargs['datapipe']](**kwargs) # python version of switch case; initialize appropriate pipeline by using some of the functions defined above

    def __call__(self, **kwargs): # change datapipeline parameters
        switcher={'default':self.change_default,'mysql':self.init_mysql,'xml':self.init_xml,'json':self.init_json,'elastic_search':self.init_esearch}
        switcher[self.pipeline](kwargs[list(kwargs)[0]]) # python version of switch case; change data source

    def append_data(self,data): # append more rows to input layer data array; extend implementation for SQL to enable combining multiple tables
        if self.pipeline=='default':
            assert isinstance(data,np.ndarray),"Data array is not a NumPy array"
            assert data.shape[1]==self.data.shape[1], "Data to be appended doesn't have same number of features"
            self.data=np.concatenate((self.data,data),axis=0)
        else:
            raise Exception("Appending data not implemented for {}".format(self.pipeline))

    def passtarget(self): # used for neural net init; load target vector to output layer and delete it in input layer to minimize redundancy
        temp = self.target
        del self.target

        self.depth = self.data.shape[1] # also use this opportunity to set depth variable, corresponding to number of columns, which will be used by backprop algorithm       
        return temp

class NeuralNetwork():
    
    def __init__(self, **kwargs): # initialize neural network by specifying input layer(data source) and output layer(loss function and optimizer)
        self.layers=[]
        self.layers.append(InputLayer(**kwargs)) # append input layer to layers list
        try:
            self.optimizer=kwargs['optimizer']  # will throw Exception if neural net is started before specifying optimization parameters
            self.loss_func=kwargs['loss']
        except KeyError:
            pass


    def r2score(self):
        predictions=self.layers[-1].data.reshape(-1,1)
        target=self.layers[-1].target.reshape(-1,1)
        error=np.sum(np.square(predictions-target))
        return 1-((error/target.shape[0])/np.var(target))
    
    def set_optimization(self, **kwargs):
        if OutputLayer in map(type, self.layers):
            if not isinstance(self.layers[-1],OutputLayer): # if there is already an output layer but it is not the last layer then throw exception
                raise('Last layer is not output layer; Check layer types')
            else:
                try:
                    self.layer[-1].bias=np.ones([1,1])
                    self.layer[-1].optimizer=kwargs['optimizer']
                    self.layer[-1].loss_func=kwargs['loss_func']
                    self.layer[-1].activation=kwargs['activation']
                    self.layers[-1].weights=np.zeros([self.layers[-2].data.shape[1],1]) # make weights vector compatible for dot product with data matrix from previous layer
                    self.lr=kwargs['lr']
                except KeyError:
                    pass

        else:
            try:
                self.optimizer=kwargs['optimizer']
                self.loss_func=kwargs['loss_func']
            except KeyError:
                raise Exception('Optimizer or loss function not specified')
            self.layers.append(OutputLayer(self.layers[0].passtarget(),self.optimizer,self.loss_func)) #layers[0].passtarget() in inputlayer deletes instance of target vector after passing it to outputlayer
            self.layers[-1].weights=np.zeros([self.layers[-2].data.shape[1],1]) # make weights vector compatible for dot product with data matrix from previous layer


    def init_weights(self): # give random values to all weights in all layers except the input layer
        for x in range(1,len(self.layers)): #because layer[0] is InputLayer
            
            
            if x<len(self.layers)-1: # Output layer has independent weight and data init. Skip init below
                self.layers[x].set_weights_shape([self.layers[x-1].data.shape[1],self.layers[x].depth])
                self.layers[x].set_data_shape([self.layers[0].data.shape[0],self.layers[x].depth])
            try:
                if self.layers[x].activation!=np.tanh:
                    #tu sam mijenjao
                    self.layers[x].weights=np.random.randn(self.layers[x].weights.shape[0],self.layers[x].weights.shape[1])*np.sqrt(2/self.layers[x-1].depth) #.weights.shape[1])  He init for rest (ReLu and others)
                else:
                    self.layers[x].weights=np.random.randn(self.layers[x].weights.shape[0],self.layers[x].weights.shape[1])*np.sqrt(1/self.layers[x].weights.shape[1]) # Xavier init for tanh activation
            except NameError:
                self.layers[x].weights=np.random.randn(self.layers[x].weights.shape[0],self.layers[x].weights.shape[1])*np.sqrt(2/self.layers[x].shape[1])

    def feed_forward(self): # iterate through layers and call every layers feed_forward function which takes x-1 layer data and updates layer x data (applying layer x activation on matrix dot)
        for x in range(1,len(self.layers)):
            self.layers[x].feed_forward(self.layers[x-1].data)

    def generate_paths(self,layer,node): # creates a lists of paths leading from the target node to the output layer nodes
        # used to calculate upstream gradient where one node affects large number of other nodes leading to output nodes
        pattern=[0]*(len(self.layers)-layer-1)
        patterns=[tuple(pattern)]

        if pattern==[]:
            return []

        constraints=list()
        for layr in range(layer+1,len(self.layers)):
            constraints.append(self.layers[layr].depth)

        x=len(constraints)-1
        while (pattern!=constraints):
            if pattern[x]+1<constraints[x]:
                pattern[x]+=1
                

                patterns.append(tuple(pattern))
                try:
                    if pattern[x+1]==0:
                        x=x+1
                except IndexError:
                    pass
                
            else:
                pattern[x]=0
                x=x-1
                if x==-1:
                    break

        return patterns

    def calculate_propagation(self,layer,node,connection, sampleid): # calculates the effect of target node on output node

        paths=self.generate_paths(layer,node)
        pathsAug=[]
        for path in paths:
            pathsAug.append(tuple([connection,node]+list(path)))
        paths=pathsAug
        del pathsAug

        summation=0
        intercept=0
        for path in paths:
            weight_product=1
            sampleUG=self.layers[-1].upstream_gradVector[sampleid] # backpropagation chain rule consists of multiplying node data values for last node holding derivate of loss w.r.t. y-hat (upstream gradient) and
            sampleData=self.layers[layer-1].data[sampleid,connection].reshape(-1,1) # node data pointed to by the connection whose weight we are now updating in backprop
            # and multiplied by all weights between those two nodes excluding the weight we are optimizing

            weight_product=1
            for x in range(1,len(path)-1):
                
                if self.layers[layer+x-1].activation_name == 'relu':
                    weight_product*=self.layers[layer+x].weights[path[x],path[x+1]]
                    sampleData[self.layers[layer+x-1].data[sampleid,path[x+1]]<=0]=0 # if relu of a node is 0 then gradient is 0
                elif self.layers[layer+x].activation_name == 'none':
                    weight_product*=self.layers[layer+x].weights[path[x],path[x+1]]
                else:
                    raise Exception('Optimization for {} activation function not implemented'.format(self.layers[layer+x].activation_name))

            summation+=weight_product*np.dot(sampleData.T,sampleUG)
            intercept+=weight_product*sampleUG
        return summation, np.sum(intercept)
            
    def backprop(self, **kwargs):
        if self.layers[-1].optimizer=='sgd':
            sgd(**kwargs)
        else:
            raise Exception("Optimizer {} not implented".format(self.layers[-1].optimizer))

    def sgd(self, **kwargs):
        try:
            lr=kwargs['lr']
            batch=kwargs['batch']
            n_iter=kwargs['n_iter']
        except KeyError:
            raise Exception('Make sure to specify learning rate, batch size and number of iterations for stochastic gradient descent')
        assert batch<=self.layers[-1].target.shape[0], "Selected batch size is larger than population size"

        for iteration in range(n_iter):
            self.feed_forward()
            self.layers[-1].init_upstreamGrad()

            ids=np.arange(self.layers[-1].target.shape[0]) # make index array
            np.random.shuffle(ids) # and shuffle it

            n_batch = int(self.layers[-1].upstream_gradVector.shape[0]/batch)

            for batch_i in range(0,n_batch):
                sampleid=ids[batch_i*batch:(batch_i+1)*batch]
                for layer in range(1,len(self.layers)-1):
                    
                    for node in range(self.layers[layer].depth):
                        for connection in range(self.layers[layer-1].depth):
                            weight,bias=self.calculate_propagation(layer,node,connection,sampleid)
                            self.layers[layer].weights[connection,node]-=lr*weight
                            self.layers[layer].bias[0,node]-=lr*bias

                #output layer is optimized separately
                self.layers[-1].upstream_gradVector=-2*(self.layers[-1].target-self.layers[-1].data) * (1/self.layers[-1].target.shape[0])
                vals=np.dot(self.layers[-2].data[sampleid,:].T,self.layers[-1].upstream_gradVector[sampleid,:])

                self.layers[-1].bias[0,0]-=lr*np.sum(self.layers[-1].upstream_gradVector)
                
                self.layers[-1].weights-=vals*lr
                
                self.feed_forward()
                self.layers[-1].upstream_gradVector=-2*(self.layers[-1].target-self.layers[-1].data) * (1/self.layers[-1].target.shape[0])
                

    def forceCoherence(self,position): # make sure weights and data shapes are coherent in all layesrs
        if position < 0:
            position=len(self.layers)+position
            
        for x in range(1,len(self.layers)):
            self.layers[x].set_weights_shape([self.layers[x-1].data.shape[1],self.layers[x].depth])
            self.layers[x].set_data_shape([self.layers[0].data.shape[0],self.layers[x].depth])


    def addLayer(self, **kwargs): #insert a hidden layer at position+1
        try:
            position=kwargs['position']
        except:
            position=-2
        try:
            activation=kwargs['activation']
        except:
            activation='none'
        try:
            kwargs['depth']
        except NameError:
            raise Exception("Depth specified in addLayer command invocatoin")
        
        if position < 0:
            position=len(self.layers)+position

            
        tempLayers=self.layers.copy()
        tempLayers.append(self.layers[-1])
        tempLayers[(position+2):]=self.layers[position+1:]
        tempLayers[position+1]=HiddenLayer(kwargs['depth'],activation) # add arguments which hidden layer should know upon init
        self.layers=tempLayers

        self.forceCoherence(position) # update weight and data shapes now that architecture has changed


class Layer():
    def relu(x):
        if x>0:
            return x
        else:
            return 0
    relu=np.vectorize(relu)

    activation_dict = {'tanh':np.tanh, 'none':None, 'relu':relu} # activation functions for purpose of classification

    def feed_forward(self,previousLayer):
        if self.activation==None:
            self.data=np.dot(previousLayer,self.weights)+self.bias
        else:
            self.data=self.activation(np.dot(previousLayer,self.weights)+self.bias)
            
    def set_weights_shape(self, shape):
        self.weights=np.zeros(shape)
        self.gradient=np.zeros([shape[0],shape[1],3]) # set loss function gradient w.r.t. layer weights

        self.bias=np.ones([1,shape[1]])
    def set_data_shape(self, shape):
        self.data=np.zeros(shape)

    def __init__(self,activation):
        self.activation=self.activation_dict[activation]
        self.activation_name=activation

class HiddenLayer(Layer):
    
    def __init__(self,depth,activation='none'):
        super().__init__(activation)
        self.depth=depth
        self.upstream_grad=np.zeros([depth])


         
class OutputLayer(Layer):
    def msre(self, y, yh):
        return np.sum(np.square(y-yh))
    def rmsprop(self):
        raise Exception('rmsprop not implemented')


    # these guys below are used in init method to link string arguments to actual np vectorizer for loss or optimizer function
    loss_dict = {'msre':msre} # add more loss functions here

    def init_upstreamGrad(self):
        if self.activation_name=='none': # confirm output neuron has no activation function ie. has linear activation
            self.upstream_gradVector=-2*(self.target-self.data) * (1/self.target.shape[0]) # has to be divided by N

        self.gradient = np.zeros([self.weights.shape[0],self.weights.shape[1],3]) # use this opportunity to init gradient vector because this func is started in backprop after architecture has been initialized
        self.upstream_grad=np.zeros([self.depth])
    
    def __init__(self,target,optimizer,loss,activation='none'):
        super().__init__(activation)
        self.loss_func=self.loss_dict[loss]
        self.optimizer=optimizer
        self.target=target

        self.depth=1
        self.bias=np.ones([1,1])

    def feed_forward(self,previousLayer):
        super().feed_forward(previousLayer) # dot product of weights and data from previous layer
        self.loss_val = self.loss_func(self, self.data,self.target) # calculate loss function value based on that dot product

        

