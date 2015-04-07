__author__ = 'llcao'
__version__ = '0.8'

'''
    __author__ = 'llcao'
    __version__ = '0.71'
        - every layer has a _desc attribute
    __author__ = 'llcao'
    __version__ = '0.71'
        - HiddenLayer:
            n_in = self.input_layer.get_output_shape()[-1]
            (previously: get_output_shape()[0])
            get_output_shape()
                shape0 = list(self.input_layer.get_output_shape())
                shape0[-1] = self.n_out
                return tuple(shape0)
        - InputTensor3Layer
    __author__ = 'llcao'
    __version__ = '0.7'
        - StochasticRescaleLayer: use for CCA
    __author__ = 'llcao'
    __version__ = '0.6'
        - add DropoutLayer
    __author__ = 'llcao'
    __version__ = '0.5'
        - AbstractLayer::set_params_values(), get_params_values()
        - every layer is a subclass of AbstractLayer
    __author__ = 'llcao'
    __version__ = '0.41'
        - SumTwoLayer (input of the same size)
    __author__ = 'llcao'
    __version__ = '0.4'
        - ConcatenateLayer (sgd_model.py:all_parameters need modification too)
    __author__ = 'llcao'
    __version__ = '0.3.1'
        - mse_output
    __author__ = 'llcao'
    __version__ = '0.3'
    implement drop-out
        - srng: from theano.tensor.shared_randomstreams import RandomStreams
        - add a parameter dropout_training to  output()function
            use dropout during training when dropout_training = True
            choose dropout_training = False when doing evaluation
    __author__ = 'llcao'
    __version__ = '0.2'
        - change output from a symbolic var to a function
            output(self, *args, **kwargs):
        - remove OutputLayer, use global funcs to compute cost/err
            mcloss_negli
            binaryloss_negli
            binaryloss_hingeloss
            mc_error
            mc_binary

    __author__ = 'llcao'
    __version__ = '0.1'
    implement
        HiddenLayer, with tanh/sigmoid/rectify/None as nonlinear activition function
        SoftmaxLayer
        InputLayer
        OutputLayer (MultiClass only, with negative log-likelihood as the cost)

to do:
    1. clean random state
    2. dropout layer
'''

import theano
import theano.tensor as T
import numpy as np

############################################################
#### Abstract layer
## every layer follows the following interface
##   - params
##   - output()
##   - get_output_shape() #to be used for next layer
###########################################################
class AbstractLayer(object):
    def __init__(self):
        self.input_layer = []
        self.params = []
        self._desc = ' '

    def set_params_values(self, param_values):
        for (p,v) in zip(self.params, param_values):
            p.set_value(v)

    def get_params_values(self):
        param_values = []
        for p in self.params:
            param_values.append(p.get_value())
        return param_values



##############################################################
#### nonlinear activation function
##############################################################
def rectify(x):
    return T.maximum(x, 0.0)
def logistic(x):
    return 1/(1+T.exp(-x))
sigmoid = T.nnet.sigmoid
tanh = T.tanh

##############################################################
#### random state
##############################################################
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
rng = np.random.RandomState(23455) # random para generator


############################################################
#### Input Layer
####    doing nothing but
####        - create a "input" symbolic
####        - get_output_shape() to be used for next layer
###########################################################

class InputLayer(AbstractLayer):
    def __init__(self, feadim, input= T.matrix('input')): #input: create a "input" symbolic
        self.input_layer = []
        self.params = []
        self.nbatch = 'nan'
        self.dim = feadim
        self.input = input
        self._desc = 'input(,%d) '% self.dim

    def output(self, *args, **kwargs):
        return self.input

    def get_output_shape(self):
        return (self.nbatch, self.dim)

class InputTensor3Layer(AbstractLayer):
    def __init__(self, feadim, feadim2, input= T.tensor3('input')): #input: create a "input" symbolic
        self.input_layer = []
        self.params = []
        self.nbatch = 'nan'
        self.dim1 = feadim
        self.dim2 = feadim2
        self.input = input
        self._desc = 'input3D(,%d,%d) '% (self.dim1,self.dim2)

    def output(self, *args, **kwargs):
        return self.input

    def get_output_shape(self):
        return (self.nbatch, self.dim1, self.dim2)

############################################################
#### ConnectLayer
####    use T.concatenate(input, axis=1) to connect multi inputlayer (list of layers)
####    difference: the input is a list of layers instead of single layer
####            Note we assume input layers are all flat
############################################################
class ConcatenateLayer(AbstractLayer):
    def __init__(self, input_layers):
        self.input_layers = input_layers
        self.params = []
        self._desc = 'ConcatLayer '

    def get_output_shape(self):
        dout = 0
        for layer in self.input_layers:
            dout += layer.get_output_shape()[1]
        return (float('nan'), dout)

    def output(self, *args, **kwargs):
        inputs = [i.output(*args, **kwargs) for i in self.input_layers]
        return T.concatenate(inputs, axis=1)

class SumTwoLayer(AbstractLayer):
    def __init__(self, input_layers):
        self.input_layers = input_layers
        self.params = []
        self._desc = 'SumLayer '

    def get_output_shape(self):
        dout = self.input_layers[0].get_output_shape()[1]
        return (float('nan'), dout)

    def output(self, *args, **kwargs):
        a0 = self.input_layers[0].output(*args, **kwargs)
        a1 = self.input_layers[0].output(*args, **kwargs)
        return a0 + a1
##############################################################################
#### Hidden Layer, or Fully-connected layer
####    Weight matrix W is of shape (n_in,n_out)
####    bias vector b is of shape (n_out,).
####    output =  tanh(dot(input,W) + b)
####       where activation function is tanh or sigmoid or rectify or None
##############################################################################
class DropoutLayer(AbstractLayer):
    def __init__(self, inputlayer, dropout_rate=0.5):
        self.input_layer = inputlayer
        self.dropout_rate = dropout_rate
        self.params = []
        self._desc = 'Dropout(rate=%d) ' % self.dropout_rate

    def output(self, dropout_training = False,  *args, **kwargs):
        input = self.input_layer.output(dropout_training=dropout_training, *args, **kwargs)

        if dropout_training and (self.dropout_rate > 0):
            retain_prob = 1 - self.dropout_rate
            mask = srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
            input = input / retain_prob * mask
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights at test time.

        return input

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

class HiddenLayer(AbstractLayer):
    def __init__(self,  inputlayer, n_out, activation=T.tanh):
        self.activation = activation
        self.n_out = n_out

        self.input_layer = inputlayer
        n_in = self.input_layer.get_output_shape()[-1]


        W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [self.W, self.b]

        self._desc = 'Hidden_%s(%dx%d) ' % (self.activation, n_in, self.n_out)
    def output(self, *args, **kwargs):


        input = self.input_layer.output( *args, **kwargs)
        lin_output = T.dot(input, self.W) + self.b
        if self.activation is None:
            return lin_output
        else:
            return self.activation(lin_output)

    def regularization(self):
        return T.sum(self.W ** 2)

    def get_output_shape(self):
        # return (float('nan'), self.n_out)
        shape0 = list(self.input_layer.get_output_shape())
        shape0[-1] = self.n_out
        return tuple(shape0)

class HiddenLayerWithoutB(AbstractLayer):
    def __init__(self,  inputlayer, n_out, activation=T.tanh):
        self.activation = activation
        self.n_out = n_out
        #self.dropout_rate = dropout_rate

        self.input_layer = inputlayer
        n_in = self.input_layer.get_output_shape()[-1]


        W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        self.params = [self.W]
        self._desc = 'HiddenWithoutB_%s(%dx%d) ' % (self.activation, n_in, self.n_out)

    def output(self, *args, **kwargs):

        input = self.input_layer.output( *args, **kwargs)


        lin_output = T.dot(input, self.W)
        if self.activation is None:
            return lin_output
        else:
            return self.activation(lin_output)

    def get_output_shape(self):
        # return (float('nan'), self.n_out)
        shape0 = list(self.input_layer.get_output_shape())
        shape0[-1] = self.n_out
        return tuple(shape0)

########################################################################
#### Softmax Layer
####    used to normalize score to [0,1]
####    input shape [n, nc]
####    output shape [n, nc]
####       output = exp(input_c)/sum_c(exp(input_c))
#########################################################################

class SoftmaxLayer(AbstractLayer):
    def __init__(self,  inputlayer):
        self.input_layer = inputlayer
        self.params = []
        self._desc = 'HiddenWithoutB_%s(%dx%d) '

    def output(self, *args, **kwargs):
        return T.nnet.softmax(self.input_layer.output(*args, **kwargs))

    def get_output_shape(self):
        return self.input_layer.get_output_shape()



########################################################################
#### New Layer used for Connotical Correlation
#########################################################################

class StochasticRescaleLayer(AbstractLayer):
    def __init__(self, inputlayer):
        self.input_layer = inputlayer
        self.params = [] # the gradient cannot be computed explictly. we will compute by ourselves

        fea_dim = self.input_layer.get_output_shape()[1]

        self.delta_square = theano.shared(value=np.ones((1,1),dtype=theano.config.floatX),name='deltasquare',borrow=True,
                                    broadcastable = (True, True))
        self.m = theano.shared(value=np.zeros((fea_dim,), dtype = theano.config.floatX), name = 'mean', borrow=True)

        self._desc = 'StochasticRescaleLayer (not clear yet) '
    def get_updates(self, learning_rate = 0.01, *args, **kwargs):

        xs = self.input_layer.output(*args, **kwargs)
        if 0:
            m_new = self.m + learning_rate * T.mean(xs - self.m, axis=0)
            delta2_new = self.delta_square + learning_rate * ( T.mean(T.sum( (xs - self.m)**2, axis=1)) - self.delta_square)
        elif 0:
            m_new = (1-learning_rate) * self.m + learning_rate * T.mean(xs, axis=0)
            delta2_new = (1-learning_rate) * self.delta_square + learning_rate *  T.mean(T.sum( (xs - self.m)**2, axis=1))
        else:
            m_new = self.m
            delta2_new = self.delta_square

        updates = []
        updates.append((self.m, m_new))
        updates.append((self.delta_square, delta2_new))

        return updates

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

    def output(self, *args, **kwargs):
        in1 = self.input_layer.output(*args, **kwargs)
        return (in1 - self.m)/T.sqrt(self.delta_square)
############################################################
#### Output cost and error
####    for multi-class
####        target: 0, 1, ... nc-1 (multi-class)
####        output: [n, nc]
####    for binary-class
####        target: 0, 1
####        output: [n, 1]
####
####    cost
####    - cross entropy = neg loglikelihood
####    - binary hinge loss = max(0, 1-y*f(x))
###########################################################
def mcloss_negli(output, target):
    return -T.mean(T.log(output)[T.arange(target.shape[0]), target])

def mc_error(output, target):
    y_pred = T.argmax(output, axis=1)
    return T.mean(T.neq(y_pred, target))

def binary_error(output, target):
    y_pred = output > 0
    return T.mean(T.neq(y_pred, target))

def binaryloss_negli(output, target):
    output0 = output.flatten()
    tmp = T.switch(target, T.log(output0), T.log(1.0-output0))
    tmp2 = T.switch(T.isinf(tmp) or T.isnan(tmp), np.log(1e-30), tmp)
    return - T.mean(tmp2)
    # naive solution, but may suffer from nan/log0.0
    #   return - T.mean(self.target * T.log(self.output) + (1-self.target) * T.log(1- self.output) )
    #   return T.mean(T.nnet.binary_crossentropy(self.output, self.target))

def binaryloss_hinge(output, target):
    target2 = T.switch(target>0, target, -1)
    margin = 1-target2* output
    return T.mean(margin* (margin>0))



def mse_loss(output, target):
    #return T.mean(T.sqrt(T.sum((output-target)**2, axis=1)))
    return T.mean(T.sum((output-target)**2, axis=1))

