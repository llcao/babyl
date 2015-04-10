__layer_version__ =  """ layer_for_theano.py defines the following layers:
	- AbstractLayer
	- InputLayer
	- HiddenLayer
	- DropoutLayer
	- SoftmaxLayer (no parameter, but used for multi-class classification)
	
	- Input2DLayer
	- FlattenLayer
	- Conv2DLayer
	- Pool2DLayer
"""

import theano
import theano.tensor as T
import numpy as np

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

##############################################################
#### nonlinear activation function
##############################################################
def rectify(x):
    return T.maximum(x, 0.0)
def logistic(x):
    return 1/(1+T.exp(-x))
sigmoid = T.nnet.sigmoid
tanh = T.tanh

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


##############################################################
#### random state
##############################################################
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
rng = np.random.RandomState(23455) # random para generator


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
        shape0 = list(self.input_layer.get_output_shape())
        shape0[-1] = self.n_out
        return tuple(shape0)

class SoftmaxLayer(AbstractLayer):
    def __init__(self,  inputlayer):
        self.input_layer = inputlayer
        self.params = []
        self._desc = 'SoftmaxLayer` '

    def output(self, *args, **kwargs):
        return T.nnet.softmax(self.input_layer.output(*args, **kwargs))

    def get_output_shape(self):
        return self.input_layer.get_output_shape()	

############################################################
## layers used for image input as 4D tensor: (#im, #channel, width, height)
###########################################################

class Input2DLayer(AbstractLayer):
    def __init__(self, nbatch, nfeature, width, height, input = T.tensor4('input')): #
        self.input_layer = []
        self.params = []
        self.nbatch = nbatch
        self.nfeature = nfeature
        self.width = width
        self.height = height
        self.input = input

        self._desc = 'inputIm(,%dx%dx%d) '% (self.nfeature, self.width, self.height)

    def output(self, *args, **kwargs):
        return self.input

    def get_output_shape(self):
        return (self.nbatch, self.nfeature, self.width, self.height)

class FlattenLayer(AbstractLayer):
    def __init__(self, input_layer, flattendim=2):
        self.input_layer = input_layer
        self.params = []
        self.flattendim = flattendim

        self._desc = 'FlattenDim=%d '% (flattendim)

    def output(self, *args, **kwargs):
        return self.input_layer.output().flatten(self.flattendim)

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        size = int(np.prod(input_shape[self.flattendim-1:]))
        return input_shape[0:self.flattendim-1] + (size,)


class Conv2DLayer(AbstractLayer):
    def __init__(self, inputlayer, filter_shape, activation=T.tanh):
        """
        :input(theano.tensor.dtensor4) symbolic image tensor, of shape image_shape

        :filter_shape: tuple or list of length 4
            (number of filters, num input feature maps, filter height,filter width)
        """
        self.activation = activation
        self.input_layer = inputlayer
        input_shape = inputlayer.get_output_shape()

        ###
        # init W:  size as filter_shape
        # init b:  size of (filter_shape[0],)
        self.filter_shape = filter_shape
        if input_shape[1] != filter_shape[1]:
            print 'input:', input_shape
            print 'filter:', filter_shape
            raise TypeError('inputshape[1] should be equal to filter_shape[1]')

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /4.0)
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        oshape = self.get_output_shape()
        self._desc = 'Conv2D_%s(,%dx%dx%d) '% (self.activation, oshape[1],oshape[2],oshape[3])

    def output(self, *args, **kwargs):

        conv_out = conv.conv2d(input=self.input_layer.output(), filters=self.W) #filter_shape=filter_shape, image_shape=image_shape)
        if self.activation is None:
            return conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            return self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_output_shape(self):
        nbatch, nfeature, w, h = self.input_layer.get_output_shape()
        filter_size = self.filter_shape[2:]
        nfilter_out = self.filter_shape[0]
        w_out = w - filter_size[0] + 1
        h_out = h - filter_size[1] + 1
        return (nbatch, nfilter_out, w_out, h_out)

class Pool2DLayer(AbstractLayer):
    def __init__(self, inputlayer, poolsize = (2,2)):
        self.input_layer = inputlayer
        self.poolsize = poolsize
        self.params = []

        self._desc = 'Pool2D_%dx%d '% (poolsize[0],poolsize[1])
    def get_output_shape(self):
        n, nf, w0, h0 = self.input_layer.get_output_shape()
        w = w0/int(self.poolsize[0])
        if w0 % self.poolsize[0] != 0 :
            w += 1
        h = h0/self.poolsize[1]
        if h0 % self.poolsize[1] != 0:
            h += 1
        return (n, nf, w, h)
    def output(self, *args, **kwargs):
        return downsample.max_pool_2d(input=self.input_layer.output(), ds=self.poolsize)		