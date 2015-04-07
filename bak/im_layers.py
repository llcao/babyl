################################################################################################################
# layer definition for 4d tensor image (nbatch x color x nw x nh) preprocessing
#
#
# Connecting with 2d matrix layer in mlp_layer.py, we define FlattenLayer
################################################################################################################

__author__ = 'llcao'
__version__ = '0.7'

"""
v0.7 by llcao
    - NIN2DLayer
v0.6 by llcao
    - change output from a symbolic var to a function
            output(self, *args, **kwargs)
v0.5 by llcao
    - separate layers to mlplayers.py
v0.42 llcao
    - add another T.swith to avoid 0*log(0.0)
v0.41 llcao
    - rewrite cross-entropy, use T.switch to avoid log(0)->nan
v0.4 llcao
    interface to choose nonlinear fun:sigmoid/tanh/rectify
v0.31 llcao
    - in BinaryLogisitcRegression: use np.array[0.0] instead of 0.0
    self.b = theano.shared(value = np.array([0.0]), name='b')
    adjust W, and do flatten for output
v0.3 llcao
    - OutputLayer_LogisticRegression
v0.2 llcao
    - Pool2DLayer
    - Conv2DLayer
v0.1: llcao
    three layers: binary LogisticRegression, HiddenLayer, LeNetConvPoolLayer
       -- remove rng from Hidden layer
       -- add Input2DLayer and FlattenLayer
   to connect layers easily, every layer should have the following member var/funcs
    - get_output_shape()
    - params
    - output (or output()?)
    - inputlayer
   output layer or no outlayer at all
    - cost function
    - error function

"""

'''
LL questions
    - why conv('full')->sampling->conv('same') lead to weired results?
    - ?? self.b.dimshuffle('x', 0, 'x', 'x')
'''

#import numpy as np
#import theano
#from theano import tensor as T




#from mlp_layers import AbstractLayer
#rng = np.random.RandomState(23455)  #:rng: np.random.RandomState

from mlp_layers import *

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv




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

################################################################################################
#### NetworkInNetwork:
#       proposed by Min Lin, Qiang Chen, Shuicheng Yan
#       funmentally is equivalent to 1x1 convolution
################################################################################################
class NIN2DLayer(AbstractLayer):
    def __init__(self, input_layer, n_outputchannel, activation=rectify):
        self.n_outputchannel = n_outputchannel
        self.activation = activation
        self.input_layer = input_layer

        ## initialize parameters
        input_shape = self.input_layer.get_output_shape()
        if 0:
            W0 = np.random.randn(input_shape[1], self.n_outputchannel)#* self.weights_std)
        else:
            fan_in = np.prod(input_shape[1:])
            fan_out = (input_shape[0] * np.prod(input_shape[2:]) /4.0)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W0 = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=[input_shape[1], self.n_outputchannel]),
                dtype=theano.config.floatX)

        self.W = theano.shared(value=W0.astype(theano.config.floatX),  borrow=True)
        b0 = np.zeros(self.n_outputchannel).astype(theano.config.floatX)  #* self.init_bias_value)
        self.b = theano.shared(value=b0,  borrow=True)

        self.params = [self.W, self.b]
        self._desc = 'NIN_%s_%d '% (self.activation, n_outputchannel)

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return (input_shape[0], self.n_outputchannel, input_shape[2], input_shape[3])

    def output(self, *args, **kwargs): # use the 'dropout_active' keyword argument to disable it at test time. It is on by default.
        input = self.input_layer.output(*args, **kwargs)
        prod = T.tensordot(input, self.W, [[1], [0]]) # this has shape (batch_size, width, height, out_maps)
        prod = prod.dimshuffle(0, 3, 1, 2) # move the feature maps to the 1st axis, where they were in the input
        if self.activation == None:
            return prod + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            return self.activation(prod + self.b.dimshuffle('x', 0, 'x', 'x'))
