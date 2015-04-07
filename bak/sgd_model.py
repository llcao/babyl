__author__ = 'llcao'
__version__ = '0.31'

''' __author__ = 'llcao'
    __version__ = '0.31'
        - add tt_sgd_dataset
            ::sgd_inds
            ::ntr,batch_size,current_pos
            ::config_tr_batch()
            ::_next_tr_batch_inds()
            ::next_tr_batch()
    __author__ = 'llcao'
    __version__ = '0.3'
        - tt_sgd_model
            ::get_layered_param_value()
            ::set_layered_param_value(layers_params)
            ::save_params(fn_output)
            ::load_params(fn_output)

    __version__ = '0.2'
        - remove function all_parameters() to tt_sgd_model::get_all_parameters()
    __author__ = 'llcao'
    __version__ = '0.1'
        - tt_sgd_model, optimized by sgd and sgd_patience

todo:
    - save parameter/ load parameter from previosu
    - dataset: sgd or other
    -
'''

import time
import numpy as np
import math
import cPickle

import copy
import theano
from theano import tensor as T

theano.config.on_unused_input = 'ignore'

'''
from mlp_layers import *
from im_layers import *


def all_parameters(layer):
    #if not layer.input_layer:
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return layer.params
    elif isinstance(layer, ConcatenateLayer) or isinstance(layer, SumTwoLayer):
        return sum([all_parameters(i) for i in layer.input_layers], [])

    else:
        return layer.params + all_parameters(layer.input_layer)
'''


#def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum=0.9, weight_decay=0.0):
#    all_grads = [theano.grad(loss, param) for param in all_parameters]
#    updates = []
#    for param_i, grad_i in zip(all_parameters, all_grads):
#        mparam_i = theano.shared(param_i.get_value()*0.)
#        v = momentum * mparam_i - weight_decay * learning_rate * param_i  - learning_rate * grad_i
#        updates.append((mparam_i, v))
#        updates.append((param_i, param_i + v))
#    return updates

def gen_updates_sgd(loss, all_parameters, learning_rate):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    return updates

class tt_sgd_dataset:
    def __init__(self):
        self.sgd_inds = []
        self.ntr = 0
        self.xs_tr = []
        self.ys_tr = []
        self.xs_te = []
        self.ys_te = []

    def get_desc(self):
        return 'sgd_dataset Ntr=%d, random = %s' %( self.ntr, self.random_batch)

    def config_tr_batch(self, batch_size = 100, brandom=True):
        Ntr = self.ntr

        self.sgd_inds =  range(0, Ntr)
        self.random_batch = brandom
        if self.random_batch:
            np.random.shuffle(self.sgd_inds)
        self.current_pos = 0
        self.batch_size = batch_size
        self.nbatch = int(np.ceil(Ntr * 1.0 / batch_size))


    def _next_tr_batch_inds(self):
        p0 = self.current_pos
        if self.current_pos + self.batch_size < self.ntr:
            p1 = self.current_pos + self.batch_size
            self.current_pos = p1
        else:
            p1 = self.ntr
            self.current_pos = 0
            if self.random_batch:
                np.random.shuffle(self.sgd_inds)

        return self.sgd_inds[p0: p1]

    def next_tr_batch(self):
        inds_i = self._next_tr_batch_inds()

        return (self.xs_tr[inds_i], self.ys_tr[inds_i])

class tt_sgd_model:
    def __init__(self):
        self.layers = []
        self.validate_model = lambda xi, yi: ''
        self.measure_model = lambda xi, yi: ''
        self.train_model =  lambda xi, yi: ''#self.update_model()

    def get_all_parameters(self):
        all_parameters = []
        for l in range(len(self.layers)-1, -1, -1):
            all_parameters += self.layers[l].params
        return all_parameters

    def get_layers_desc(self):
        desc = ''
        for lay in self.layers:
            desc += lay._desc + '\n'
        return desc

    def get_layered_param_value(self):
        layers_params = []
        for l in self.layers:
            param_l = l.get_params_values()
            layers_params.append([l.__class__, param_l])
        return layers_params

    def set_layered_param_value(self, layers_params):
        for (l,lv) in zip(self.layers, layers_params):
            if lv[0] != l.__class__:
                raise Exceptil,on('layer param does not match: %s, %s'%(lv[0],l.__class__))
            v = lv[1]
            param_l = l.set_params_values(v)

    def save_params(self, fn_output):
        layers_params = self.get_layered_param_value()
        with open(fn_output,'wb') as fp:
            cPickle.dump(layers_params, fp)

    def load_params(self, fn_output):
        with open(fn_output,'rb') as fp:
            layers_params = cPickle.load(fp)
        self.set_layered_param_value(layers_params)

    def _unit_test(self, xs_batch, ys_batch):
        print 'before training, err=',
        print self.validate_model(xs_batch, ys_batch)
        print 'now training, cost = ',
        print self.train_model(xs_batch, ys_batch)
        print 'after training, err=',
        print self.validate_model(xs_batch, ys_batch)

def sgd_batch(sgd_model, xs_tr, ys_tr,  xs_val, ys_val, batch_size = 100, n_epochs=20):
    n_tr_batches = xs_tr.shape[0]/batch_size
    n_val_batches = xs_val.shape[0]/batch_size
    if n_tr_batches * batch_size < xs_tr.shape[0]:
        n_tr_batches += 1
    if n_val_batches * batch_size < xs_val.shape[0]:
        n_val_batches += 1

    validation_frequency = 10 #min(n_tr_batches, patience / 2)


    best_model = copy.deepcopy(sgd_model)
    best_validation_loss = np.inf

    epoch = 0

    inds_for_batch_sampling = range(0, xs_tr.shape[0])
    np.random.shuffle(inds_for_batch_sampling)

    err_list_val = []
    err_list_tr = []
    bstop = False
    while (epoch < n_epochs and bstop == False):
        epoch += + 1
        for bi in xrange(n_tr_batches):

            p0 = bi * batch_size
            p1 = min((bi + 1) * batch_size, xs_tr.shape[0])
            inds_i = inds_for_batch_sampling[p0: p1]
            xs_tr_i = xs_tr[inds_i]
            ys_tr_i = ys_tr[inds_i]


            minibatch_avg_cost = sgd_model.train_model(xs_tr_i, ys_tr_i)
            print minibatch_avg_cost,
            if math.isnan(minibatch_avg_cost):
                print '\n find nan, stop'
                bstop = True
                break

            iter = (epoch - 1) * n_tr_batches + bi
            if (iter + 1) % validation_frequency == 0:
                val_cost = []
                for jj in xrange(n_val_batches):
                    p0 = jj* batch_size
                    p1 = min((jj + 1) * batch_size, xs_val.shape[0])
                    xj_val = xs_val[p0: p1]
                    yj_val = ys_val[p0: p1]
                    val_cost.append( sgd_model.validate_model(xj_val, yj_val))

                err_list_val.append(np.mean(val_cost))

                this_validation_loss = np.mean(val_cost)
                if 1:
                    print '\n epoch %i, minibatch %i/%i, validation error %f ' % \
                        (epoch, bi + 1, n_tr_batches, this_validation_loss),

                tr_cost = []
                for jj in xrange(n_tr_batches):
                    p0 = jj* batch_size
                    p1 = min((jj + 1) * batch_size, xs_tr.shape[0])
                    xj = xs_tr[p0: p1]
                    yj = ys_tr[p0: p1]
                    tr_cost.append( sgd_model.validate_model(xj, yj))
                err_list_tr.append(np.mean(tr_cost))

                if 1:
                    print 'training error %f' % np.mean(tr_cost)

                # improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    best_model = copy.deepcopy(sgd_model)

        if 1: #randomly shuffling. but not efficient, and ??? cause error!!!
            print 'randomly shuffling...'
            np.random.shuffle(inds_for_batch_sampling)

    return best_validation_loss, best_model, err_list_val, err_list_tr

########################################################################################
##################### SGD patience #####################################################
########################################################################################

def sgd_patience(sgd_model, xs_tr, ys_tr,  xs_val, ys_val, batch_size = 100, n_epochs=100, verbose = 0):

    n_train_batches = xs_tr.shape[0] / batch_size
    n_valid_batches = xs_val.shape[0] / batch_size
    validation_frequency = n_train_batches

    # early-stopping parameters
    patience = n_epochs * n_train_batches   # look as this many iteration regardless
    patience_increase = 10 * n_train_batches  # patience = max(patience, iter + ntrain_patch* patience_increase) when a new best is found
    improvement_threshold = 1 #0.995  # a relative improvement of this much is considered significant

    best_model = None
    best_validation_loss = np.inf
    start_time = time.clock()


    inds_for_batch_sampling = range(0, xs_tr.shape[0])
    np.random.shuffle(inds_for_batch_sampling)
    err_list_val = []
    err_list_tr = []


    done_looping = False
    epoch = 0
    while not done_looping: #(epoch < n_epochs)
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            p0 = minibatch_index * batch_size
            p1 = min((minibatch_index + 1) * batch_size, xs_tr.shape[0])
            inds_i = inds_for_batch_sampling[p0: p1]
            xs_tr_i = xs_tr[inds_i]
            ys_tr_i = ys_tr[inds_i]
            minibatch_avg_cost = sgd_model.train_model(xs_tr_i, ys_tr_i)
            if verbose>0:
                print minibatch_avg_cost,
            if math.isnan(minibatch_avg_cost):
                print '\n find nan, stop'
                done_looping = True
                break

            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                val_cost = []
                for jj in xrange(n_valid_batches):
                    p0 = jj* batch_size
                    p1 = min((jj + 1) * batch_size, xs_val.shape[0])
                    xj_val = xs_val[p0: p1]
                    yj_val = ys_val[p0: p1]
                    val_cost.append( sgd_model.validate_model(xj_val, yj_val))

                err_list_val.append(np.mean(val_cost))

                this_validation_loss = np.mean(val_cost)
                if verbose>=0:
                    print '\n epoch %i, minibatch %i/%i, validation error %f ' % \
                        (epoch, minibatch_index, n_train_batches, this_validation_loss),

                tr_cost = []
                for jj in xrange(n_train_batches):
                    p0 = jj* batch_size
                    p1 = min((jj + 1) * batch_size, xs_tr.shape[0])
                    xj = xs_tr[p0: p1]
                    yj = ys_tr[p0: p1]
                    tr_cost.append( sgd_model.validate_model(xj, yj))
                err_list_tr.append(np.mean(tr_cost))
                print 'traing error %f ' % np.mean(tr_cost)


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter + patience_increase)

                    best_validation_loss = this_validation_loss
                    try:
                        best_model = copy.deepcopy(sgd_model)
                    except Exception as e:
                        print 'could not copy model:', e

            if patience <= iter:
                done_looping = True
                break


        if verbose>0: #randomly shuffling. but not efficient, and ??? cause error!!!
            print 'randomly shuffling...'
            np.random.shuffle(inds_for_batch_sampling)

    return best_validation_loss, best_model, err_list_val, err_list_tr

'''
if __name__ == '__main__':
    import os

    from mlp_layers import *
    class tmp_model(tt_sgd_model):
        def __init__(self):
            self.layers = [InputLayer(5)]
            self.layers += [HiddenLayer(self.layers[-1], n_out= 3)]
            self.layers += [HiddenLayer(self.layers[-1], n_out= 2)]

    fn_model = 'temp_param_value'
    if os.path.exists(fn_model):
        model = tmp_model()
        print 'load model'
        model.load_params(fn_model)

    else:
        model = tmp_model()
        print 'save model'
        model.save_params(fn_model)

    values = model.get_layered_param_value()
    print values
'''
