import time
import numpy as np
import math
import cPickle

import copy
import theano
from theano import tensor as T

theano.config.on_unused_input = 'ignore'

def gen_updates_sgd(loss, all_parameters, learning_rate):
    
    all_grads = T.grad(loss, all_parameters) #all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    return updates
	
def get_update_rmsprop(loss, all_parameters, learning_rate):
    # this function is copied from Chad DeChant's implementation
    all_grads = T.grad(loss, all_parameters)
    updates = []
    for p, g in zip(all_parameters, all_grads):
        MeanSquare = theano.shared(p.get_value() * 0.)
        nextMeanSquare = 0.9 * MeanSquare + (1 - 0.9) * g ** 2
        g = g / T.sqrt(nextMeanSquare + 0.000001)
        updates.append((MeanSquare, nextMeanSquare))
        updates.append((p, p - learning_rate * g))
    return updates

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

##############################################################
#### naive sgd optimization
##############################################################
def sgd_batch(sgd_model, xs_tr, ys_tr,  xs_val, ys_val, batch_size = 500, n_epochs=5, verbose = 0):
    n_tr_batches = xs_tr.shape[0]/batch_size
    n_val_batches = xs_val.shape[0]/batch_size
    if n_tr_batches * batch_size < xs_tr.shape[0]:
        n_tr_batches += 1
    if n_val_batches * batch_size < xs_val.shape[0]:
        n_val_batches += 1

    #validation_frequency = 10 #min(n_tr_batches, patience / 2)
    validation_frequency = n_tr_batches

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
            if verbose>1:
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

        if verbose>0: #randomly shuffling. but not efficient, and ??? cause error!!!
            print 'randomly shuffling...'
            np.random.shuffle(inds_for_batch_sampling)

    return best_validation_loss, best_model, err_list_val, err_list_tr