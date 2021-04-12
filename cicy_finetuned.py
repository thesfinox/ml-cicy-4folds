#!/usr/bin/env python
# coding: utf-8

#####################################################
# AI for CICY 4-folds                               #
#                                                   #
# Authors: H. Erbin, R. Finotello                   #
# Code: R. Finotello (riccardo.finotello@gmail.com) #
#                                                   #
#####################################################

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import logging
import joblib
import json
import time
import multiprocessing
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# helper functions

@tf.keras.utils.register_keras_serializable()
class ScalarMult(keras.layers.Layer):
    
    def __init__(self, seed=None, **kwargs):
        
        super(ScalarMult, self).__init__(**kwargs)
        
        self.seed = seed
        self.k = self.add_weight(shape=(1,), initializer=keras.initializers.GlorotUniform(self.seed), trainable=True, name='kappa')
        
    def call(self, inputs):
        
        return self.k * inputs
    
    def get_config(self):
        
        config = super(ScalarMult, self).get_config()
        config.update({'seed': self.seed})
        
        return config

@tf.keras.utils.register_keras_serializable()
class SAM(keras.layers.Layer):
    
    def __init__(self, ratio=1, seed=None, **kwargs):
        
        super(SAM, self).__init__(**kwargs)
        
        self.ratio = ratio
        self.seed  = seed
        
    def build(self, input_shape):
        
        # register the input shape
        _, height, width, channel = input_shape # N x H x W x C
        new_channel = int(channel * self.ratio)
        
        # layers
        self.conv_1 = keras.layers.Conv2D(new_channel,
                                          kernel_size=1,
                                          kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                          bias_initializer=keras.initializers.Zeros()
                                         )
        self.conv_2 = keras.layers.Conv2D(new_channel,
                                          kernel_size=1,
                                          kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                          bias_initializer=keras.initializers.Zeros()
                                         )
        
        self.reshape_1 = keras.layers.Reshape((height * width, channel))
        self.reshape_2 = keras.layers.Reshape((height * width, new_channel))
        self.reshape_3 = keras.layers.Reshape((height * width, new_channel))
        self.reshape_4 = keras.layers.Reshape((height, width, channel))
        
        self.aff     = keras.layers.Dot(axes=-1)
        self.aff_act = keras.layers.Activation('softmax')
        
        self.mult = keras.layers.Dot(axes=1)
        
        self.scalar = ScalarMult()
        
        self.add = keras.layers.Add()
        
        
    def call(self, inputs):

        # first convolutions
        A = self.conv_1(inputs) # N x H x W x C'
        B = self.conv_2(inputs) # N x H x W x C'

        # reshape
        I1 = self.reshape_1(inputs) # N x D x C
        A1 = self.reshape_2(A)      # N x D x C'
        B1 = self.reshape_3(B)      # N x D x C'

        # affinity matrix
        F = self.aff([A1, B1]) # N x D x D
        F = self.aff_act(F)    # N x D x D

        # multiply by the original matrix (reshaped)
        M = self.mult([F, I1]) # N x D x C

        # reshape to original
        M = self.reshape_4(M) # N x H x W x C

        # multiplication by a scalar
        M = self.scalar(M) # N x H x W x C

        # sum with the input
        M = self.add([inputs, M]) # N x H x W x C

        return M
    
    def get_config(self):
        
        config = super(SAM, self).get_config()
        config.update({'ratio': self.ratio, 'seed': self.seed})
        
        return config

@tf.keras.utils.register_keras_serializable()
class CHAM(keras.layers.Layer):
    
    def __init__(self, ratio=1, seed=None, **kwargs):
        
        super(CHAM, self).__init__(**kwargs)
        
        self.ratio = ratio
        self.seed  = seed
        
    def build(self, input_shape):
        
        # register the input shape
        _, height, width, channel = input_shape # N x H x W x C
        new_channel = int(channel * self.ratio)
        
        # layers
        self.conv_1 = keras.layers.Conv2D(new_channel,
                                          activation='relu',
                                          kernel_size=1,
                                          kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                          bias_initializer=keras.initializers.Zeros()
                                         )
        self.conv_2 = keras.layers.Conv2D(channel,
                                          kernel_size=1,
                                          kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                          bias_initializer=keras.initializers.Zeros()
                                         )
        
        self.act = keras.layers.Activation('softmax')
        
        self.mult = keras.layers.Multiply()
        
    def call(self, inputs):

        # first convolutions
        A = self.conv_1(inputs) # N x H x W x C'

        # second convolution
        A = self.conv_2(A) # N x H x W x C

        A = self.act(A) # N x H x W x C

        # element-wise multiplication
        M = self.mult([inputs, A]) # N x H x W x C

        return M
    
    def get_config(self):
        
        config = super(CHAM, self).get_config()
        config.update({'ratio': self.ratio, 'seed': self.seed})
        
        return config

@tf.keras.utils.register_keras_serializable()
class AM(keras.layers.Layer):
    
    def __init__(self, sam=1, cham=1, seed=None, **kwargs):
        
        super(AM, self).__init__(**kwargs)
        
        self.sam  = sam
        self.cham = cham
        self.seed = seed
        
    def build(self, input_shape):
        
        # layers
        self.sam  = SAM(ratio=self.sam, seed=self.seed)
        self.cham = CHAM(ratio=self.cham, seed=self.seed)
        
    def call(self, inputs):
        
        x = self.sam(inputs)
        x = self.cham(x)
        
        return x
    
    def get_config(self):
        
        config = super(AM, self).get_config()
        config.update({'sam': self.sam, 'cham': self.cham, 'seed': self.seed})
        
        return config

@tf.keras.utils.register_keras_serializable()
class FeatureMap(keras.layers.Layer):
    
    def __init__(self, filters, alpha=0.0, seed=None, **kwargs):
        
        super(FeatureMap, self).__init__(**kwargs)
        
        self.filters = filters
        self.alpha   = alpha
        self.seed    = seed
        
    def build(self, input_shape):
        
        # register the input shape
        _, height, width, _ = input_shape
        
        # layers
        self.fm = keras.layers.Conv2D(self.filters,
                                      kernel_size=(height, width),
                                      padding='valid',
                                      kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                      bias_initializer=keras.initializers.Zeros(),
                                     )
        
        self.act  = keras.layers.LeakyReLU(self.alpha)
        
    def call(self, inputs):
        
        x = self.fm(inputs)
        x = self.act(x)
        
        # get the new shape
        _, H, W, C = x.shape
        
        x = keras.layers.Reshape((H * W * C,))(x)
        
        return x
    
    def get_config(self):
        
        config = super(FeatureMap, self).get_config()
        config.update({'filters': self.filters, 'alpha': self.alpha, 'seed': self.seed})
        
        return config

@tf.keras.utils.register_keras_serializable()
class DenseActivation(keras.layers.Layer):
    
    def __init__(self, units, alpha=0.0, l1_reg=0.0, l2_reg=0.0, norm='bn', dropout=0.0, seed=None, **kwargs):
        
        super(DenseActivation, self).__init__(**kwargs)
        
        self.units   = units
        self.alpha   = alpha
        self.l1_reg  = l1_reg
        self.l2_reg  = l2_reg
        self.norm    = norm
        self.dropout = dropout
        self.seed    = seed
        
    def build(self, input_shape):
        
        # layers
        self.dense = keras.layers.Dense(self.units,
                                        kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                        bias_initializer=keras.initializers.Zeros(),
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                                       )
        self.act  = keras.layers.LeakyReLU(self.alpha)
        
        # normalisation layer
        self.norm_layer = None
        
        if self.norm == 'bn':
            
            self.norm_layer = keras.layers.BatchNormalization()
            
        if self.norm == 'layer':
            
            self.norm_layer = keras.layers.LayerNormalization()
            
        # dropout
        self.drop_layer = None
        
        if self.dropout:
            
            self.drop_layer = keras.layers.Dropout(rate=self.dropout, seed=self.seed)
        
    def call(self, inputs):
        
        x = self.dense(inputs)
        x = self.act(x)
        
        if self.norm_layer is not None:
            x = self.norm_layer(x)
            
        if self.drop_layer is not None:
            x = self.drop_layer(x)
        
        return x
    
    def get_config(self):
        
        config = super(DenseActivation, self).get_config()
        config.update({'units': self.units,
                       'alpha': self.alpha,
                       'l1_reg': self.l1_reg,
                       'l2_reg': self.l2_reg,
                       'seed': self.seed
                      }
                     )
        
        return config

@tf.keras.utils.register_keras_serializable()
class InceptionModule(keras.layers.Layer):
    
    def __init__(self,
                 filters,
                 reduce=False,
                 single=False,
                 kernels=[(3, 3), (5, 5)], # use 'H' or 'W' for 1D kernels
                 padding='valid',
                 pool='max',               # None, 'max', 'avg'
                 pool_size=(2, 2),
                 alpha=0.0,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 norm='bn',                # 'bn' or 'layer'
                 dropout=0.0,
                 seed=None,
                 **kwargs
                ):
        
        super(InceptionModule, self).__init__(**kwargs)
        
        self.filters   = filters
        self.reduce    = reduce
        self.single    = single
        self.kernels   = kernels
        self.padding   = padding
        self.pool      = pool
        self.pool_size = pool_size
        self.alpha     = alpha
        self.l1_reg    = l1_reg
        self.l2_reg    = l2_reg
        self.norm      = norm
        self.dropout   = dropout
        self.seed      = seed
        
    def build(self, input_shape):
        
        # register the input shape
        _, height, width, channel = input_shape
        
        # reduction layers
        self.reduction_layers = []
        
        if self.reduce:
            
            for _ in self.kernels:

                k = keras.layers.Conv2D(self.filters,
                                        kernel_size=(1, 1),
                                        padding='valid',
                                        kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                        bias_initializer=keras.initializers.Zeros()
                                       )

                self.reduction_layers.append(k)
        
        # inception layers
        self.inception_kernels = []
        
        for kernel in self.kernels:
            
            if kernel == 'W':
                
                k = keras.layers.Conv2D(self.filters,
                                        kernel_size=(1, width),
                                        padding='same',
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                                        kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                        bias_initializer=keras.initializers.Zeros()
                                       )
            elif kernel == 'H':
                
                k = keras.layers.Conv2D(self.filters,
                                        kernel_size=(height, 1),
                                        padding='same',
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                                        kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                        bias_initializer=keras.initializers.Zeros()
                                       )
                
            else:
                
                k = keras.layers.Conv2D(self.filters,
                                        kernel_size=kernel,
                                        padding=self.padding,
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                                        kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                        bias_initializer=keras.initializers.Zeros()
                                       )
                
            self.inception_kernels.append(k)
            
        # pooling
        if self.pool is not None:
            
            if self.reduce:

                k = keras.layers.Conv2D(self.filters,
                                        kernel_size=(1, 1),
                                        padding='valid',
                                        kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                        bias_initializer=keras.initializers.Zeros()
                                       )

                self.reduction_layers.append(k)
                
            if self.pool == 'avg':
                
                k = keras.layers.AveragePooling2D(pool_size=self.pool_size, padding=self.padding)
                
                self.inception_kernels.append(k)
                
            else:
                
                k = keras.layers.MaxPool2D(pool_size=self.pool_size, padding=self.padding)
                
                self.inception_kernels.append(k)
                
        # standalone 1 x 1 convolution
        self.standalone = None
        
        if self.single:
            
            self.standalone = keras.layers.Conv2D(self.filters,
                                                  kernel_size=(1, 1),
                                                  padding='valid',
                                                  kernel_initializer=keras.initializers.GlorotUniform(self.seed),
                                                  bias_initializer=keras.initializers.Zeros()
                                                 )
        
        # assertion
        if self.reduce:
            
            assert len(self.inception_kernels) == len(self.reduction_layers), 'Lengths of reduction and inception layers differ!'
            
        # concatenation layer
        self.concat = keras.layers.Concatenate()
        
        # normalisation layer
        self.norm_layer = None
        
        if self.norm == 'bn':
            
            self.norm_layer = keras.layers.BatchNormalization()
            
        if self.norm == 'layer':
            
            self.norm_layer = keras.layers.LayerNormalization()
            
        # dropout
        self.drop_layer = None
        
        if self.dropout:
            
            self.drop_layer = keras.layers.Dropout(rate=self.dropout, seed=self.seed)
                
    def call(self, inputs):
        
        layers = []
        
        # add each inception kernel (with the 1 x 1 convolutions if needed)
        for n, kernel in enumerate(self.inception_kernels):
            
            x = inputs
            
            if self.reduce:
                
                x = self.reduction_layers[n](x)
                x = keras.layers.LeakyReLU(self.alpha)(x)
                
            x = kernel(x)
            x = keras.layers.LeakyReLU(self.alpha)(x)
            layers.append(x)
        
        # add the standalone 1 x 1 convolution if needed
        if self.standalone is not None:
            
            x = self.standalone(inputs)
            x = keras.layers.LeakyReLU(self.alpha)(x)
            layers.append(x)
            
        # concatenate the layers
        if len(layers) > 0:
            
            M = self.concat(layers)
            
        # normalisation
        if self.norm_layer is not None:
            
            M = self.norm_layer(M)
            M = keras.layers.LeakyReLU(self.alpha)(M)
            
        # dropout
        if self.drop_layer is not None:
            
            M = self.drop_layer(M)
            
        return M
    
    def get_config(self):
        
        config = super(InceptionModule, self).get_config()
        config.update({'filters': self.filters,
                       'reduce': self.reduce,
                       'single': self.single,
                       'kernels': self.kernels,
                       'padding': self.padding,
                       'pool': self.pool,
                       'pool_size': self.pool_size,
                       'alpha': self.alpha,
                       'l1_reg': self.l1_reg,
                       'l2_reg': self.l2_reg,
                       'norm': self.norm,
                       'dropout': self.dropout,
                       'seed': self.seed
                      }
                     )
        
        return config


def ImbalancedMSE(beta):

    def loss(y_true, y_pred):

        # compute weights
        values = tf.cast(y_true, tf.int32)
        bins   = tf.math.bincount(values)
        counts = tf.gather(bins, values)
        counts = tf.cast(counts, tf.float32)

        weights = (1.0 - beta) / (1.0 - tf.math.pow(beta, counts))

        # compute the loss
        squares = tf.math.square(y_true - y_pred)
        squares = tf.math.multiply(weights, squares)

        return tf.math.reduce_mean(squares)

    return loss

def ImbalancedHuber(beta, delta):

    def loss(y_true, y_pred):

        # compute weights
        values = tf.cast(y_true, tf.int32)
        bins   = tf.math.bincount(values)
        counts = tf.gather(bins, values)
        counts = tf.cast(counts, tf.float32)

        weights = (1.0 - beta) / (1.0 - tf.math.pow(beta, counts))

        # compute loss
        error = tf.math.abs(y_true - y_pred)
        less_than = tf.cast(error <= delta, tf.float32)
        grt_than  = tf.cast(error > delta, tf.float32)

        loss_1 = less_than * (0.5 * tf.math.square(error))
        loss_2 = grt_than  * (0.5 * (delta ** 2) + delta * (error - delta))
        loss   = loss_1 + loss_2

        return tf.math.multiply(weights, loss)

    return loss
    
class PrintCheckpoint(keras.callbacks.Callback):
    '''
    Print status of the training after a given number of epochs.
    '''
    
    def __init__(self, interval, log):
        '''
        Required arguments:
            interval: print status after given interval of epochs,
            log:      Python logger.
        '''
        super(PrintCheckpoint, self).__init__()
        self.interval    = int(interval)
        self.log         =log
        self.epoch_times = []
        
    def on_train_begin(self, logs=None):
        if self.interval > 1:
            self.log.debug(f'Training has started. Callouts will be printed every {self.interval:d} epochs.')
        else:
            self.log.debug(f'Training has started. Callouts will be printed every epoch.')
                
        self.train_time = time.time()
        
    def on_train_end(self, logs=None):
        self.train_time = time.time() - self.train_time
        self.train_time = time.gmtime(self.train_time)
        self.train_time = time.strftime('%H hours, %M minutes, %S seconds', self.train_time)
        now = time.strftime('%d/%m/%Y at %H:%M:%S', time.localtime())
        self.log.debug(f'Training ended on {now} after {self.train_time}.')
        
    def on_epoch_begin(self, epoch, logs=None):
        
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            self.epoch_time = time.time()
            now = time.strftime('%d/%m/%Y at %H:%M:%S', time.localtime())
            self.log.debug(f'Training epoch {epoch+1:d}. Started on {now}.')
        
    def on_epoch_end(self, epoch, logs=None):
        
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            self.epoch_time = time.time() - self.epoch_time
            self.epoch_times.append(self.epoch_time)
            epoch_time = time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(np.mean(self.epoch_times)))
            self.log.debug(f'Average epoch training time: {epoch_time}\n')
            for key, value in logs.items():
                self.log.debug(f'{key} = {value:.6f}')

if __name__ == '__main__':

    # set argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-xtn', '--xtrain', help='path or URL to the training features')
    parser.add_argument('-xvd', '--xvalid', help='path or URL to the validation features')
    parser.add_argument('-xtt', '--xtest', help='path or URL to the test features')
    parser.add_argument('-ytn', '--ytrain', help='path or URL to the training labels')
    parser.add_argument('-yvd', '--yvalid', help='path or URL to the validation labels')
    parser.add_argument('-ytt', '--ytest', help='path or URL to the test labels')
    parser.add_argument('-hyp', '--hyperparameters', type=int, default=0, help='position on the hyperparameter list')
    parser.add_argument('-p', '--params', help='path to a list of hyperparameters')
    parser.add_argument('-r', '--random', default=123, type=int, help='random seed')
    parser.add_argument('session', help='name of the session')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    
    args = parser.parse_args()
    
    # set matplotlib backend to batch use
    sns.set_theme(context='paper', palette='tab10')
    mpl.use('agg')
    
    # create relevant directory structure    
    img_dir = os.path.join(args.session, 'img')
    mod_dir = os.path.join(args.session, 'mod')
    log_dir = os.path.join(args.session, 'log')
    
    os.makedirs(img_dir, exist_ok=True) # image directory
    os.makedirs(mod_dir, exist_ok=True) # models directory
    os.makedirs(log_dir, exist_ok=True) # log directory
    
    # create log file
    level = logging.ERROR
    if args.verbose == 1:
        level = logging.INFO
    if args.verbose > 1:
        level = logging.DEBUG
    
    log = logging.getLogger('cicy4_finetuned_' + args.session)
    log.setLevel(level)
    
    logfile = os.path.join(log_dir, f'model_finetuned_{args.hyperparameters}.log')
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    log.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    
    stream_handler.setFormatter(formatter)
    
    log.addHandler(stream_handler)
    
    log.debug(f'Logfile {logfile} has been created!')
    log.info('AI for Complete Intersection Calabi-Yau 4-folds')
    log.info('Authors: H. Erbin, R. Finotello')
    log.info('Code: R. Finotello')
    
    # load the dataset
    X_train, y_train = None, None
    X_val, y_val = None, None
    X_test, y_test = None, None
    
    with open(args.xtrain) as f:
        log.debug('Loading training features...')
        X_train = json.load(f)
        X_train = {key: np.asarray(value) for key, value in X_train.items()}
      
    with open(args.xvalid) as f:
        log.debug('Loading validation features...')
        X_val = json.load(f)
        X_val = {key: np.asarray(value) for key, value in X_val.items()}
    
    with open(args.xtest) as f:
        log.debug('Loading test features...')
        X_test = json.load(f)
        X_test = {key: np.asarray(value) for key, value in X_test.items()}
    
    with open(args.ytrain) as f:
        log.debug('Loading training labels...')
        y_train = json.load(f)
        y_train = {key: np.asarray(value) for key, value in y_train.items()}
    
    with open(args.yvalid) as f:
        log.debug('Loading validation labels...')
        y_val = json.load(f)
        y_val = {key: np.asarray(value) for key, value in y_val.items()}
    
    with open(args.ytest) as f:
        log.debug('Loading test labels...')
        y_test = json.load(f)
        y_test = {key: np.asarray(value) for key, value in y_test.items()}
    
    
    log.debug('Dataset loaded!')

    # select features and labels
    feat = list(X_train.keys())
    labs = list(y_train.keys())

    # rescale the labels
    log.debug('Rescaling the labels in [0, 1]...')
    
    y_min = {k: v.min() for k, v in y_train.items()}
    y_max = {k: v.max() for k, v in y_train.items()}

    y_store = {'training': y_train, 'validation': y_val, 'test': y_test}

    y_train = {k: (v - y_min[k]) / (y_max[k] - y_min[k]) for k, v in y_train.items()}
    y_val   = {k: (v - y_min[k]) / (y_max[k] - y_min[k]) for k, v in y_val.items()}
    y_test  = {k: (v - y_min[k]) / (y_max[k] - y_min[k]) for k, v in y_test.items()}

    # rescale the features
    log.debug('Rescaling the features in [0, 1]...')
    X_min = {k: v.min() for k, v in X_train.items()}
    X_max = {k: v.max() for k, v in X_train.items()}

    X_train = {k: (v - X_min[k]) / (X_max[k] - X_min[k]) for k, v in X_train.items()}
    X_val   = {k: (v - X_min[k]) / (X_max[k] - X_min[k]) for k, v in X_val.items()}
    X_test  = {k: (v - X_min[k]) / (X_max[k] - X_min[k]) for k, v in X_test.items()}
    
    # define the input shape
    input_shape = X_train['matrix'].shape[1:]
    
    log.debug(f'Input shape of the model: {input_shape}.')
    
    # load the hyperparameter list
    parameters_list = None
    with open(args.params) as f:
        parameters_list = json.load(f)
    
    # select the parameters
    p = parameters_list[args.hyperparameters]
    
    log.info(f'Investigating model {args.hyperparameters:d}.')
    log.debug(f'Parameters: {p}')

    # build and train the model
    tf.random.set_seed(args.random)
    keras.backend.clear_session()

    # input layer
    x = keras.layers.Input(shape=input_shape, name='matrix')
    I = {'matrix': x}

    # reshape
    x = keras.layers.Reshape(input_shape + (1,))(x)

    # inception modules
    for i, f in enumerate(p['filters']):

        A = None # separate max pooling
        if i > 0 and p['sep_mpool'] is not None:

            A = keras.layers.MaxPool2D(pool_size=p['sep_mpool'],
                                       padding=p['padding'],
                                       strides=(1,1)
                                      )(x)

            A = InceptionModule(filters=f,
                                reduce=p['reduce'],
                                single=p['single'],
                                kernels=p['kernels'],
                                padding=p['padding'],
                                pool=p['pool'],
                                pool_size=p['pool_size'],
                                alpha=p['alpha'],
                                l1_reg=p['l1_reg'],
                                l2_reg=p['l2_reg'],
                                norm=p['norm'],
                                dropout=p['full_dropout'],
                                seed=p['seed']
                               )(A)

            A = AM(sam=p['sam'], cham=p['cham'], seed=p['seed'])(A)

            A = ScalarMult()(A)

        B = None # separate max pooling
        if i > 0 and p['sep_apool'] is not None:

            B = keras.layers.AveragePooling2D(pool_size=p['sep_apool'],
                                              padding=p['padding'],
                                              strides=(1,1)
                                             )(x)

            B = InceptionModule(filters=f,
                                reduce=p['reduce'],
                                single=p['single'],
                                kernels=p['kernels'],
                                padding=p['padding'],
                                pool=p['pool'],
                                pool_size=p['pool_size'],
                                alpha=p['alpha'],
                                l1_reg=p['l1_reg'],
                                l2_reg=p['l2_reg'],
                                norm=p['norm'],
                                dropout=p['full_dropout'],
                                seed=p['seed']
                               )(B)

            B = AM(sam=p['sam'], cham=p['cham'], seed=p['seed'])(B)

            B = ScalarMult()(B)
            
        # add regular inception modules
        x = InceptionModule(filters=f,
                            reduce=p['reduce'],
                            single=p['single'],
                            kernels=p['kernels'],
                            padding=p['padding'],
                            pool=p['pool'],
                            pool_size=p['pool_size'],
                            alpha=p['alpha'],
                            l1_reg=p['l1_reg'],
                            l2_reg=p['l2_reg'],
                            norm=p['norm'],
                            dropout=p['full_dropout'],
                            seed=p['seed']
                           )(x)
            
        if i > 0: # avoid attention modules in first layer

            x = AM(sam=p['sam'], cham=p['cham'], seed=p['seed'])(x)

        # add the pooling effects
        if A is not None:

            x = keras.layers.Add()([x, A])

        if B is not None:

            x = keras.layers.Add()([x, B])

    # additional inception modules
    fine_tuned = {l: x for l in labs}
    for i, f in enumerate(p['fine_tuning']):

        for l in labs:

            fine_tuned[l] = InceptionModule(filters=f,
                                            reduce=p['reduce'],
                                            single=p['single'],
                                            kernels=p['kernels'],
                                            padding=p['padding'],
                                            pool=p['pool'],
                                            pool_size=p['pool_size'],
                                            alpha=p['alpha'],
                                            l1_reg=p['l1_reg'],
                                            l2_reg=p['l2_reg'],
                                            norm=p['norm'],
                                            dropout=p['full_dropout'],
                                            seed=p['seed']
                                           )(fine_tuned[l])
                        
            fine_tuned[l] = AM(sam=p['sam'], cham=p['cham'], seed=p['seed'])(fine_tuned[l])

    # feature vector
    if p['fmap'] is not None:

        for l in labs:

            fine_tuned[l] = FeatureMap(p['fmap'], alpha=p['alpha'], seed=p['seed'])(fine_tuned[l])

    else:

        for l in labs:

            fine_tuned[l] = keras.layers.Flatten()(fine_tuned[l])
        
    # dropout
    if p['dropout'] is not None:

        for l in labs:

            fine_tuned[l] = keras.layers.Dropout(rate=p['dropout'], seed=p['seed'])(fine_tuned[l])

    # dense network
    if p['dense'] is not None:
            
        for u in p['dense']:

            for l in labs:
                
                fine_tuned[l] = DenseActivation(u,
                                                alpha=p['alpha'],
                                                l1_reg=p['l1_reg'],
                                                l2_reg=p['l2_reg'],
                                                norm=p['norm'],
                                                dropout=p['full_dropout'],
                                                seed=p['seed']
                                               )(fine_tuned[l])

    # outputs
    O = {l: keras.layers.Dense(1,
                               activation='relu',
                               kernel_initializer=keras.initializers.GlorotUniform(p['seed']),
                               bias_initializer=keras.initializers.Zeros(),
                               name=l
                              )(fine_tuned[l]) for l in labs
        }

    # define the loss
    loss = keras.losses.MeanSquaredError()
    loss_name = 'mean squared error'
    log_plot = True

    if p['beta'] is not None:

        loss = ImbalancedMSE(p['beta'])
        loss_name = 'imbalanced mean squared error'

    if p['delta'] is not None:

        loss = keras.losses.Huber(delta=p['delta'])
        loss_name = 'huber'

        if p['beta'] is not None:

            loss = ImbalancedHuber(p['beta'], p['delta'])
            loss_name = 'imbalanced huber'

    # build and compile the model
    model = keras.Model(inputs=I, outputs=O, name=f'attentive_inception_{args.hyperparameters:d}')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p['learning_rate']),
                  loss=loss,
                  metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()],
                  loss_weights=p['loss_weights']
                 )

    # print the summary
    log.debug(f'loss function: {loss_name}')
    for k, v in p.items():
        log.debug(f'{k}: {v}')
        
    model.summary(print_fn=lambda x: log.info(x))
        
    # save the model
    root_mod = os.path.join(mod_dir, f'model_{args.hyperparameters:d}')
    root_img = os.path.join(img_dir, f'model_{args.hyperparameters:d}')
    os.makedirs(root_mod, exist_ok=True)
    os.makedirs(root_img, exist_ok=True)
        
    with open(os.path.join(root_mod, f'model_{args.hyperparameters:d}_arch.json'), 'w') as f:
        f.write(model.to_json())
            
    # fit the model (add the callbacks)
    callbacks = []
    callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(root_mod, 'val_loss.h5'),
                                                     monitor='val_loss',
                                                     save_best_only=True,
                                                     save_weights_only=True
                                                    )
                    )
                    
    for l in labs:
    
        callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(root_mod, f'val_{l}_loss.h5'),
                                                         monitor=f'val_{l}_loss',
                                                         save_best_only=True,
                                                         save_weights_only=True
                                                        )
                        )
                        
    callbacks.append(keras.callbacks.ReduceLROnPlateau(factor=p['reduce_lr'], patience=p['lr_patience'], min_lr=1.0e-6))
    callbacks.append(PrintCheckpoint(int(p['epochs'] / 10) if p['epochs'] > 10 else 1, log))

    mod_hst = model.fit(x=X_train,
                        y=y_train,
                        batch_size=p['batch_size'],
                        epochs=p['epochs'],
                        verbose=0,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                       )
    # save the history
    hist = mod_hst.history

    joblib.dump(hist, os.path.join(root_mod, 'history.joblib'))
    log.debug('History file saved!')
        
    # plot the loss function for the general model
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    x = np.arange(p['epochs'])

    sns.lineplot(x=x,
                 y=hist['loss'],
                 color='tab:blue',
                 linestyle='-',
                 label='training',
                 ax=ax
                )

    sns.lineplot(x=x,
                 y=hist['val_loss'],
                 color='tab:red',
                 linestyle='--',
                 label='validation',
                 ax=ax
                )

    ax.set(title='Loss Function (general model)',
           xlabel='epochs',
           ylabel='loss'
          )

    if log_plot:
        ax.set_yscale('log')

    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(root_img, 'loss_function_general_model.pdf'), dpi=300)
    plt.savefig(os.path.join(root_img, 'loss_function_general_model.png'), dpi=300)
    plt.close(fig)


    # plot the loss function for each optimised model
    for l in labs:

        fig, ax = plt.subplots(1, 1, figsize=(6,5))

        sns.lineplot(x=x,
                     y=hist[f'{l}_loss'],
                     color='tab:blue',
                     linestyle='-',
                     label='training',
                     ax=ax
                    )

        sns.lineplot(x=x,
                     y=hist[f'val_{l}_loss'],
                     color='tab:red',
                     linestyle='--',
                     label='validation',
                     ax=ax
                    )

        ax.set(title=f'Loss Function ({l} model)',
               xlabel='epochs',
               ylabel='loss'
              )

        if log_plot:
            ax.set_yscale('log')

        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(root_img, f'loss_function_{l}_model.pdf'), dpi=300)
        plt.savefig(os.path.join(root_img, f'loss_function_{l}_model.png'), dpi=300)
        plt.close(fig)
        
    # plot the learning rate
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    sns.lineplot(x=x,
                 y=hist['lr'],
                 color='tab:blue',
                 label='learning rate',
                 ax=ax
                )

    ax.set(title='Learning Rate',
           xlabel='epochs',
           ylabel='learning rate'
          )

    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(root_img, 'learning_rate.pdf'), dpi=300)
    plt.savefig(os.path.join(root_img, 'learning_rate.png'), dpi=300)
    plt.close(fig)
        
    log.debug('Losses and learning rate have been plotted!')
            
    # compute predictions of the general model
    model.load_weights(os.path.join(root_mod, 'val_loss.h5'))

    predictions = {}
    predictions['training']   = dict(zip(labs, model.predict(X_train)))
    predictions['validation'] = dict(zip(labs, model.predict(X_val)))
    predictions['test']       = dict(zip(labs, model.predict(X_test)))
    
    # rescale back
    y_train = y_store['training']
    y_val   = y_store['validation']
    y_test  = y_store['test']

    for t in ['training', 'validation', 'test']:
        predictions[t] = {out: value.reshape(-1,) * (y_max[out] - y_min[out]) + y_min[out] for out, value in predictions[t].items()}

    # convert to lists for JSON serialisation
    for t in ['training', 'validation', 'test']:
        predictions[t] = {out: value.reshape(-1,).tolist() for out, value in predictions[t].items()}

    with open(os.path.join(root_mod, 'predictions.json'), 'w') as f:
        json.dump(predictions, f)

    residuals = {}
    residuals['training'] = {out: (np.array(values) - y_train[out].reshape(-1,)).tolist() for out, values in predictions['training'].items()}
    residuals['validation'] = {out: (np.array(values) - y_val[out].reshape(-1,)).tolist() for out, values in predictions['validation'].items()}
    residuals['test'] = {out: (np.array(values) - y_test[out].reshape(-1,)).tolist() for out, values in predictions['test'].items()}
    
    with open(os.path.join(root_mod, 'residuals.json'), 'w') as f:
        json.dump(residuals, f)
        
    # compute the metrics of the general model
    metrics = {}
    metrics['training']   = {out: tf.reduce_mean(tf.cast(tf.equal(y_train[out].reshape(-1,), tf.math.rint(predictions['training'][out])), tf.float32)).numpy().astype(float) for out in labs}
    metrics['validation'] = {out: tf.reduce_mean(tf.cast(tf.equal(y_val[out].reshape(-1,), tf.math.rint(predictions['validation'][out])), tf.float32)).numpy().astype(float) for out in labs}
    metrics['test']       = {out: tf.reduce_mean(tf.cast(tf.equal(y_test[out].reshape(-1,), tf.math.rint(predictions['test'][out])), tf.float32)).numpy().astype(float) for out in labs}

    with open(os.path.join(root_mod, 'metrics_rint_general.json'), 'w') as f:
        json.dump(metrics, f)
            
    log.info(f'Metrics of the general model (RINT): {metrics}.')

    metrics = {}
    metrics['training']   = {out: tf.reduce_mean(tf.cast(tf.equal(y_train[out].reshape(-1,), tf.math.floor(predictions['training'][out])), tf.float32)).numpy().astype(float) for out in labs}
    metrics['validation'] = {out: tf.reduce_mean(tf.cast(tf.equal(y_val[out].reshape(-1,), tf.math.floor(predictions['validation'][out])), tf.float32)).numpy().astype(float) for out in labs}
    metrics['test']       = {out: tf.reduce_mean(tf.cast(tf.equal(y_test[out].reshape(-1,), tf.math.floor(predictions['test'][out])), tf.float32)).numpy().astype(float) for out in labs}

    with open(os.path.join(root_mod, 'metrics_floor_general.json'), 'w') as f:
        json.dump(metrics, f)
            
    log.info(f'Metrics of the general model (FLOOR): {metrics}.')

    metrics = {}
    metrics['training']   = {out: tf.reduce_mean(tf.cast(tf.equal(y_train[out].reshape(-1,), tf.math.ceil(predictions['training'][out])), tf.float32)).numpy().astype(float) for out in labs}
    metrics['validation'] = {out: tf.reduce_mean(tf.cast(tf.equal(y_val[out].reshape(-1,), tf.math.ceil(predictions['validation'][out])), tf.float32)).numpy().astype(float) for out in labs}
    metrics['test']       = {out: tf.reduce_mean(tf.cast(tf.equal(y_test[out].reshape(-1,), tf.math.ceil(predictions['test'][out])), tf.float32)).numpy().astype(float) for out in labs}

    with open(os.path.join(root_mod, 'metrics_ceil_general.json'), 'w') as f:
        json.dump(metrics, f)
            
    log.info(f'Metrics of the general model (CEIL): {metrics}.')

    # plot the residuals of the general model
    for l in labs:

        fig, ax = plt.subplots(1, 1, figsize=(6,5))

        sns.histplot(residuals['training'][l],
                     binwidth=1,
                     alpha=0.35,
                     color='tab:blue',
                     label='training',
                     log_scale=(False,True),
                     ax=ax
                    )

        sns.histplot(residuals['validation'][l],
                     binwidth=1,
                     alpha=0.35,
                     color='tab:red',
                     label='validation',
                     log_scale=(False,True),
                     ax=ax
                    )

        sns.histplot(residuals['test'][l],
                     binwidth=1,
                     alpha=0.35,
                     color='tab:green',
                     label='test',
                     log_scale=(False,True),
                     ax=ax
                    )

        ax.set(title=f'Residual Histogram (general model, {l} output)',
               xlabel='residuals',
               ylabel='count'
              )
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(root_img, f'reshist_general_{l}_output.pdf'), dpi=300)
        plt.savefig(os.path.join(root_img, f'reshist_general_{l}_output.png'), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6,5))

        sns.scatterplot(x=predictions['training'][l],
                        y=residuals['training'][l],
                        alpha=0.35,
                        color='tab:blue',
                        label='training',
                        ax=ax
                       )

        sns.scatterplot(x=predictions['validation'][l],
                        y=residuals['validation'][l],
                        alpha=0.35,
                        color='tab:red',
                        label='validation',
                        ax=ax
                       )

        sns.scatterplot(x=predictions['test'][l],
                        y=residuals['test'][l],
                        alpha=0.35,
                        color='tab:green',
                        label='test',
                        ax=ax
                       )

        ax.set(title=f'Residual Plot (general model, {l} output)',
               xlabel='predictions',
               ylabel='residuals'
              )
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(root_img, f'resplot_general_{l}_output.pdf'), dpi=300)
        plt.savefig(os.path.join(root_img, f'resplot_general_{l}_output.png'), dpi=300)
        plt.close(fig)
        
    # compute the predictions for the optimised models
    for l in labs:

        model.load_weights(os.path.join(root_mod, f'val_{l}_loss.h5'))

        predictions = {}
        predictions['training']   = dict(zip(labs, list(model.predict(X_train))))
        predictions['validation'] = dict(zip(labs, list(model.predict(X_val))))
        predictions['test']       = dict(zip(labs, list(model.predict(X_test))))
        
        # rescale back
        for t in ['training', 'validation', 'test']:
            predictions[t] = {out: value.reshape(-1,) * (y_max[out] - y_min[out]) + y_min[out] for out, value in predictions[t].items()}
    
        # convert to lists for JSON serialisation
        for t in ['training', 'validation', 'test']:
            predictions[t] = {out: value.reshape(-1,).tolist() for out, value in predictions[t].items()}

        with open(os.path.join(root_mod, f'predictions_{l}_optimised.json'), 'w') as f:
            json.dump(predictions, f)

        residuals = {}
        residuals['training'] = {out: (np.array(values) - y_train[out].reshape(-1,)).tolist() for out, values in predictions['training'].items()}
        residuals['validation'] = {out: (np.array(values) - y_val[out].reshape(-1,)).tolist() for out, values in predictions['validation'].items()}
        residuals['test'] = {out: (np.array(values) - y_test[out].reshape(-1,)).tolist() for out, values in predictions['test'].items()}
        
        with open(os.path.join(root_mod, f'residuals_{l}_optimised.json'), 'w') as f:
            json.dump(residuals, f)
        
        # compute the metrics of the optimised model
        metrics = {}
        metrics['training']   = tf.reduce_mean(tf.cast(tf.equal(y_train[l].reshape(-1,), tf.math.rint(predictions['training'][l])), tf.float32)).numpy().astype(float)
        metrics['validation'] = tf.reduce_mean(tf.cast(tf.equal(y_val[l].reshape(-1,), tf.math.rint(predictions['validation'][l])), tf.float32)).numpy().astype(float)
        metrics['test']       = tf.reduce_mean(tf.cast(tf.equal(y_test[l].reshape(-1,), tf.math.rint(predictions['test'][l])), tf.float32)).numpy().astype(float)

        with open(os.path.join(root_mod, f'metrics_rint_{l}_optimised.json'), 'w') as f:
            json.dump(metrics, f)
            
        log.info(f'Metrics of the {l} optimised model (RINT): {metrics}.')

        metrics = {}
        metrics['training']   = tf.reduce_mean(tf.cast(tf.equal(y_train[l].reshape(-1,), tf.math.floor(predictions['training'][l])), tf.float32)).numpy().astype(float)
        metrics['validation'] = tf.reduce_mean(tf.cast(tf.equal(y_val[l].reshape(-1,), tf.math.floor(predictions['validation'][l])), tf.float32)).numpy().astype(float)
        metrics['test']       = tf.reduce_mean(tf.cast(tf.equal(y_test[l].reshape(-1,), tf.math.floor(predictions['test'][l])), tf.float32)).numpy().astype(float)

        with open(os.path.join(root_mod, f'metrics_floor_{l}_optimised.json'), 'w') as f:
            json.dump(metrics, f)
            
        log.info(f'Metrics of the {l} optimised model (FLOOR): {metrics}.')

        metrics = {}
        metrics['training']   = tf.reduce_mean(tf.cast(tf.equal(y_train[l].reshape(-1,), tf.math.ceil(predictions['training'][l])), tf.float32)).numpy().astype(float)
        metrics['validation'] = tf.reduce_mean(tf.cast(tf.equal(y_val[l].reshape(-1,), tf.math.ceil(predictions['validation'][l])), tf.float32)).numpy().astype(float)
        metrics['test']       = tf.reduce_mean(tf.cast(tf.equal(y_test[l].reshape(-1,), tf.math.ceil(predictions['test'][l])), tf.float32)).numpy().astype(float)

        with open(os.path.join(root_mod, f'metrics_ceil_{l}_optimised.json'), 'w') as f:
            json.dump(metrics, f)
            
        log.info(f'Metrics of the {l} optimised model (CEIL): {metrics}.')

        # plot the residuals of the optimised models
        fig, ax = plt.subplots(1, 1, figsize=(6,5))

        sns.histplot(residuals['training'][l],
                     binwidth=1,
                     alpha=0.35,
                     color='tab:blue',
                     label='training',
                     log_scale=(False,True),
                     ax=ax
                    )

        sns.histplot(residuals['validation'][l],
                     binwidth=1,
                     alpha=0.35,
                     color='tab:red',
                     label='validation',
                     log_scale=(False,True),
                     ax=ax
                    )

        sns.histplot(residuals['test'][l],
                     binwidth=1,
                     alpha=0.35,
                     color='tab:green',
                     label='test',
                     log_scale=(False,True),
                     ax=ax
                    )

        ax.set(title=f'Residual Histogram ({l} optimised model)',
               xlabel='residuals',
               ylabel='count'
              )
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(root_img, f'reshist_{l}_optimised.pdf'), dpi=300)
        plt.savefig(os.path.join(root_img, f'reshist_{l}_optimised.png'), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6,5))

        sns.scatterplot(x=predictions['training'][l],
                        y=residuals['training'][l],
                        alpha=0.35,
                        color='tab:blue',
                        label='training',
                        ax=ax
                       )

        sns.scatterplot(x=predictions['validation'][l],
                        y=residuals['validation'][l],
                        alpha=0.35,
                        color='tab:red',
                        label='validation',
                        ax=ax
                       )

        sns.scatterplot(x=predictions['test'][l],
                        y=residuals['test'][l],
                        alpha=0.35,
                        color='tab:green',
                        label='test',
                        ax=ax
                       )

        ax.set(title=f'Residual Plot ({l} optimised model)',
               xlabel='predictions',
               ylabel='residuals'
              )
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(root_img, f'resplot_{l}_optimised.pdf'), dpi=300)
        plt.savefig(os.path.join(root_img, f'resplot_{l}_optimised.png'), dpi=300)
        plt.close(fig)

    exit(0)
