import tensorflow as tf
from tensorflow import keras

#####################
# AUXILIARY         #
#####################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

#####################
# INCEPTION         #
#####################
def nn_inception(input_shape,
                 name='inception',
                 learning_rate=1.0e-3,
                 output_names=['h11', 'h21', 'h31', 'h22'],
                 conv_1b1=False,
                 conv_layers=[32, 64, 32],
                 conv_alpha=0.0,
                 conv_class=None,
                 conv_single=False,
                 conv_red=0,
                 feat_map=0,
                 recurrent=False,
                 fc_layers=[],
                 fc_alpha=0.0,
                 dropout=0.2,
                 full_dropout=0.0,
                 batchnorm=0.99,
                 last_relu=True,
                 loss_weights=[0.25, 0.25, 0.25, 0.25],
                 l1_reg=0.0,
                 l2_reg=0.0,
                 auxiliary_fc=None,
                 auxiliary_alpha=[],
                 auxiliary_last_relu=False,
                 auxiliary_suff='aux',
                 output_layers=True,
                 random_state=None
                ):
    '''
    Create a Tensorflow model of the Inception Network.
    
    Arguments:
        input_shape: the shape of the input tensor (tuple).
        
    Optional:
        name:                the name of the model,
        learning_rate:       the shrinking parameter of gradient descent,
        output_names:        list of names of the output,
        conv_1b1:            add 1x1 convolutions at the beginning (normalise the number of filters),
        conv_layers:         list of filters in convolutional layers,
        conv_alpha:          slope of the LeakyReLU activation of the convolutional layers,
        conv_class:          "classical" kernel of the additional convolution in the inception module (tuple or int),
        conv_single:         add 1x1 convolution in the inception module (in parallel to other convolutions),
        conv_red:            reduce the number of filters of the inception module (filters of the 1x1 convolutions),
        feat_map:            lenght of the feature map (from inception modules to (1,1,feat_map) tensor),
        recurrent:           make it a recurrent network,
        fc_layers:           list of units in the fully connected network,
        fc_alpha:            slope of the LeakyReLU activation of the fully connected layers,
        dropout:             dropout rate before the flatten layer,
        full_dropout:        dropout rate after every layer,
        batchnorm:           momentum of the batch normalisation (no batch normalisation if < 0),
        last_relu:           include ReLU in the output layers (bool),
        loss_weights:        the weights of the loss function,
        l1_reg:              l1 regularisation factor,
        l2_reg:              l2 regularisation factor,
        auxiliary_fc:        list of auxiliary fully connected layers,
        auxiliary_alpha:     slope of the LeakyReLU activation in the auxiliary network,
        auxiliary_last_relu: include relu in the auxiliary output,
        auxiliary_suff:      suffix to add to the names of the auxiliary output,
        output_layers:       include output layers (e.g. False when this is only part of a model),
        random_state:        the random state.
        
    Returns:
        the Tensorflow model.
        
    N.B.:
        the `auxiliary_fc` parameter add creates a model with multiple outputs:
        
            INPUT
              |
              |
            HODGE
           NUMBERS ---- OUTPUT ---- LOSS
              |
              |
          AUXILIARY
             FC
              |
              |
            HODGE
           NUMBERS ---- OUTPUT ---- LOSS
    '''
    keras.backend.clear_session()
    
    # input layer
    x = keras.layers.Input(shape=input_shape, name=name)
    I = {name: x}
    
    # placeholder for ResNet
    previous_concat = x
    
    # build convolutions
    for n, filters in enumerate(conv_layers):
        # first branch
        a = x
        if conv_1b1 and n != 0:
            a = keras.layers.Conv2D(filters,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                    bias_initializer=keras.initializers.Zeros(),
                                    name=name + '_convA1b1a_' + str(n)
                                   )(a)
            a = keras.layers.LeakyReLU(conv_alpha, name=name + '_actA1b1a_' + str(n))(a)
            
        a = keras.layers.Conv2D(filters,
                                kernel_size=(int(x.shape[1]), 1),
                                padding='same',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                bias_initializer=keras.initializers.Zeros(),
                                name=name + '_convAa_' + str(n)
                               )(a)
        a = keras.layers.LeakyReLU(conv_alpha, name=name+ '_actAa_' + str(n))(a)
        
        # second branch
        b = x
        if conv_1b1 and n != 0:
            b = keras.layers.Conv2D(filters,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                    bias_initializer=keras.initializers.Zeros(),
                                    name=name + '_convA1b1b_' + str(n)
                                   )(b)
            b = keras.layers.LeakyReLU(conv_alpha, name=name+ '_actA1b1b_' + str(n))(b)
        
        b = keras.layers.Conv2D(filters,
                                kernel_size=(1, int(x.shape[2])),
                                padding='same',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                bias_initializer=keras.initializers.Zeros(),
                                name=name + '_convAb_' + str(n)
                               )(b)
        b = keras.layers.LeakyReLU(conv_alpha, name=name+ '_actAb_' + str(n))(b)
        
        # classical convolution
        c = None
        if conv_class is not None:
            # factorise the kernel (5x5 ~ 3x3 + 3x3)
            if isinstance(conv_class, int):
                conv_class = (conv_class, conv_class)
            
            # third branch
            c = x
            if conv_1b1 and n != 0:
                c = keras.layers.Conv2D(filters,
                                        kernel_size=(1, 1),
                                        padding='same',
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                        bias_initializer=keras.initializers.Zeros(),
                                        name=name + '_convB1b1_' + str(n)
                                       )(c)
                c = keras.layers.LeakyReLU(conv_alpha, name=name + '_actB1b1_' + str(n))(c)
            
            c = keras.layers.Conv2D(filters,
                                    kernel_size=conv_class,
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                    bias_initializer=keras.initializers.Zeros(),
                                    name=name + '_convB_' + str(n)
                                   )(c)
            c = keras.layers.LeakyReLU(conv_alpha, name=name + '_actB_' + str(n))(c)
                
        
        # fourth branch (1x1 convolution)
        d = None
        if conv_single:
            d = x
            d = keras.layers.Conv2D(filters,
                                    kernel_size=(1,1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                    bias_initializer=keras.initializers.Zeros(),
                                    name=name + '_convC_' + str(n)
                                   )(d)
            d = keras.layers.LeakyReLU(conv_alpha, name=name + '_actC_' + str(n))(d)
            
        # concatenate the branches
        conc = [a, b]
        if c is not None:
            conc.append(c)
        if d is not None:
            conc.append(d)
        x = keras.layers.Concatenate(axis=-1, name=name + '_cat_' + str(n))(conc)
        
        # add batchnorm and dropout
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_convBN_' + str(n))(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_convFullDrop_' + str(n))(x)
        
        # ResNet
        if previous_concat is not None and recurrent is not False:
            x = keras.layers.Concatenate(axis=-1, name=name + '_resCat_' + str(n))([x, previous_concat])
            if batchnorm > 0.0:
                x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_resBN_' + str(n))(x)
            if full_dropout > 0.0:
                x = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_resFullDrop_' + str(n))(x)
        previous_concat = x
    
    # 1x1 convolution to reduce the no. of filters
    if conv_red > 0:
        x = keras.layers.Conv2D(conv_red,
                                kernel_size=(1,1),
                                padding='valid',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                bias_initializer=keras.initializers.Zeros(),
                                name=name + '_convRed'
                               )(x)
        x = keras.layers.LeakyReLU(conv_alpha, name=name + '_actRed')(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_redBN')(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_redFullDrop')(x)
        
    # flatten and dropout (add a feature map if needed)
    if feat_map > 0:
        x = keras.layers.Conv2D(feat_map,
                                kernel_size=(int(x.shape[1]), int(x.shape[2])),
                                padding='valid',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                bias_initializer=keras.initializers.Zeros(),
                                name=name + '_featmap'
                               )(x)
        x = keras.layers.Flatten(name=name + '_flat')(x)
        x = keras.layers.LeakyReLU(conv_alpha, name=name + '_flatAct')(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_flatBN')(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_flatFullDrop')(x)
    else:
        x = keras.layers.Flatten(name=name + '_flat')(x)
        
    if dropout > 0.0:
        x = keras.layers.Dropout(rate=dropout, seed=random_state, name=name + '_drop')(x)
    
    # fully connected network
    for n, units in enumerate(fc_layers):
        x = keras.layers.Dense(units,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               kernel_initializer=keras.initializers.GlorotUniform(random_state),
                               bias_initializer=keras.initializers.Zeros(),
                               name=name + '_fc_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcact_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_' + str(n))(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_fcFullDrop_' + str(n))(x)
            
    # output layers
    if output_layers:
        if last_relu:
            O = {name: keras.layers.Dense(1,
                                          activation='relu',
                                          kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                          bias_initializer=keras.initializers.Zeros(),
                                          name=name
                                         )(x) for name in output_names
                }
        else:
            O = {name: keras.layers.Dense(1,
                                          kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                          bias_initializer=keras.initializers.Zeros(),
                                          name=name
                                         )(x) for name in output_names
                }
        
        # add auxiliary output
        O_aux = None
        if auxiliary_fc is not None:
        
            if not isinstance(auxiliary_fc, list):
                auxiliary_fc = [auxiliary_fc]
                
            aux = keras.layers.concatenate([O[name] for name in O.keys()])

            for n, units in enumerate(auxiliary_fc):
                aux = keras.layers.Dense(units,
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                         kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                         bias_initializer=keras.initializers.Zeros(),
                                         name=name + '_fc_aux_' + str(n)
                                        )(aux)
                aux = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcact_aux_' + str(n))(aux)
                if batchnorm > 0.0:
                    aux = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_aux_' + str(n))(aux)
                if full_dropout > 0.0:
                    aux = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_fcFullDrop_aux_' + str(n))(aux)
                    
            if last_relu:
                O_aux = {name: keras.layers.Dense(1,
                                                  activation='relu',
                                                  kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                                  bias_initializer=keras.initializers.Zeros(),
                                                  name=name
                                                 )(aux) for name in [name + '_' + auxiliary_suff for name in output_names]
                        }
            else:
                O_aux = {name: keras.layers.Dense(1,
                                              kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                              bias_initializer=keras.initializers.Zeros(),
                                              name=name
                                             )(aux) for name in [name + '_' + auxiliary_suff for name in output_names]
                    }
                
        # create the loss weights
        loss_weights = {name: loss_weights[n] for n, name in enumerate(output_names)}
        
        # compile the model
        if O_aux is not None:
            O.update(O_aux)
            model = keras.models.Model(inputs=I, outputs=O, name=name)
            loss_weights.update({key + '_' + auxiliary_suff: value for key, value in loss_weights.items()})
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=keras.losses.MeanSquaredError(),
                          loss_weights=loss_weights,
                          metrics=[keras.metrics.MeanSquaredError(),
                                   keras.metrics.MeanAbsoluteError()
                                  ]
                         )
        else:
            model = keras.models.Model(inputs=I, outputs=O, name=name)
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=keras.losses.MeanSquaredError(),
                          loss_weights=loss_weights,
                          metrics=[keras.metrics.MeanSquaredError(),
                                   keras.metrics.MeanAbsoluteError()
                                  ]
                         )
    else:
        O = x
        
        # build the model
        model = keras.models.Model(inputs=I, outputs=O, name=name)
    
    return model


#####################
# FULLY CONNECTED   #
#####################
def nn_dense(input_shape,
             name='dense',
             learning_rate=1.0e-3,
             output_names=['h11', 'h21', 'h31', 'h22'],
             fc_layers=[32],
             fc_alpha=0.0,
             recurrent=False,
             dropout=0.2,
             batchnorm=0.99,
             last_relu=True,
             loss_weights=[0.25, 0.25, 0.25, 0.25],
             l1_reg=0.0,
             l2_reg=0.0,
             auxiliary_fc=None,
             auxiliary_alpha=[],
             auxiliary_last_relu=False,
             auxiliary_suff='aux',
             output_layers=True,
             random_state=None
            ):
    '''
    Create a Tensorflow model of a fully connected Network.
    
    Arguments:
        input_shape: the shape of the input tensor.
        
    Optional:
        name:                the name of the model,
        learning_rate:       the shrinking parameter of gradient descent,
        output_names:        list of names of the output,
        fc_layers:           list of units in the fully connected network,
        fc_alpha:            slope of the LeakyReLU activation of the fully connected layers,
        recurrent:           make it a recurrent network,
        dropout:             dropout rate,
        batchnorm:           momentum of the batch normalisation (no batch normalisation if < 0),
        last_relu:           include ReLU in the output layers (bool),
        loss_weights:        the weights of the loss function,
        l1_reg:              l1 regularisation factor,
        l2_reg:              l2 regularisation factor,
        auxiliary_fc:        list of auxiliary fully connected layers,
        auxiliary_alpha:     slope of the LeakyReLU activation in the auxiliary network,
        auxiliary_last_relu: include relu in the auxiliary output,
        auxiliary_suff:      suffix to add to the names of the auxiliary output,
        output_layers:       include output layers (e.g. False when this is only part of a model),
        random_state:        the random state.
        
    Returns:
        the Tensorflow model.
    '''
    keras.backend.clear_session()
    
    # input layer
    x = keras.layers.Input(shape=input_shape, name=name)
    I = {name: x}
    
    # placeholder for ResNet
    previous_concat = None
    
    # build connections
    for n, units in enumerate(fc_layers):
        x = keras.layers.Dense(units,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               kernel_initializer=keras.initializers.GlorotUniform(random_state),
                               bias_initializer=keras.initializers.Zeros(),
                               name=name + '_fc_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcAct_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_' + str(n))(x)
        if dropout > 0.0:
            x = keras.layers.Dropout(rate=dropout, seed=random_state, name=name + '_fcDrop_' + str(n))(x)
        
        # ResNet
        if previous_concat is not None and recurrent is not False:
            x = keras.layers.Concatenate(axis=-1, name=name + '_catRes_' + str(n))([x, previous_concat])
            if batchnorm > 0.0:
                x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_resBN_' + str(n))(x)
            if dropout > 0.0:
                x = keras.layers.Dropout(rate=dropout, seed=random_state, name=name + '_resDrop_' + str(n))(x)
        previous_concat = x
            
    # output layers
    if output_layers:
        if last_relu:
            O = {name: keras.layers.Dense(1,
                                          activation='relu',
                                          kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                          bias_initializer=keras.initializers.Zeros(),
                                          name=name
                                         )(x) for name in output_names
                }
        else:
            O = {name: keras.layers.Dense(1,
                                          kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                          bias_initializer=keras.initializers.Zeros(),
                                          name=name
                                         )(x) for name in output_names
                }
        
        # add auxiliary output
        O_aux = None
        if auxiliary_fc is not None:
        
            if not isinstance(auxiliary_fc, list):
                auxiliary_fc = [auxiliary_fc]
                
            aux = keras.layers.concatenate([O[name] for name in O.keys()])

            for n, units in enumerate(auxiliary_fc):
                aux = keras.layers.Dense(units,
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                         kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                         bias_initializer=keras.initializers.Zeros(),
                                         name=name + '_fc_aux_' + str(n)
                                        )(aux)
                aux = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcact_aux_' + str(n))(aux)
                if batchnorm > 0.0:
                    aux = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_aux_' + str(n))(aux)
                if full_dropout > 0.0:
                    aux = keras.layers.Dropout(rate=full_dropout, seed=random_state, name=name + '_fcFullDrop_aux_' + str(n))(aux)
                    
            if last_relu:
                O_aux = {name: keras.layers.Dense(1,
                                                  activation='relu',
                                                  kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                                  bias_initializer=keras.initializers.Zeros(),
                                                  name=name
                                                 )(aux) for name in [name + '_' + auxiliary_suff for name in output_names]
                        }
            else:
                O_aux = {name: keras.layers.Dense(1,
                                              kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                              bias_initializer=keras.initializers.Zeros(),
                                              name=name
                                             )(aux) for name in [name + '_' + auxiliary_suff for name in output_names]
                    }
                
        # create the loss weights
        loss_weights = {name: loss_weights[n] for n, name in enumerate(output_names)}
        
        # compile the model
        if O_aux is not None:
            model = keras.models.Model(inputs=I, outputs=[O, O_aux], name=name)
            loss_weights.update({key + '_' + auxiliary_suff: value for key, value in loss_weights.items()})
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=keras.losses.MeanSquaredError(),
                          loss_weights=loss_weights,
                          metrics=[keras.metrics.MeanSquaredError(),
                                   keras.metrics.MeanAbsoluteError()
                                  ]
                         )
        else:
            model = keras.models.Model(inputs=I, outputs=O, name=name)
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=keras.losses.MeanSquaredError(),
                          loss_weights=loss_weights,
                          metrics=[keras.metrics.MeanSquaredError(),
                                   keras.metrics.MeanAbsoluteError()
                                  ]
                         )
    else:
        O = x
        
        # build the model
        model = keras.models.Model(inputs=I, outputs=O, name=name)
    
    return model


#####################
# MERGE MODELS      #
#####################
def nn_full(models,
            name='full_model',
            learning_rate=1.0e-3,
            fc_layers=[],
            fc_alpha=0.0,
            batchnorm=0.99,
            dropout=0.1,
            last_relu=True,
            loss_weights=[0.25, 0.25, 0.25, 0.25],
            l1_reg=0.0,
            l2_reg=0.0,
            random_state=None
           ):
    '''
    Build a full model from given separate models.
    
    Arguments:
        models: list of models to merge.
        
    Optional:
        name:          the name of the full model,
        learning_rate: the shrinking parameter of the model,
        fc_layers:     list of units of fully connected layers,
        fc_alpha:      slope of the LeakyReLU activation function,
        batchnorm:     momentum of the batch normalisation (no batch normalisation if < 0),
        dropout:       dropout rate,
        last_relu:     add ReLU activation to output layer (bool),
        loss_weights:  list of loss weights,
        random_state:  the random state.
    
    Returns:
        the full model.
    '''
    keras.backend.clear_session()
    
    # build inputs
    I = {model.name: model.input[model.name] for model in models}
    x = [model.output for model in models]
    x = keras.layers.Concatenate(axis=-1)(x)
    
    for n in range(len(fc_layers)):
        x = keras.layers.Dense(units,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               kernel_initializer=keras.initializers.GlorotUniform(random_state),
                               bias_initializer=keras.initializers.Zeros(),
                               name=name + '_fc_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcAct_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_' + str(n))(x)
        if dropout > 0.0:
            x = keras.layers.Dropout(rate=dropout, seed=random_state, name=name + '_fcDrop_' + str(n))(x)
            
    # output layers
    if last_relu:
        O = {'h11': keras.layers.Dense(1,
                                       activation='relu',
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h11'
                                      )(x),
             'h21': keras.layers.Dense(1,
                                       activation='relu',
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h21'
                                      )(x),
             'h31': keras.layers.Dense(1,
                                       activation='relu',
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h31'
                                      )(x),
             'h22': keras.layers.Dense(1,
                                       activation='relu',
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h22'
                                      )(x),
            }
    else:
        O = {'h11': keras.layers.Dense(1,
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h11'
                                      )(x),
             'h21': keras.layers.Dense(1,
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h21'
                                      )(x),
             'h31': keras.layers.Dense(1,
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h31'
                                      )(x),
             'h22': keras.layers.Dense(1,
                                       kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                       bias_initializer=keras.initializers.Zeros(),
                                       name='h22'
                                      )(x),
            }
    
    # build model
    model = keras.models.Model(inputs=I, outputs=O, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  loss_weights=loss_weights,
                  metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
                 )
    
    return model


#####################
# ADD FC MODEL      #
#####################
def add_fc(model_path,
           name='boost_model',
           set_trainable=False,
           learning_rate=1.0e-3,
           outputs=['output'],
           fc_layers=[],
           fc_alpha=0.0,
           batchnorm=0.99,
           dropout=0.1,
           last_relu=True,
           loss_weights=None,
           l1_reg=0.0,
           l2_reg=0.0,
           random_state=None
          ):
    '''
    Add a fully connected network to a model (boosting a model).
    
    Arguments:
        model_path: path to the model to be boosted.
        
    Optional:
        name:          the name of the full model,
        set_trainable: trainable attribute of the original model (bool),
        learning_rate: the shrinking parameter of the model,
        outputs:       the list of names of the output,
        fc_layers:     list of units of fully connected layers,
        fc_alpha:      slope of the LeakyReLU activation function,
        batchnorm:     momentum of the batch normalisation,
        dropout:       dropout rate,
        last_relu:     add ReLU activation to output layer (bool),
        loss_weights:  list of loss weights,
        random_state:  the random state.
    
    Returns:
        the boosted model.
    '''
    
    keras.backend.clear_session()
    
    # select input and output layers
    model_tmp = keras.models.load_model(model_path)
    model_tmp.trainable = set_trainable
    I = model_tmp.input
    x = model_tmp.outputs
    
    if isinstance(x, list):
        x = keras.layers.concatenate(x)
        
    # add fully connected network
    for n, units in enumerate(fc_layers):
        x = keras.layers.Dense(units,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               kernel_initializer=keras.initializers.GlorotUniform(random_state),
                               bias_initializer=keras.initializers.Zeros(),
                               name=name + '_bst_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_bstAct_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_bstBN_' + str(n))(x)
        if dropout > 0.0:
            x = keras.layers.Dropout(rate=dropout, seed=random_state, name=name + '_bstDrop_' + str(n))(x)
            
    # add output layer
            
    # output layers
    if not isinstance(outputs, list):
        outputs = [outputs]
        
    if last_relu:
        O = {name: keras.layers.Dense(1,
                                      activation='relu',
                                      kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                      bias_initializer=keras.initializers.Zeros(),
                                      name=name
                                     )(x)
             for name in outputs
            }
    else:
        O = {name: keras.layers.Dense(1,
                                      kernel_initializer=keras.initializers.GlorotUniform(random_state),
                                      bias_initializer=keras.initializers.Zeros(),
                                      name=name
                                     )(x)
             for name in outputs
            }
    
    # build model
    model = keras.models.Model(inputs=I, outputs=O, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  loss_weights=loss_weights,
                  metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
                 )
    
    return model