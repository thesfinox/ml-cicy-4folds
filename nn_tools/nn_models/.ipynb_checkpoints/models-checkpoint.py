from tensorflow import keras

#####################
# INCEPTION         #
#####################
def nn_inception(input_shape,
                 name='inception',
                 learning_rate=1.0e-3,
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
                 l1_reg=1.0e-5,
                 l2_reg=1.0e-5,
                 output_layers=True
                ):
    '''
    Create a Tensorflow model of the Inception Network.
    
    Arguments:
        input_shape: the shape of the input tensor.
        
    Optional:
        name:          the name of the model,
        learning_rate: the shrinking parameter of gradient descent,
        conv_1b1:      add 1x1 convolutions at the beginning,
        conv_layers:   list of filters in convolutional layers,
        conv_alpha:    slope of the LeakyReLU activation of the convolutional layers,
        conv_class:    "classical" kernel of the additional convolution in the inception module,
        conv_single:   add 1x1 convolution in the inception module,
        conv_red:      reduce output of convolutions (filters of the 1x1 convolutions),
        feat_map:      reduce to feature map,
        recurrent:     make it a recurrent network,
        fc_layers:     list of units in the fully connected network,
        fc_alpha:      slope of the LeakyReLU activation of the fully connected layers,
        dropout:       dropout rate before the flatten layer,
        full_dropout:  dropout rate after every layer,
        batchnorm:     momentum of the batch normalisation,
        last_relu:     include ReLU in the output layers (if True),
        loss_weights:  the weights of the loss function,
        l1_reg:        l1 regularisation,
        l2_reg:        l2 regularisation,
        output_layers: include output layers (e.g. False when this is only part of a model).
        
    Returns:
        the Tensorflow model.
    '''
    keras.backend.clear_session()
    
    # input layer
    x = keras.layers.Input(shape=input_shape, name=name)
    I = {name: x}
    
    # placeholder for ResNet
    previous_concat = None
    
    # build convolutions
    for n in range(len(conv_layers)):
        # first branch
        a = x
        if conv_1b1:
            a = keras.layers.Conv2D(conv_layers[n],
                                    kernel_size=(1, 1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    name=name + '_convA1b1a_' + str(n)
                                   )(a)
            a = keras.layers.LeakyReLU(conv_alpha, name=name + '_actA1b1a_' + str(n))(a)
        
        a = keras.layers.Conv2D(conv_layers[n],
                                kernel_size=(x.shape[1], 1),
                                padding='same',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                name=name + '_convAa_' + str(n)
                               )(a)
        a = keras.layers.LeakyReLU(conv_alpha, name=name+ '_actAa_' + str(n))(a)
        
        # second branch
        b = x
        if conv_1b1:
            b = keras.layers.Conv2D(conv_layers[n],
                                    kernel_size=(1, 1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    name=name + '_convA1b1b_' + str(n)
                                   )(b)
            b = keras.layers.LeakyReLU(conv_alpha, name=name+ '_actA1b1b_' + str(n))(b)
        
        b = keras.layers.Conv2D(conv_layers[n],
                                kernel_size=(1, x.shape[2]),
                                padding='same',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                name=name + '_convAb_' + str(n)
                               )(b)
        b = keras.layers.LeakyReLU(conv_alpha, name=name+ '_actAb_' + str(n))(b)
        
        # classical convolution
        c = None
        if conv_class is not None:
            # factorise the kernel (5x5 ~ 3x3 + 3x3)
            if isinstance(conv_class, int):
                conv_class = (conv_class, conv_class)
            left  = int(conv_class[0] / 2) if conv_class[0] > 3 else conv_class[0]
            right = int(conv_class[1] / 2) if conv_class[1] > 3 else conv_class[1]
            
            # third branch
            c = x
            if conv_1b1:
                c = keras.layers.Conv2D(conv_layers[n],
                                        kernel_size=(1, 1),
                                        padding='same',
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        name=name + '_convB1b1_' + str(n)
                                       )(c)
                c = keras.layers.LeakyReLU(conv_alpha, name=name + '_actB1b1_' + str(n))(c)
            
            c = keras.layers.Conv2D(conv_layers[n],
                                    kernel_size=(left, 1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    name=name + '_convBa_' + str(n)
                                   )(c)
            c = keras.layers.LeakyReLU(conv_alpha, name=name + '_actBa_' + str(n))(c)
            
            c = keras.layers.Conv2D(conv_layers[n],
                                    kernel_size=(1, right),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                    name=name + '_convBb_' + str(n)
                                   )(c)
            c = keras.layers.LeakyReLU(conv_alpha, name=name + '_actBb_' + str(n))(c)
            
            # duplicate the convolution for large kernels (wider representation)
            if conv_class[0] > 3 and conv_class[1] > 3:
                v = keras.layers.Conv2D(conv_layers[n],
                                        kernel_size=(left, 1),
                                        padding='same',
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        name=name + '_convB2a_' + str(n)
                                       )(c)
                v = keras.layers.LeakyReLU(conv_alpha, name=name + '_actB2a_' + str(n))(v)
                
                w = keras.layers.Conv2D(conv_layers[n],
                                        kernel_size=(1, right),
                                        padding='same',
                                        kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        name=name + '_convB2b_' + str(n)
                                       )(c)
                w = keras.layers.LeakyReLU(conv_alpha, name=name + '_actB2b_' + str(n))(w)
                
                c = keras.layers.Concatenate(axis=-1, name=name + '_catB2_' + str(n))([v, w])
                
                if conv_1b1:
                    c = keras.layers.Conv2D(conv_layers[n],
                                            kernel_size=(1,1),
                                            padding='valid',
                                            kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                            name=name + '_convB21b1_' + str(n)
                                           )(c)
                    c = keras.layers.LeakyReLU(conv_alpha, name=name + '_actB21b1_' + str(n))(c)
                
        
        # fourth branch (1x1 convolution)
        d = None
        if conv_single:
            d = x
            d = keras.layers.Conv2D(conv_layers[n],
                                    kernel_size=(1,1),
                                    padding='same',
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
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
            x = keras.layers.Dropout(rate=full_dropout, name=name + '_convDrop_' + str(n))(x)
        
        # ResNet
        if previous_concat is not None and recurrent is not False:
            x = keras.layers.Concatenate(axis=-1, name=name + '_resCat_' + str(n))([x, previous_concat])
            if batchnorm > 0.0:
                x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_resBN_' + str(n))(x)
            if full_dropout > 0.0:
                x = keras.layers.Dropout(rate=full_dropout, name=name + '_resDrop_' + str(n))(x)
        previous_concat = x
    
    # 1x1 convolution to reduce the no. of filters
    if conv_red > 0:
        x = keras.layers.Conv2D(conv_red,
                                kernel_size=(1,1),
                                padding='valid',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                name=name + '_convRed'
                               )(x)
        x = keras.layers.LeakyReLU(conv_alpha, name=name + '_actRed')(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_redBN')(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, name=name + '_redDrop')(x)
        
    # flatten and dropout (add a feature map if needed)
    if feat_map > 0:
        x = keras.layers.Conv2D(feat_map,
                                kernel_size=(x.shape[1], x.shape[2]),
                                padding='valid',
                                kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                name=name + '_featmap'
                               )(x)
        x = keras.layers.Flatten(name=name + '_flat')(x)
        x = keras.layers.LeakyReLU(conv_alpha, name=name + '_flatAct')(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_flatBN')(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, name=name + '_flatDrop')(x)
    else:
        x = keras.layers.Flatten(name=name + '_flat')(x)
        
    if dropout > 0.0:
        x = keras.layers.Dropout(rate=dropout, name=name + '_drop')(x)
    
    # fully connected network
    for n in range(len(fc_layers)):
        x = keras.layers.Dense(fc_layers[n],
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               name=name + '_fc_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcact_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_' + str(n))(x)
        if full_dropout > 0.0:
            x = keras.layers.Dropout(rate=full_dropout, name=name + '_fcDrop_' + str(n))(x)
            
    # output layers
    if output_layers:
        if last_relu:
            O = {'h11': keras.layers.Dense(1, activation='relu', name='h11')(x),
                 'h21': keras.layers.Dense(1, activation='relu', name='h21')(x),
                 'h31': keras.layers.Dense(1, activation='relu', name='h31')(x),
                 'h22': keras.layers.Dense(1, activation='relu', name='h22')(x),
                }
        else:
            O = {'h11': keras.layers.Dense(1, name='h11')(x),
                 'h21': keras.layers.Dense(1, name='h21')(x),
                 'h31': keras.layers.Dense(1, name='h31')(x),
                 'h22': keras.layers.Dense(1, name='h22')(x),
                }
        
        # compile the model
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
             fc_layers=[32],
             fc_alpha=0.0,
             recurrent=False,
             dropout=0.2,
             batchnorm=0.99,
             last_relu=True,
             loss_weights=[0.25, 0.25, 0.25, 0.25],
             l1_reg=1.0e-5,
             l2_reg=1.0e-5,
             output_layers=True
            ):
    '''
    Create a Tensorflow model of a fully connected Network.
    
    Arguments:
        input_shape: the shape of the input tensor.
        
    Optional:
        name:          the name of the model,
        learning_rate: the shrinking parameter of gradient descent,
        fc_layers:     list of units in the fully connected network,
        fc_alpha:      slope of the LeakyReLU activation of the fully connected layers,
        recurrent:     make it a recurrent network,
        dropout:       dropout rate,
        batchnorm:     momentum of the batch normalisation,
        last_relu:     include ReLU in the output layers (if True),
        loss_weights:  the weights of the loss function,
        l1_reg:        l1 regularisation,
        l2_reg:        l2 regularisation,
        output_layers: include output layers (e.g. False when this is only part of a model).
        
    Returns:
        the Tensorflow model.
    '''
    keras.backend.clear_session()
    
    # input layer
    x = keras.layers.Input(shape=input_shape, name=name)
    I = {name: x}
    
    # placeholder for ResNet
    previous_concat = None
    
    # build convolutions
    for n in range(len(fc_layers)):
        x = keras.layers.Dense(fc_layers[n],
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               name=name + '_fc_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcAct_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_' + str(n))(x)
        if dropout > 0.0:
            x = keras.layers.Dropout(rate=dropout, name=name + '_fcDrop_' + str(n))(x)
        
        # ResNet
        if previous_concat is not None and recurrent is not False:
            x = keras.layers.Concatenate(axis=-1, name=name + '_catRes_' + str(n))([x, previous_concat])
            if batchnorm > 0.0:
                x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_resBN_' + str(n))(x)
            if dropout > 0.0:
                x = keras.layers.Dropout(rate=dropout, name=name + '_resDrop_' + str(n))(x)
        previous_concat = x
            
    # output layers
    if output_layers:
        if last_relu:
            O = {'h11': keras.layers.Dense(1, activation='relu', name='h11')(x),
                 'h21': keras.layers.Dense(1, activation='relu', name='h21')(x),
                 'h31': keras.layers.Dense(1, activation='relu', name='h31')(x),
                 'h22': keras.layers.Dense(1, activation='relu', name='h22')(x),
                }
        else:
            O = {'h11': keras.layers.Dense(1, name='h11')(x),
                 'h21': keras.layers.Dense(1, name='h21')(x),
                 'h31': keras.layers.Dense(1, name='h31')(x),
                 'h22': keras.layers.Dense(1, name='h22')(x),
                }
        
        # compile the model
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
            l1_reg=1.0e-5,
            l2_reg=1.0e-5
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
        batchnorm:     momentum of the batch normalisation,
        dropout:       dropout rate,
        last_relu:     add ReLU activation to output layer (if True),
        loss_weights:  list of loss weights.
    
    Returns:
        the full model.
    '''
    keras.backend.clear_session()
    
    # build inputs
    I = {model.name: model.input[model.name] for model in models}
    x = [model.output for model in models]
    x = keras.layers.Concatenate(axis=-1)(x)
    
    for n in range(len(fc_layers)):
        x = keras.layers.Dense(fc_layers[n],
                               kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                               name=name + '_fc_' + str(n)
                              )(x)
        x = keras.layers.LeakyReLU(fc_alpha, name=name + '_fcAct_' + str(n))(x)
        if batchnorm > 0.0:
            x = keras.layers.BatchNormalization(momentum=batchnorm, name=name + '_fcBN_' + str(n))(x)
        if dropout > 0.0:
            x = keras.layers.Dropout(rate=dropout, name=name + '_fcDrop_' + str(n))(x)
            
    # output layers
    if last_relu:
        O = {'h11': keras.layers.Dense(1, activation='relu', name='h11')(x),
             'h21': keras.layers.Dense(1, activation='relu', name='h21')(x),
             'h31': keras.layers.Dense(1, activation='relu', name='h31')(x),
             'h22': keras.layers.Dense(1, activation='relu', name='h22')(x),
            }
    else:
        O = {'h11': keras.layers.Dense(1, name='h11')(x),
             'h21': keras.layers.Dense(1, name='h21')(x),
             'h31': keras.layers.Dense(1, name='h31')(x),
             'h22': keras.layers.Dense(1, name='h22')(x),
            }
    
    # build model
    model = keras.models.Model(inputs=I, outputs=O, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  loss_weights=loss_weights,
                  metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
                 )
    
    return model