from tensorflow import keras
import numpy as np
import time
import os

class PrintCheckpoint(keras.callbacks.Callback):
    '''
    Print status of the training after a given number of epochs.
    '''
    
    def __init__(self, interval):
        '''
        Required arguments:
            interval: print status after given interval of epochs.
        '''
        super(PrintCheckpoint, self).__init__()
        self.interval    = int(interval)
        self.epoch_times = []
        
    def on_train_begin(self, logs=None):
        if self.interval > 1:
            print(f'Training has started. Callouts will be printed every {self.interval:d} epochs.', flush=True)
        else:
            print(f'Training has started. Callouts will be printed every epoch.', flush=True)
                
        self.train_time = time.time()
        
    def on_train_end(self, logs=None):
        self.train_time = time.time() - self.train_time
        self.train_time = time.gmtime(self.train_time)
        self.train_time = time.strftime('%H hours, %M minutes, %S seconds', self.train_time)
        now = time.strftime('%d/%m/%Y at %H:%M:%S', time.localtime())
        print(f'\nTraining ended on {now} after {self.train_time}.', flush=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            self.epoch_time = time.time()
            now = time.strftime('%d/%m/%Y at %H:%M:%S', time.localtime())
            print(f'\nTraining epoch {epoch+1:d}. Started on {now}.\n', flush=True)
        
    def on_epoch_end(self, epoch, logs=None):
        
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            self.epoch_time = time.time() - self.epoch_time
            self.epoch_times.append(self.epoch_time)
            epoch_time = time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(np.mean(self.epoch_times)))
            print(f'    Average epoch training time: {epoch_time}\n', flush=True)
            for key, value in logs.items():
                print(f'    {key} = {value:.6f}')
                

def model_checkpoints(outputs, root='.', validation=True, reduce_lr=None, lr_patience=150, min_lr=1.0e-6, summary=1, tensorboard=None):
    '''
    Create a list of checkpoints for each output.
    
    Required arguments:
        outputs: list of outputs to checkpoints.
        
    Optional arguments:
        root:        root directory to save the models,
        validation:  whether to use validation losses (or training losses if False),
        reduce_lr:   learning rate reduction factor (if not None),
        lr_patience: patience of the learning rate reduction,
        min_lr:      minimum learning rate,
        summary:     frequency of the summary statistics being printed on screen,
        tensorboard: the directory where to save the Tensorboard data.
        
    Returns:
        a list of keras.callbacks.
    '''
        
    # add '_loss' to the outputs
    outputs = ['loss'] + [output + '_loss' for output in outputs]
    
    # if validation, then add 'val_' in fron of the list
    if validation:
        outputs = ['val_' + output for output in outputs]
    
    # create the checkpoints
    checkpoints = []
    for output in outputs:
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(root, output + '.h5'), monitor=output, save_best_only=True)
        checkpoints.append(checkpoint)
        
    # add learning rate reduction
    if reduce_lr is not None:
        checkpoints.append(keras.callbacks.ReduceLROnPlateau(factor=reduce_lr, patience=lr_patience, min_lr=min_lr))
        
    # add summary
    if summary >= 1:
        checkpoints.append(PrintCheckpoint(summary))
        
    # add tensorboard
    if tensorboard is not None:
        
        os.makedirs(tensorboard, exist_ok=True)
        checkpoints.append(keras.callbacks.TensorBoard(log_dir=tensorboard,
                                                       histogram_freq=1,
                                                       write_graph=True,
                                                       write_images=True,
                                                       update_freq='epoch',
                                                       profile_batch=2,
                                                       embeddings_freq=0,
                                                       embeddings_metadata=None
                                                      )
                          )
        
    return checkpoints