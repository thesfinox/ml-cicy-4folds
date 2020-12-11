import os
import time
from tensorflow import keras

def dir_struct(prefix, dirs=['img', 'models'], base='.'):
    '''
    Create a directory structure with subdirectories.
    
    Needed arguments:
        prefix: name of the prefix of the subdirectories.
    
    Optional arguments:
        dirs: list of root directory names to be created,
        base: base name of the structure.
        
    Returns:
        the subdirectory structure
    '''
    
    name      = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    structure = []
    for d in dirs:
        path = os.path.join(base, d, prefix + '_' + name)
        os.makedirs(path, exist_ok=False)
        structure.append(path)
        
    print('Current working directories:\n')
    for d in structure:
        print(f'  {d}')
    return structure


def list_models(outputs, root='.', validation=True):
    '''
    Find the list of saved models in a directory structure.
    
    Needed argumets:
        outputs: list of monitored outputs.
        
    Optional arguments:
        root:       the root directory,
        validation: whether validation losses were monitored (training if False).
        
    Returns:
        a dictionary of saved models.
    '''
        
    # add '_loss' to the outputs
    labels  = ['loss'] + [output + '_loss' for output in outputs]
    outputs = ['full_model'] + outputs
    
    # if validation, then add 'val_' in fron of the list
    if validation:
        labels = ['val_' + label for label in labels]
        
    # load the models
    models = {outputs[n]: keras.models.load_model(os.path.join(root, label + '.h5')) for n, label in enumerate(labels)}
    
    return models