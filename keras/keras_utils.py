import time
import numpy as np
import h5py


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
    

def forward(model, generator, return_input=False, return_target=False):
    '''Forward data to model in mini-batch. 
    
    Args: 
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up validation
    '''
    output_dict = {}

    # Evaluate on mini-batch
    for n, batch_data_dict in enumerate(generator):
        print(n)

        # Predict
        batch_output = model.predict(batch_data_dict['waveform'])

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])
        append_to_dict(output_dict, 'output', batch_output)
            
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])
                
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def save_dict(dict, path):
    with h5py.File(path, 'w') as hf:
        for key in dict.keys():
            hf.create_dataset(key, data=dict[key], dtype=dict[key].dtype)

def load_dict(path):
    dict = {}
    with h5py.File('data.h5', 'w') as hf:
        for key in hf.keys():
            dict[key] = hf[key][:]    
    return dict 
