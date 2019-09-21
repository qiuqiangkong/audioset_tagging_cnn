import os
import sys
import numpy as np
import h5py
import csv
import time
import logging

from utilities import int16_to_float32


def read_black_list(black_list_csv):
    """Read audio names from black list. 
    """
    with open(black_list_csv, 'r') as fr:
        reader = csv.reader(fr)
        lines = list(reader)

    black_list_names = ['Y{}.wav'.format(line[0]) for line in lines]
    return black_list_names



class AudioSetDataset(object):
    def __init__(self, target_hdf5_path, waveform_hdf5s_dir, audio_length, classes_num):
        """AduioSet dataset for later used by DataLoader. This class takes an 
        audio index as input and output corresponding waveform and target. 
        """
        self.waveform_hdf5s_dir = waveform_hdf5s_dir
        self.audio_length = audio_length
        self.classes_num = classes_num

        with h5py.File(target_hdf5_path, 'r') as hf:
            """
            {'audio_name': (audios_num,) e.g. ['YtwJdQzi7x7Q.wav', ...], 
             'waveform': (audios_num, audio_length), 
             'target': (audios_num, classes_num)}
             """
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_names = [hdf5_name.decode() for hdf5_name in hf['hdf5_name'][:]]
            self.indexes_in_part_hdf5 = hf['index_in_hdf5'][:]
       
        logging.info('Audio samples: {}'.format(len(self.audio_names)))
 
    def get_relative_hdf5_path(self, hdf5_name):
        if hdf5_name in ['balanced_train.h5', 'eval.h5']:
            return hdf5_name
        elif 'unbalanced_train' in hdf5_name:
            relative_path = os.path.join('unbalanced_train', hdf5_name)
        else:
            raise Exception('Incorrect hdf5_name!')

        return relative_path
    
    def __getitem__(self, index):
        """Load waveform and target of the audio index. If index is -1 then 
            return None. 
        
        Returns: {'audio_name': str, 'waveform': (audio_length,), 'target': (classes_num,)}
        """
        if index == -1:
            audio_name = None
            waveform = np.zeros((self.audio_length,), dtype=np.float32)
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            audio_name = self.audio_names[index]
            hdf5_name = self.hdf5_names[index]
            index_in_part_hdf5 = self.indexes_in_part_hdf5[index]

            relative_hdf5_path = self.get_relative_hdf5_path(hdf5_name)
            hdf5_path = os.path.join(self.waveform_hdf5s_dir, relative_hdf5_path)

            with h5py.File(hdf5_path, 'r') as hf:
                audio_name = hf['audio_name'][index_in_part_hdf5].decode()
                waveform = int16_to_float32(hf['waveform'][index_in_part_hdf5])
                target = hf['target'][index_in_part_hdf5].astype(np.float32)
                
        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict
    
    def __len__(self):
        return len(self.audio_names)


class BalancedSampler(object):

    def __init__(self, target_hdf5_path, black_list_csv, batch_size, 
        random_seed=1234, verbose=1):
        """Balanced sampler. Generate audio indexes for DataLoader. 
        
        Args:
          target_hdf5_path: string
          black_list_csv: string
          batch_size: int
          start_mix_epoch: int, only do mix up after this samples have been 
            trained after this times. 
        """

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        self.black_list_names = read_black_list(black_list_csv)
        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()
        with h5py.File(target_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.target = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.target.shape
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))
        
        self.samples_num_per_class = np.sum(self.target, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int64)))
        
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.target[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate audio indexes for training. 
        
        Returns: batch_indexes: (batch_size,). 
        """
        batch_size = self.batch_size

        while True:
            batch_indexes = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                audio_index = self.indexes_per_class[class_id][pointer]
                
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_names[audio_index] in self.black_list_names:
                    continue
                else:
                    batch_indexes.append(audio_index)
                    i += 1

            yield batch_indexes

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


class BalancedSamplerMixup(object):

    def __init__(self, target_hdf5_path, black_list_csv, batch_size, 
        start_mix_epoch, random_seed=1234, verbose=1):
        """Balanced sampler. Generate audio indexes for DataLoader. 
        
        Args:
          target_hdf5_path: string
          black_list_csv: string
          batch_size: int
          start_mix_epoch: int, only do mix up after this samples have been 
            trained after this times. 
        """

        self.batch_size = batch_size
        self.start_mix_epoch = start_mix_epoch
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        self.black_list_names = read_black_list(black_list_csv)
        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()
        with h5py.File(target_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.target = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.target.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))
        
        self.samples_num_per_class = np.sum(self.target, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int64)))
        
        # Two indexes list are used for containing sample indexes used for mixup. 
        # Each indexes list looks like: [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...], 
        # which means class 1 contains audio indexes of [0, 11, 12], 
        # class 2 contains audio indexes of [3, 4, 15, 16], ...
        self.indexes1_per_class = []
        self.indexes2_per_class = []
        
        for k in range(self.classes_num):
            self.indexes1_per_class.append(
                np.where(self.target[:, k] == 1)[0])
                
            self.indexes2_per_class.append(
                np.where(self.target[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes1_per_class[k])
            self.random_state.shuffle(self.indexes2_per_class[k])
        
        # Queue containing sound classes to be selected
        self.queue1 = []
        self.queue2 = []
        self.pointers1_of_classes = [0] * self.classes_num
        self.pointers2_of_classes = [0] * self.classes_num
        self.finished_epochs_per_class = [0] * self.classes_num

    # def get_classes_set(self):
    #     return np.arange(self.classes_num).tolist()
    '''
    def __iter__(self):
        """Generate audio indexes for training. 
        
        Returns: batch_indexes: (batch_size * 2,). 
            Data from batch_indexes[0 : batch_size] will be mixup with batch_indexes[batch_size : batch_size * 2]
            index can be -1 in batch_indexes[batch_size : batch_size * 2] indicating no mixup. 
        """
        batch_size = self.batch_size

        while True:
            # If queue1 is not long enough then append more classes
            while len(self.queue1) < batch_size:
                classes_set = self.get_classes_set()
                self.random_state.shuffle(classes_set)
                self.queue1 += classes_set
                
            # Fetch classes from queue1
            batch_class_ids = self.queue1[0 : batch_size]
            self.queue1[0 : batch_size] = []
                
            batch_samples_num_per_class = [
                batch_class_ids.count(k) for k in range(self.classes_num)]
                
            batch_indexes1 = []
            batch_indexes2 = []

            # Get indexes from each class
            for k in range(self.classes_num):
                bgn_pointer = self.pointers1_of_classes[k]
                fin_pointer = self.pointers1_of_classes[k] + \
                    batch_samples_num_per_class[k]
                    
                self.pointers1_of_classes[k] += batch_samples_num_per_class[k]
                audio_idxes = self.indexes1_per_class[k][bgn_pointer : fin_pointer]
                
                # Remove indexes appeared in black list
                for audio_idx in audio_idxes:
                    if self.audio_names[audio_idx] in self.black_list_names:
                        audio_idxes = np.delete(audio_idxes, np.argwhere(audio_idxes == audio_idx))
                
                for audio_idx in audio_idxes:
                    batch_indexes1.append(audio_idx)

                # If exceed some epochs then prepare batch_indexes2 for mixup
                if self.finished_epochs_per_class[k] >= self.start_mix_epoch:
                    # Index array should be the same length as idxes
                    j = 0
                    while j < len(audio_idxes):
                        # If queue2 is not long enough then append more indexes
                        while len(self.queue2) < batch_size:
                            classes_set = self.get_classes_set()
                            self.random_state.shuffle(classes_set)
                            self.queue2 += classes_set
                            
                        # Fetch one class
                        k2 = self.queue2[0]
                        self.queue2[0 : 1] = []
                        audio_idx = self.indexes2_per_class[k2][self.pointers2_of_classes[k2]]

                        self.pointers2_of_classes[k2] += 1

                        if self.pointers2_of_classes[k2] >= self.samples_num_per_class[k2]:
                            self.pointers2_of_classes[k2] = 0
                            self.random_state.shuffle(self.indexes2_per_class[k2])

                        if self.audio_names[audio_idx] in self.black_list_names:
                            continue
                        else:
                            batch_indexes2.append(audio_idx)
                            j += 1

                else:
                    for _ in range(len(audio_idxes)):
                        batch_indexes2.append(-1)

                if self.pointers1_of_classes[k] >= self.samples_num_per_class[k]:
                    self.finished_epochs_per_class[k] += 1
                    self.pointers1_of_classes[k] = 0
                    self.random_state.shuffle(self.indexes1_per_class[k])
                    
            batch_indexes = np.array(batch_indexes1 + batch_indexes2)

            yield batch_indexes
    '''

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate audio indexes for training. 
        
        Returns: batch_indexes: (batch_size * 2,). 
            Data from batch_indexes[0 : batch_size] will be mixup with batch_indexes[batch_size : batch_size * 2]
            index can be -1 in batch_indexes[batch_size : batch_size * 2] indicating no mixup. 
        """
        batch_size = self.batch_size

        while True:
            batch_indexes = []
            i = 0
            while i < batch_size:
                if len(self.queue1) == 0:
                    self.queue1 = self.expand_queue(self.queue1)

                class1_id = self.queue1.pop(0)
                pointer1 = self.pointers1_of_classes[class1_id]
                self.pointers1_of_classes[class1_id] += 1
                audio1_index = self.indexes1_per_class[class1_id][pointer1]
                
                if self.pointers1_of_classes[class1_id] >= self.samples_num_per_class[class1_id]:
                    self.finished_epochs_per_class[class1_id] += 1
                    self.pointers1_of_classes[class1_id] = 0
                    self.random_state.shuffle(self.indexes1_per_class[class1_id])

                # If audio in black list then continue
                if self.audio_names[audio1_index] in self.black_list_names:
                    continue
                else:
                    batch_indexes.append(audio1_index)
                    i += 1

                # If exceed some epochs then get audio for mixup
                if self.finished_epochs_per_class[class1_id] >= self.start_mix_epoch:
                    j = 0
                    while j < 1:
                        if len(self.queue2) == 0:
                            self.queue2 = self.expand_queue(self.queue2)

                        class2_id = self.queue2.pop(0)
                        pointer2 = self.pointers2_of_classes[class2_id]
                        self.pointers2_of_classes[class2_id] += 1
                        audio2_index = self.indexes2_per_class[class2_id][pointer2]

                        if self.pointers2_of_classes[class2_id] >= self.samples_num_per_class[class2_id]:
                            self.pointers2_of_classes[class2_id] = 0
                            self.random_state.shuffle(self.indexes2_per_class[class2_id])

                        # If audio in black list then continue
                        if self.audio_names[audio2_index] in self.black_list_names:
                            continue
                        else:
                            batch_indexes.append(audio2_index)
                            j += 1
                else:
                    batch_indexes.append(-1)

            yield batch_indexes
    

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'indexes1_per_class': self.indexes1_per_class, 
            'indexes2_per_class': self.indexes2_per_class, 
            'queue1': self.queue1, 
            'queue2': self.queue2, 
            'pointers1_of_classes': self.pointers1_of_classes, 
            'pointers2_of_classes': self.pointers2_of_classes, 
            'finished_epochs_per_class': self.finished_epochs_per_class}
        return state
            
    def load_state_dict(self, state):
        self.indexes1_per_class = state['indexes1_per_class']
        self.indexes2_per_class = state['indexes2_per_class']
        self.queue1 = state['queue1']
        self.queue2 = state['queue2']
        self.pointers1_of_classes = state['pointers1_of_classes']
        self.pointers2_of_classes = state['pointers2_of_classes']
        self.finished_epochs_per_class = state['finished_epochs_per_class']


class EvaluateSampler(object):

    def __init__(self, dataset_size, batch_size):
        """Inference sampler. Generate audio indexes for DataLoader. 
        
        Args:
          batch_size: int
        """
        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __iter__(self):
        """Generate audio indexes for evaluation.
        
        Returns: batch_indexes: (batch_size,)
        """
        batch_size = self.batch_size

        pointer = 0

        while pointer < self.dataset_size:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.dataset_size))

            pointer += batch_size
            yield batch_indexes


class Collator(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Data collator.
        
        Args:
          mixup: bool
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)
        
    
    def __call__(self, list_data_dict):
        """Collate data to tensor. Add mixup information to list_data_dict. 
        
        Args:
          list_data_dict: 
            [{'audio_name': 'YtwJdQzi7x7Q.wav', 'waveform': (audio_length,), 'target': (classes_num)}, 
            ...]

        Returns:
          np_data_dict: {
            'audio_name': (audios_num,), 
            'waveform': (audios_num, audio_length), 
            'target': (audios_num, classes_num), 
            (optional) 'mixup_lambda': (audios_num,)}
        """
        np_data_dict = {}
        
        if self.mixup_alpha:

            mixup_lambdas = []
            for n in range(1, len(list_data_dict), 2):
                if list_data_dict[n]['audio_name'] is None:
                    lam = 1.
                else:
                    lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
                
                mixup_lambdas.append(lam)
                mixup_lambdas.append(1. - lam)
                
            np_data_dict['mixup_lambda'] = np.array(mixup_lambdas)
        
        np_data_dict['audio_name'] = np.array([data_dict['audio_name'] for data_dict in list_data_dict])
        np_data_dict['waveform'] = np.array([data_dict['waveform'] for data_dict in list_data_dict])
        np_data_dict['target'] = np.array([data_dict['target'] for data_dict in list_data_dict])

        return np_data_dict