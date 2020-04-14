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
    def __init__(self, clip_samples, classes_num):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 

        Args:
          clip_samples: int
          classes_num: int
        """
        self.clip_samples = clip_samples
        self.classes_num = classes_num
    
    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        if meta is None:
            """Dummy waveform and target. This is used for samples with mixup 
            lamda of 0."""
            audio_name = None
            waveform = np.zeros((self.clip_samples,), dtype=np.float32)
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            hdf5_path = meta['hdf5_path']
            index_in_hdf5 = meta['index_in_hdf5']

            with h5py.File(hdf5_path, 'r') as hf:
                audio_name = hf['audio_name'][index_in_hdf5].decode()
                waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
                target = hf['target'][index_in_hdf5].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict


class Base(object):
    def __init__(self, indexes_hdf5_path, black_list_csv, batch_size, random_seed):
        """Base class of train sampler.
        
        Args:
          indexes_hdf5_path: string
          black_list_csv: string
          batch_size: int
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        self.black_list_names = read_black_list(black_list_csv)
        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.targets.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class Sampler(Base):
    def __init__(self, indexes_hdf5_path, black_list_csv, batch_size, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          black_list_csv: string
          batch_size: int
          random_seed: int
        """
        super(Sampler, self).__init__(indexes_hdf5_path, black_list_csv, 
            batch_size, random_seed)
        
        self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'audio_name': self.audio_names[index], 
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'target': self.targets[index]})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class BalancedSampler(Base):
    def __init__(self, indexes_hdf5_path, black_list_csv, batch_size, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are evenly 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          black_list_csv: string
          batch_size: int
          random_seed: int
        """
        super(BalancedSampler, self).__init__(indexes_hdf5_path, black_list_csv, 
            batch_size, random_seed)
        
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
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
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'audio_name': self.audio_names[index], 
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'target': self.targets[index]})
                    i += 1

            yield batch_meta

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


class BalancedMixupSampler(Base):
    def __init__(self, indexes_hdf5_path, black_list_csv, batch_size, 
        start_mix_epoch, random_seed=1234):
        """Balanced mixup sampler. Generate batch meta for training. Data are 
        evenly sampled from different sound classes. Data with even indexes 
        (0, 2, 4, ...) will be mixup with data of odd indexes (1, 3, 5, ...) in
        training.
        
        Args:
          indexes_hdf5_path: string
          black_list_csv: string
          batch_size: int
          start_mix_epoch: int, mixup data after a sound class has been trained 
              for this epochs.
          random_seed: int
        """
        super(BalancedMixupSampler, self).__init__(indexes_hdf5_path, black_list_csv, 
            batch_size, random_seed)

        self.start_mix_epoch = start_mix_epoch
        
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Two indexes list are used for containing sample indexes used for mixup. 
        # Each indexes list looks like: [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...], 
        # which means class 1 contains audio indexes of [0, 11, 12], 
        # class 2 contains audio indexes of [3, 4, 15, 16], ...
        self.indexes1_per_class = []
        self.indexes2_per_class = []
        
        for k in range(self.classes_num):
            self.indexes1_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
                
            self.indexes2_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
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

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        while True:
            batch_meta = []
            i = 0
            while i < self.batch_size // 2:
                if len(self.queue1) == 0:
                    self.queue1 = self.expand_queue(self.queue1)

                class1_id = self.queue1.pop(0)
                pointer1 = self.pointers1_of_classes[class1_id]
                self.pointers1_of_classes[class1_id] += 1
                index1 = self.indexes1_per_class[class1_id][pointer1]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers1_of_classes[class1_id] >= self.samples_num_per_class[class1_id]:
                    self.finished_epochs_per_class[class1_id] += 1
                    self.pointers1_of_classes[class1_id] = 0
                    self.random_state.shuffle(self.indexes1_per_class[class1_id])

                # If audio in black list then continue
                if self.audio_names[index1] in self.black_list_names:
                    continue
                else:
                    # batch_indexes.append(index1)
                    batch_meta.append({
                        'audio_name': self.audio_names[index1], 
                        'hdf5_path': self.hdf5_paths[index1], 
                        'index_in_hdf5': self.indexes_in_hdf5[index1], 
                        'target': self.targets[index1]})
                    i += 1

                # Append meta if a sound class has been trained after 
                # start_mix_epoch epochs, otherwise append None.
                if self.finished_epochs_per_class[class1_id] >= self.start_mix_epoch:
                    j = 0
                    while j < 1:
                        if len(self.queue2) == 0:
                            self.queue2 = self.expand_queue(self.queue2)

                        class2_id = self.queue2.pop(0)
                        pointer2 = self.pointers2_of_classes[class2_id]
                        self.pointers2_of_classes[class2_id] += 1
                        index2 = self.indexes2_per_class[class2_id][pointer2]

                        if self.pointers2_of_classes[class2_id] >= self.samples_num_per_class[class2_id]:
                            self.pointers2_of_classes[class2_id] = 0
                            self.random_state.shuffle(self.indexes2_per_class[class2_id])

                        # If audio in black list then continue
                        if self.audio_names[index2] in self.black_list_names:
                            continue
                        else:
                            batch_meta.append({
                                'audio_name': self.audio_names[index2], 
                                'hdf5_path': self.hdf5_paths[index2], 
                                'index_in_hdf5': self.indexes_in_hdf5[index2], 
                                'target': self.targets[index2]})
                            j += 1
                else:
                    batch_meta.append(None)

            yield batch_meta

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
    def __init__(self, indexes_hdf5_path, batch_size):
        """Evaluate sampler. Generate batch meta for evaluation.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
        """
        self.batch_size = batch_size

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            
        self.audios_num = len(self.audio_names)

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'Y--PJHxphWEs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 0, 'target':
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'audio_name': self.audio_names[index], 
                    'hdf5_path': self.hdf5_paths[index], 
                    'index_in_hdf5': self.indexes_in_hdf5[index], 
                    'target': self.targets[index]})

            pointer += batch_size
            yield batch_meta


class Collator(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Collate data to a mini-batch.
        
        Args:
          mixup_alpha: float
          random_seed: int
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)
    
    def __call__(self, list_data_dict):
        """Collate list of data to a mini-batch.
        
        Args:
          list_data_dict: 
            [{'audio_name': 'YtwJdQzi7x7Q.wav', 'waveform': (clip_samples,), 'target': (classes_num)}, 
             ...]

        Returns:
          np_data_dict: {
            'audio_name': (batch_size,), 
            'waveform': (batch_size, clip_samples), 
            'target': (batch_size, classes_num), 
            (optional) 'mixup_lambda': (batch_size,)}
        """
        np_data_dict = {}
        
        if self.mixup_alpha:
            """Data with even indexes (0, 2, 4, ...) will be mixup with data 
            with odd indexes (1, 3, 5, ...)."""
            mixup_lambdas = []
            for n in range(0, len(list_data_dict), 2):
                if list_data_dict[n + 1]['audio_name'] is None:
                    lam = 1.
                else:
                    lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
                
                mixup_lambdas.append(lam)
                mixup_lambdas.append(1. - lam)
                
            np_data_dict['mixup_lambda'] = np.array(mixup_lambdas)

        else:
            pass
        
        np_data_dict['audio_name'] = np.array([data_dict['audio_name'] for data_dict in list_data_dict])
        np_data_dict['waveform'] = np.array([data_dict['waveform'] for data_dict in list_data_dict])
        np_data_dict['target'] = np.array([data_dict['target'] for data_dict in list_data_dict])

        return np_data_dict