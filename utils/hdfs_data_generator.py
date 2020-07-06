import os
import sys
import numpy as np
import h5py
import csv
import time
import logging
from dataloader import KVReader

from utilities import int16_to_float32


class HdfsAudioSetDataset(object):
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
            audio_name_hdfs_path = meta['hdf5_path'] + '/audio_names'
            waveforms_hdfs_path = meta['hdf5_path'] + '/waveforms'
            targets_hdfs_path = meta['hdf5_path'] + '/targets'
            index_in_hdf5 = meta['index_in_hdf5']

            num_parallel_reader = 1
            waveform_reader = KVReader(waveforms_hdfs_path, num_parallel_reader)
            waveform = waveform_reader.read_many([str(index_in_hdf5)])
            waveform = int16_to_float32(np.frombuffer(waveform[0], dtype=np.int16))

            target_reader = KVReader(targets_hdfs_path, num_parallel_reader)
            target = target_reader.read_many([str(index_in_hdf5)])
            target = np.frombuffer(target[0], dtype=np.bool).astype(np.float32)

            audio_name_reader = KVReader(audio_name_hdfs_path, num_parallel_reader)
            audio_name = target_reader.read_many([str(index_in_hdf5)])[0].decode()

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict