import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from utilities import create_folder
import config


def create_partial_indexes(args):

    # Arguments & parameters
    full_indexes_hdf5_path = args.full_indexes_hdf5_path
    partial_indexes_hdf5_path = args.partial_indexes_hdf5_path
    partial = args.partial

    random_state = np.random.RandomState(1234)

    with h5py.File(full_indexes_hdf5_path, 'r') as hf:
        new_hf = h5py.File(partial_indexes_hdf5_path, 'w')

        audios_num = hf['audio_name'].shape[0]
        indexes = np.arange(audios_num)
        random_state.shuffle(indexes)
        partial_indexes = indexes[0 : int(audios_num * partial)]
        partial_indexes = np.sort(partial_indexes)

        for key in hf.keys():
            new_hf.create_dataset(key, data=hf[key][:][partial_indexes], dtype=hf[key].dtype)

        new_hf.close()

    print('Write indexes to {}'.format(partial_indexes_hdf5_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    parser.add_argument('--full_indexes_hdf5_path', type=str, required=True)
    parser.add_argument('--partial_indexes_hdf5_path', type=str, required=True)
    parser.add_argument('--partial', type=float, required=True)

    args = parser.parse_args()

    create_partial_indexes(args)