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
from dataloader import KVWriter

from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate, read_metadata)
import config


def pack_waveforms_to_hdfs(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    csv_path = args.csv_path
    waveforms_hdfs_path = args.waveforms_hdfs_path
    mini_data = args.mini_data

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    sample_rate = config.sample_rate
    id_to_ix = config.id_to_ix

    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdfs_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdfs_path))

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    
    # Read csv file
    meta_dict = read_metadata(csv_path, classes_num, id_to_ix)

    if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]

    audios_num = len(meta_dict['audio_name'])

    values = []

    for n in range(audios_num):
        audio_path = os.path.join(audios_dir, meta_dict['audio_name'][n])

        if os.path.isfile(audio_path):
            logging.info('{} {}'.format(n, audio_path))
            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            audio = pad_or_truncate(audio, clip_samples)

            values.append({'audio_name': meta_dict['audio_name'][n].encode(), 
                'waveform': float32_to_int16(audio), 
                'target': meta_dict['target'][n]})

        else:
            logging.info('{} File does not exist! {}'.format(n, audio_path))

    import crash
    asdf

    num_shard = 1
    writer = KVWriter(waveforms_hdf5_path, num_shard)
    writer.write_many(keys, values)
    writer.flush() # Make sure to flush at the end

    # Pack waveform to hdf5
    total_time = time.time()

    # Pack waveform & target of several audio clips to a single hdf5 file
    

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_wavs = subparsers.add_parser('pack_waveforms_to_hdfs')
    parser_pack_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_pack_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_pack_wavs.add_argument('--waveforms_hdfs_path', type=str, required=True, help='Path to save out packed hdf5.')
    parser_pack_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')

    args = parser.parse_args()
    
    if args.mode == 'pack_waveforms_to_hdfs':
        pack_waveforms_to_hdfs(args)

    else:
        raise Exception('Incorrect arguments!')