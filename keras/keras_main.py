import torch
import torch.utils.data
from tensorflow import keras
# import keras

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as cPickle

from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer)
from models import *
from data_generator import (AudioSetDataset, BalancedSampler, BalancedSamplerMixup, 
    EvaluateSampler, Collator)
from evaluate import Evaluator
import config
# from losses import get_loss_func


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'unbalanced_train'
      frames_per_second: int
      mel_bins: int
      model_type: str
      loss_type: 'bce'
      balanced: bool
      augmentation: str
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    # dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 0
    sample_rate = config.sample_rate
    audio_length = config.audio_length
    classes_num = config.classes_num
    assert loss_type == 'clip_bce'

    # Paths
    black_list_csv = os.path.join(workspace, 'black_list', 'dcase2017task4.csv')
    
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'waveforms')

    # Target hdf5 path
    eval_train_targets_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'targets', 'balanced_train.h5')

    eval_test_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
        'eval.h5')

    if data_type == 'balanced_train':
        train_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
            'balanced_train.h5')
    elif data_type == 'full_train':
        train_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
            'full_train.h5')
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    model = Cnn13(audio_length, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
    model.summary()
    logging.info('Parameters number: {}'.format(model.count_params()))

    # Optimizer
    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)

    # Loss
    loss = keras.losses.binary_crossentropy

    model.compile(loss=loss, optimizer=optimizer)

    # Dataset will be used by DataLoader later. Provide an index and return 
    # waveform and target of audio
    train_dataset = AudioSetDataset(
        target_hdf5_path=train_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    bal_dataset = AudioSetDataset(
        target_hdf5_path=eval_train_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    test_dataset = AudioSetDataset(
        target_hdf5_path=eval_test_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    # Sampler
    if balanced == 'balanced':
        if 'mixup' in augmentation:
            train_sampler = BalancedSamplerMixup(
                target_hdf5_path=train_targets_hdf5_path, 
                black_list_csv=black_list_csv, batch_size=batch_size, 
                start_mix_epoch=1)
            train_collector = Collator(mixup_alpha=1.)
            assert batch_size % torch.cuda.device_count() == 0, 'To let mixup working properly this must be satisfied.'
        else:
            train_sampler = BalancedSampler(
                target_hdf5_path=train_targets_hdf5_path, 
                black_list_csv=black_list_csv, batch_size=batch_size)
            train_collector = Collator(mixup_alpha=None)
    
    bal_sampler = EvaluateSampler(dataset_size=len(bal_dataset), 
        batch_size=batch_size)

    test_sampler = EvaluateSampler(dataset_size=len(test_dataset), 
        batch_size=batch_size)

    eval_collector = Collator(mixup_alpha=None)
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=train_collector, 
        num_workers=num_workers, pin_memory=True)
    
    bal_loader = torch.utils.data.DataLoader(dataset=bal_dataset, 
        batch_sampler=bal_sampler, collate_fn=eval_collector, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_sampler=test_sampler, collate_fn=eval_collector, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    bal_evaluator = Evaluator(
        model=model, 
        generator=bal_loader)
    
    test_evaluator = Evaluator(
        model=model, 
        generator=test_loader)
    
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_weights_path = os.path.join(checkpoints_dir, '{}_iterations.weights.h5'.format(resume_iteration))
        resume_sampler_path = os.path.join(checkpoints_dir, '{}_iterations.sampler.h5'.format(resume_iteration))
        iteration = resume_iteration

        model.load_weights(resume_weights_path)
        sampler_state_dict = cPickle.load(open(resume_sampler_path, 'rb'))
        train_sampler.load_state_dict(sampler_state_dict)
        statistics_container.load_state_dict(resume_iteration)
        
    else:
        iteration = 0

    
    t_ = time.time()
    
    for batch_data_dict in train_loader:

        # Evaluate
        if (iteration % 2000 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = bal_evaluator.evaluate()
            test_statistics = test_evaluator.evaluate()
                            
            logging.info('Validate bal mAP: {:.3f}'.format(
                np.mean(bal_statistics['average_precision'])))

            logging.info('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))

            statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Save model
        # if iteration % 20000 == 0 and iteration > resume_iteration:
        if iteration == 10:
            weights_path = os.path.join(
                checkpoints_dir, '{}_iterations.weights.h5'.format(iteration))

            sampler_path = os.path.join(
                checkpoints_dir, '{}_iterations.sampler.h5'.format(iteration))
                
            model.save_weights(weights_path)
            cPickle.dump(train_sampler.state_dict(), open(sampler_path, 'wb'))

            logging.info('Model weights saved to {}'.format(weights_path))
            logging.info('Sampler saved to {}'.format(sampler_path))

        
        '''
        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], batch_data_dict['mixup_lambda'])
            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda'])}
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            batch_target_dict = {'target': batch_data_dict['target']}
        '''

        loss = model.train_on_batch(x=batch_data_dict['waveform'], y=batch_data_dict['target'])
        print(iteration, loss)
        
        iteration += 1

        # Stop learning
        if iteration == early_stop:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--balanced', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--dataset_dir', type=str, required=True)    
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--frames_per_second', type=int, required=True)
    parser_inference.add_argument('--mel_bins', type=int, required=True)
    parser_inference.add_argument('--model_type', type=str, required=True)    
    parser_inference.add_argument('--balanced', type=str, required=True)
    parser_inference.add_argument('--augmentation', type=str, required=True)
    parser_inference.add_argument('--batch_size', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'calculate_scalar':
        calculate_scalar(args)

    elif args.mode == 'train':
        train(args)

    elif args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error argument!')
