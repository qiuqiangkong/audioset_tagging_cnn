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

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer)
from models import *
from models2 import *
from models3 import *
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, BalancedSampler, BalancedSamplerMixup, 
    EvaluateSampler, Collator)
from evaluate import Evaluator
import config
from losses import get_loss_func


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

    num_workers = 8
    sample_rate = config.sample_rate
    audio_length = config.audio_length
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

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
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
     
    params_num = count_parameters(model)
    # flops_num = count_flops(model, audio_length)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
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
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    t_ = time.time()
    
    for batch_data_dict in train_loader:
        
        """batch_list_data_dict: 
            [{'audio_name': 'YtwJdQzi7x7Q.wav', 'waveform': (audio_length,), 'target': (classes_num)}, 
            ...]"""
        
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
        if iteration % 20000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], batch_data_dict['mixup_lambda'])
            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda'])}
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            batch_target_dict = {'target': batch_data_dict['target']}

        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        print(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print(iteration, 'time: {:.3f}'.format(time.time() - t_))
            t_ = time.time()
        
        iteration += 1

        # Stop learning
        if iteration == early_stop:
            break


def inference(args):
    """Inference.
    """

    # Arugments & parameters
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = config.sample_rate
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    classes_num = config.classes_num

    if 'cuda' in str(device):
        device = 'cuda'
    else:
        device = 'cpu'

    # Paths
    '''
    Model = Cnn13
    checkpoints_path = "/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
    '''
    Model = Cnn13_DecisionLevelMax
    checkpoints_path = "/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_DecisionLevelMax/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/400000_iterations.pth"

    # Model
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)

    checkpoint = torch.load(checkpoints_path)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
 
    data = np.zeros((1, 32000))
    # data = np.sin(np.arange(320000)/440 * 6.28)[None, :]
    data = move_data_to_device(data, device)

    inference_time = time.time()
    output_dict = model(data, None)
    clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()[0]

    sorted_indexes = np.argsort(clipwise_output)[::-1][0:10]
    clipwise_output[sorted_indexes]
    np.array(config.labels)[sorted_indexes]

    print('Inference time: {} s'.format(time.time() - inference_time))
    import crash
    asdf


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
