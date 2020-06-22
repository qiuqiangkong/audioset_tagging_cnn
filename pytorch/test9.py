import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import pickle
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utilities import (create_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout, 
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19, 
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128, 
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup, forward)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler, 
    AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func


def get_train_sampler(balanced):
    """Get train sampler.

    Args:
      balanced: str
      augmentation: str
      train_indexes_hdf5_path: str
      black_list_csv: str
      batch_size: int

    Returns:
      train_sampler: object
      train_collector: object
    """
    if balanced == 'none':
        _Sampler = TrainSampler
    elif balanced == 'balanced':
        _Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        _Sampler = AlternateTrainSampler


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'unbalanced_train'
      frames_per_second: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    batch_size = args.batch_size
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    checkpoint_path = args.checkpoint_path

    num_workers = 8
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num

    eval_bal_indexes_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'indexes', 'balanced_train.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        'eval.h5')
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(clip_samples=clip_samples, classes_num=classes_num)

    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluate

    # bal_statistics = evaluator.evaluate(eval_bal_loader)
    # test_statistics = evaluator.evaluate(eval_test_loader)
                    
    # logging.info('Validate bal mAP: {:.3f}'.format(
    #     np.mean(bal_statistics['average_precision'])))

    # logging.info('Validate test mAP: {:.3f}'.format(
    #     np.mean(test_statistics['average_precision'])))

    output_dict = forward(
        model=model, 
        generator=eval_test_loader, 
        return_target=True)

    clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
    target = output_dict['target']    # (audios_num, classes_num)

    opt_thres = []

    for k in range(classes_num):        
        (precision, recall, thresholds) = metrics.precision_recall_curve(target[:, k], clipwise_output[:, k])
        for i in range(len(thresholds)):
            if precision[i] >+ recall[i]:
                break
        opt_thres.append(thresholds[i])

    pickle.dump(opt_thres, open('opt_thres.pkl', 'wb'))
    
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--checkpoint_path', type=str, required=True)

    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')