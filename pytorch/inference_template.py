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
import librosa

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utilities import get_filename
from models import *
from pytorch_utils import (move_data_to_device, count_parameters, count_flops)
import config

"""
MODEL_TYPE="Cnn14"
CHECKPOINT_PATH="/vol/vssp/msos/qk/workspaces/pub_audioset_tagging_cnn_transfer/checkpoints_for_paper/Cnn14_mAP=0.431.pth"
CUDA_VISIBLE_DEVICES=1 python3 pytorch/inference_template.py inference --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --cuda
"""

def inference(args):

    # Arugments & parameters
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    sample_rate = config.sample_rate
    classes_num = config.classes_num

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
    
    if True:
        waveform = np.zeros(sample_rate * 10)
    else:
        audio_path = "/vol/vssp/msos/qk/test9/YwfSPbhnpOlQ.wav"
        (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, device)

    # Forward
    model.eval()
    batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
    print(embedding.shape)

    for k in range(10):
        print('{}, {}'.format(np.array(config.labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_inference = subparsers.add_parser('inference') 
    parser_inference.add_argument('--window_size', type=int, required=True)
    parser_inference.add_argument('--hop_size', type=int, required=True)
    parser_inference.add_argument('--mel_bins', type=int, required=True)
    parser_inference.add_argument('--fmin', type=int, required=True)
    parser_inference.add_argument('--fmax', type=int, required=True) 
    parser_inference.add_argument('--model_type', type=str, required=True)
    parser_inference.add_argument('--checkpoint_path', type=str, required=True)
    parser_inference.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error argument!')
