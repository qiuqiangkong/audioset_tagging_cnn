import os
import sys
import numpy as np
import argparse
import h5py
import _pickle as cPickle
import matplotlib.pyplot as plt

from utilities import (create_folder, get_filename)
import config


def plot(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    select = args.select
    
    classes_num = config.classes_num
    max_plot_iteration = 1000000
    iterations = np.arange(0, max_plot_iteration, 2000)

    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')
        
    save_out_path = 'results_map/{}.pdf'.format(select)
    create_folder(os.path.dirname(save_out_path))
    
    # Read labels
    labels = config.labels
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
        
    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))

        bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
        bal_map = np.mean(bal_map, axis=-1)
        test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
        test_map = np.mean(test_map, axis=-1)
        legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

        # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
        return bal_map, test_map, legend
        
    bal_alpha = 0.3
    test_alpha = 1.0
    lines = []

    if select == '1_cnn13':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_no_dropout', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_no_specaug', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_no_specaug', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_no_dropout', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_no_mixup', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_mixup_in_wave', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_mixup_in_wave', color='c', alpha=test_alpha)
        lines.append(line)

    elif select == '1_pooling':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_gwrp', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_gmpgapgwrp', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_att', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_gmpgapatt', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_resnet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet18', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='ResNet18', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet34', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet50', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet50', color='c', alpha=test_alpha)
        lines.append(line)

    elif select == '1_densenet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'DenseNet121', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='densenet121', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'DenseNet201', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='densenet201', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_cnn9':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn5', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn5', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn9', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn9', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_hop':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            500, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_hop500', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            640, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_hop640', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            1000, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_hop1000', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_emb':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_emb32', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_emb128', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb512', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_emb512', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_mobilenet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='mobilenetv1', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV2', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='mobilenetv2', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_waveform':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_LeeNet', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_LeeNet', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_LeeNet18', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_LeeNet18', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_DaiNet', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_DaiNet', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_ResNet34', color='c', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet50', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_ResNet50', color='m', alpha=test_alpha)
        lines.append(line)

    elif select == '1_waveform_cnn2d':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_SpAndWav', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_WavCnn2d', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_WavCnn2d', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_decision_level':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelMax', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_DecisionLevelMax', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelAvg', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_DecisionLevelAvg', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelAtt', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_DecisionLevelAtt', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_transformer':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_Transformer1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_Transformer1', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_Transformer3', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_Transformer3', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_Transformer6', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_Transformer6', color='k', alpha=test_alpha)
        lines.append(line)

    ax.set_ylim(0, 1.)
    ax.set_xlim(0, len(iterations))
    ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=2)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=lines, bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--dataset_dir', type=str, required=True)
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--select', type=str, required=True)
    
    args = parser.parse_args()

    if args.mode == 'plot':
        plot(args)
        
    else:
        raise Exception('Error argument!')
