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
        
    if select == '1':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug_debug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13_fc_debug', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_fc_debug', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug_debug', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, label='cnn13_fc_mixup_debug', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_fc_mixup_debug', color='b', alpha=test_alpha)
        lines.append(line)

    if select == '1_cnn13':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug_debug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13_fc_debug', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_fc_debug', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13_fc_', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_fc', color='b', alpha=test_alpha)
        lines.append(line)
 
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug_debug3', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13_fc_debug3', color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_fc_debug3', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug_debug4', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='xavier', color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='xavier', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_mixup':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, label='cnn13_mixup', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_mixup', color='b', alpha=test_alpha)
        lines.append(line)

    elif select == '1_dropout':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_dropout_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13_dropout', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_dropout', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_dropout_specaug', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, label='cnn13_dropout_mixup', color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_dropout_mixup', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_cnn9':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn9_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn5_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_resnet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'DenseNet121_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='densenet121', color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='densenet121', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'DenseNet201_specaug', 'clip_bce', 'balanced', 'none', 16)
        line, = ax.plot(bal_map, label='densenet201', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='densenet201', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet18_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='resnet18', color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet18', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet34_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='resnet34', color='y', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet34', color='y', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet50_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='resnet50', color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet50', color='c', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet152_specaug', 'clip_bce', 'balanced', 'none', 16)
        line, = ax.plot(bal_map, label='resnet152', color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet152', color='m', alpha=test_alpha)
        lines.append(line)

    elif select == '1_mobilenet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='cnn13', color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='mobilenetv1', color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='mobilenetv1', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV2_specaug', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, label='mobilenetv2', color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='mobilenetv2', color='g', alpha=test_alpha)
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
