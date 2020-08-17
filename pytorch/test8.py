import pickle


def load_statistics(statistics_path):
    statistics_dict = pickle.load(open(statistics_path, 'rb'))

    bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
    bal_map = np.mean(bal_map, axis=-1)
    test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
    test_map = np.mean(test_map, axis=-1)

    return bal_map, test_map


stats = pickle.load(open('paper_statistics/stats_for_long_fig.pkl', 'rb'))
stats2 = {'labels': stats['labels'], 'label_quality': stats['label_quality'], 'sorted_indexes_for_plot': stats['sorted_indexes_for_plot'], 'official_balanced_training_samples': stats['official_balanced_trainig_samples'], 'official_unbalanced_training_samples': stats['official_unbalanced_training_samples'], 'official_eval_samples': stats['official_eval_samples'], 'downloaded_full_training_samples': stats['downloaded_full_training_samples'], 'averaging_instance_system_avg_9_probs_from_10000_to_50000_iterations': stats['averaging_instance_system_avg_9_probs_from_10000_to_50000_iterations'], 'panns_cnn14': stats['cnn13_system_iteration60k'], 'panns_mobilenetv1': stats['mobilenetv1_system_iteration56k']}

tmp = pickle.load(open('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_WavegramLogmelCnn_balanced_mixup_bs32.pkl', 'rb'))

N = 300
tmp2 = {'bal_train': {'average_precision': tmp['bal'][N]['average_precision'], 'auc': tmp['bal'][N]['auc']}, 'eval': {'average_precision': tmp['test'][N]['average_precision'], 'auc': tmp['test'][N]['auc']}}

stats2['panns_wavegram_logmel_cnn14'] = tmp2
pickle.dump(stats2, open('paper_statistics/stats_for_long_fig2.pkl', 'wb'))
import crash
asdf