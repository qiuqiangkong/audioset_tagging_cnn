import argparse
import csv
import os

from utilities import create_folder


def dcase2017task4(args):
    """Create black list. Black list is a list of audio ids that will be 
    skipped in training. 
    """

    # Augments & parameters
    workspace = args.workspace
    
    # Paths
    dcase2017task4_dataset_dir = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/dcase2017_task4'

    test_weak_csv = os.path.join(dcase2017task4_dataset_dir, 'metadata/groundtruth_weak_label_testing_set.csv')
    evaluation_weak_csv = os.path.join(dcase2017task4_dataset_dir, 'metadata/groundtruth_weak_label_evaluation_set.csv')
    
    black_list_csv = os.path.join(workspace, 'black_list', 'dcase2017task4.csv')
    create_folder(os.path.dirname(black_list_csv))
    
    def get_id_sets(csv_path):
        with open(csv_path, 'r') as fr:
            reader = csv.reader(fr, delimiter='\t')
            lines = list(reader)
         
        ids_set = [] 
        
        for line in lines:
            ids_set.append(line[0][0 : 11])
            
        ids_set = list(set(ids_set))
        return ids_set
        
    test_ids_set = get_id_sets(test_weak_csv)
    evaluation_ids_set = get_id_sets(evaluation_weak_csv)
    
    full_ids_set = test_ids_set + evaluation_ids_set
    
    # Write black list
    fw = open(black_list_csv, 'w')
    
    for id in full_ids_set:
        fw.write('{}\n'.format(id))
        
    print('Write black list to {}'.format(black_list_csv))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_dcase2017task4 = subparsers.add_parser('dcase2017task4')
    parser_dcase2017task4.add_argument('--workspace', type=str, required=True)
        
    args = parser.parse_args()

    if args.mode == 'dcase2017task4':
        dcase2017task4(args)
        
    else:
        raise Exception('Error argument!')