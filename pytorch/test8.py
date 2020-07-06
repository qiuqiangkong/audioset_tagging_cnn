import numpy as np
from dataloader import KVReader
num_parallel_reader = 1
# reader = KVReader("/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn/hdfs/waveforms/balanced_train/waveforms", num_parallel_reader)
reader = KVReader("/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn/hdfs/waveforms/eval/waveforms", num_parallel_reader)
keys = reader.list_keys()
values = reader.read_many(keys[0:10])
a1 = np.frombuffer(values[0], dtype=np.int16)


import crash
asdf