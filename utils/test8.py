from dataloader import KVReader

num_parallel_reader = 1
reader = KVReader("hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/test9/dataset", num_parallel_reader)
# reader = KVReader("hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/workspaces/audioset_tagging/hdfs/waveforms/balanced_train/waveforms", num_parallel_reader)
keys = reader.list_keys()
values = reader.read_many(keys)

print(keys)
# print(values)

import crash
asdf