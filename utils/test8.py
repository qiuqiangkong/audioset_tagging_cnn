import time
import random
from dataloader import KVReader

num_parallel_reader = 8
# reader = KVReader("hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/test9/dataset", num_parallel_reader)
reader = KVReader("hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/workspaces/audioset_tagging/hdfs/waveforms/balanced_train/waveforms", num_parallel_reader)
keys = reader.list_keys()
# values = reader.read_many(keys)

random.shuffle(keys)
print(keys)
# print(values)


pointer = 0
batch_size = 64

while pointer < len(keys):
	t1 = time.time()
	values = reader.read_many(keys[pointer : pointer + batch_size])
	pointer += batch_size
	print(time.time() - t1, len(values))

