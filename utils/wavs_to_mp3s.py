import os

wavs_dir = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/audioset/audios/unbalanced_train_segments/unbalanced_train_segments_part40'
mp3s_dir = '/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/audioset/audios_mp3/unbalanced_train_segments/unbalanced_train_segments_part40'

os.makedirs(mp3s_dir, exist_ok=True)

wav_names = os.listdir(wavs_dir)

for n, wav_name in enumerate(wav_names):
    print(n, wav_name)
    wav_path = os.path.join(wavs_dir, wav_name)
    mp3_path = os.path.join(mp3s_dir, wav_name[0 : -4] + '.mp3')
    os.system('ffmpeg -loglevel panic -y -i "{}" "{}"'.format(wav_path, mp3_path))
