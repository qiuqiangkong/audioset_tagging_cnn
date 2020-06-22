cd /mnt/cephfs_new_wj/speechsv/kongqiuqiang/my_code_2019.12-/python/cvssp/pub_audioset_tagging_cnn
DATASET_DIR="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/audioset"
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn"

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda
