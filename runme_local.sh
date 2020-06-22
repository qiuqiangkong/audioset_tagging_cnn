
MODEL_TYPE="Cnn14"
CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/sed/Cnn14_mAP=0.431.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="examples/R9_ZSCveAHg_7s.wav" --cuda

MODEL_TYPE="Cnn14_DecisionLevelMax"
CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py sound_event_detection --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="examples/R9_ZSCveAHg_7s.wav" --cuda

MODEL_TYPE="Transfer_Cnn14"
CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/sed/Cnn14_mAP=0.431.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/finetune_template.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$CHECKPOINT_PATH --cuda

###########
cd /mnt/cephfs_new_wj/speechsv/kongqiuqiang/my_code_2019.12-/python/cvssp/pub_audioset_tagging_cnn
DATASET_DIR="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/audioset"
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn"

python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR"/audios/eval_segments" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5"

python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/eval.h5"

python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/balanced_train.h5"

for IDX in {00..40}; do
    echo $IDX
    python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

python3 utils/create_indexes.py combine_full_indexes --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5"

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='balanced_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14_mixup_time_domain' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_aug
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_bal_train_aug
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_sr
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_time_domain
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_partial_full
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_window
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_melbins
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_alternate

######
python3 utils/create_indexes_partial.py --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5" --partial_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/partial_0.9_full_train.h5" --partial=0.9

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='partial_0.5_full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='alternate' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=3000000 --cuda

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=128 --fmin=50 --fmax=14000 --model_type='Cnn14_mel128' --loss_type='clip_bce' --balanced='alternate' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=3000000 --cuda

python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=128 --fmin=50 --fmax=14000 --model_type='Wavegram_Logmel128_Cnn14' --loss_type='clip_bce' --balanced='alternate' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=3000000 --cuda


####
MODEL_TYPE="Cnn14"
CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/sed/Cnn14_mAP=0.431.pth"
python3 pytorch/test9.py train --workspace=$WORKSPACE --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --cuda
