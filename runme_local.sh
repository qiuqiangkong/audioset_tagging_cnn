
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



####### HDFS

HDFS_WORKSPACE='hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/workspaces/audioset_tagging'
WORKSPACE="/home/tiger/workspaces/audioset_tagging"

python3 utils/hdfs_dataset.py pack_waveforms_to_hdfs --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR"/audios/eval_segments" --hdfs_path=$WORKSPACE"/hdfs/waveforms/eval"

python3 utils/hdfs_dataset.py pack_waveforms_to_hdfs --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" --audios_dir=$DATASET_DIR"/audios/balanced_train_segments" --hdfs_path=$WORKSPACE"/hdfs/waveforms/balanced_train"

for IDX in {30..41}; do
    echo $IDX
    python3 utils/hdfs_dataset.py pack_waveforms_to_hdfs --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" --hdfs_path=$WORKSPACE"/hdfs/waveforms/unbalanced_train_segments/unbalanced_train_segments_part$IDX"
done

# Create indexes
python3 utils/hdfs_create_indexes.py create_indexes --hdfs_dir=$WORKSPACE"/hdfs/waveforms/balanced_train" --indexes_hdf5_path=$WORKSPACE"/hdfs/indexes/balanced_train.h5"

python3 utils/hdfs_create_indexes.py create_indexes --hdfs_dir=$WORKSPACE"/hdfs/waveforms/eval" --indexes_hdf5_path=$WORKSPACE"/hdfs/indexes/eval.h5"

for IDX in {00..40}; do
    echo $IDX
    python3 utils/hdfs_create_indexes.py create_indexes --hdfs_dir=$WORKSPACE"/hdfs/waveforms/unbalanced_train_segments/unbalanced_train_segments_part$IDX" --indexes_hdf5_path=$WORKSPACE"/hdfs/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

python3 utils/hdfs_create_indexes.py combine_full_indexes --indexes_hdf5s_dir=$WORKSPACE"/hdfs/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdfs/indexes/full_train.h5"

# Train
CUDA_VISIBLE_DEVICES=0 python3 pytorch/hdfs_main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

####
# WORKSPACE="/home/tiger/workspaces/audioset_tagging"
WORKSPACE="/root/workspaces/audioset_tagging"

CUDA_VISIBLE_DEVICES=2 python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

MODEL_TYPE="Cnn14"
CHECKPOINT_PATH='/home/tiger/released_models/sed/Cnn14_mAP=0.431.pth'
CUDA_VISIBLE_DEVICES=0 python3 pytorch/calculate_opt_thresholds.py opt_thres --workspace=$WORKSPACE --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --cuda

cd /root/my_code_2019.12-/python/audioset_tagging_pytorch
python3 utils/wavs_to_mp3s.py


##
python3 utils/dataset_mp3.py pack_waveforms_to_hdf5 --csv_path="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/audioset/metadata/eval_segments.csv" --audios_dir="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/audioset/audios_mp3/eval_segments" --waveforms_hdf5_path="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn/hdf5s_mp3/waveforms/eval.h5"

# Pack balanced training waveforms to a single hdf5 file
python3 utils/dataset_mp3.py pack_waveforms_to_hdf5 --csv_path="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/audioset/metadata/balanced_train_segments.csv" --audios_dir="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/audioset/audios_mp3/balanced_train_segments" --waveforms_hdf5_path="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn/hdf5s_mp3/waveforms/balanced_train.h5"


IDX="39"
python3 utils/dataset_mp3.py pack_waveforms_to_hdf5 --csv_path="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/datasets/audioset/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" --audios_dir="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/audioset/audios_mp3/unbalanced_train_segments/unbalanced_train_segments_part$IDX" --waveforms_hdf5_path="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn/hdf5s_mp3/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5"

python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s_mp3/waveforms/eval.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s_mp3/indexes/eval.h5"

python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s_mp3/waveforms/balanced_train.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s_mp3/indexes/balanced_train.h5"

for IDX in {00..40}; do
    echo $IDX
    python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s_mp3/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s_mp3/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

python3 utils/create_indexes.py combine_full_indexes --indexes_hdf5s_dir=$WORKSPACE"/hdf5s_mp3/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s_mp3/indexes/full_train.h5"

WORKSPACE="/home/tiger/workspaces/audioset_tagging"

CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --sample_rate=32000 --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14_Transformer_pos' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda


CUDA_VISIBLE_DEVICES=3 python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --sample_rate=16000 --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000 --model_type='Cnn14_small_16k' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

CUDA_VISIBLE_DEVICES=1 python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --sample_rate=8000 --window_size=256 --hop_size=80 --mel_bins=64 --fmin=50 --fmax=4000 --model_type='Cnn14_8k' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

CUDA_VISIBLE_DEVICES=3 python3 pytorch/inference.py audio_tagging --sample_rate=16000 --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000 --model_type='Cnn14_16k' --checkpoint_path='/home/tiger/workspaces/audioset_tagging/checkpoints/main/sample_rate=16000,window_size=512,hop_size=160,mel_bins=64,fmin=50,fmax=8000/data_type=full_train/Cnn14_16k/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/900000_iterations.pth' --audio_path='resources/R9_ZSCveAHg_7s.wav' --cuda

CUDA_VISIBLE_DEVICES=3 python3 pytorch/inference.py audio_tagging --sample_rate=8000 --window_size=256 --hop_size=80 --mel_bins=64 --fmin=50 --fmax=4000 --model_type='Cnn14_8k' --checkpoint_path='/home/tiger/workspaces/audioset_tagging/checkpoints/main/sample_rate=8000,window_size=256,hop_size=80,mel_bins=64,fmin=50,fmax=4000/data_type=full_train/Cnn14_8k/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/900000_iterations.pth' --audio_path='resources/R9_ZSCveAHg_7s.wav' --cuda

python3 utils/plot_statistics2.py plot --workspace=$WORKSPACE --select=3a

CHECKPOINT_PATH="/home/tiger/workspaces/audioset_tagging/checkpoints/main/sample_rate=16000,window_size=512,hop_size=160,mel_bins=64,fmin=50,fmax=8000/data_type=full_train/Cnn14_16k/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/200000_iterations.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging --model_type=Cnn14_16k --checkpoint_path=$CHECKPOINT_PATH --audio_path="resources/R9_ZSCveAHg_7s.wav" --cuda --sample_rate=16000 --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000

wget -O paper_statistics.zip https://zenodo.org/record/3987831/files/paper_statistics.zip?download=1
unzip paper_statistics.zip
python3 utils/plot_for_paper.py plot_classwise_iteration_map
python3 utils/plot_for_paper.py plot_six_figures
python3 utils/plot_for_paper.py plot_complexity_map
python3 utils/plot_for_paper.py plot_long_fig
