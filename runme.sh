#!/bin/bash
DATASET_DIR="/vol/vssp/datasets/audio/audioset/audioset201906"
WORKSPACE="/vol/vssp/cvpnobackup/scratch_4weeks/qk00006/workspaces/pub_audioset_tagging_cnn_transfer"

# ============ Inference with pretrained modela ============
# Inference audio tagging with pretrained model
MODEL_TYPE="Cnn14"
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth?download=1"
python3 pytorch/inference.py audio_tagging --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="examples/R9_ZSCveAHg_7s.wav" --cuda

# Inference sound event detection with pretrained model
MODEL_TYPE="Cnn14_DecisionLevelMax"
CHECKPOINT_PATH="Cnn14_DecisionLevelMax_mAP=0.385.pth"
wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3576403/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
python3 pytorch/inference.py sound_event_detection --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="examples/R9_ZSCveAHg_7s.wav" --cuda

# ============ Download dataset ============
echo "------ Download metadata ------"
mkdir -p $DATASET_DIR"/metadata"

# Video list csv
wget -O $DATASET_DIR"/metadata/eval_segments.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
wget -O $DATASET_DIR"/metadata/balanced_train_segments.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
wget -O $DATASET_DIR"/metadata/unbalanced_train_segments.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

# Class labels indices
wget -O $DATASET_DIR"/metadata/class_labels_indices.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

# Quality of counts
wget -O $DATASET_DIR"/metadata/qa_true_counts.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv

echo "Download metadata to $DATASET_DIR/metadata"

# Split large unbalanced csv file (2,041,789) to 41 partial csv files. 
# Each csv file contains at most 50,000 audio info.
echo "------ Split unbalanced csv to csvs ------"
python3 utils/dataset.py split_unbalanced_csv_to_partial_csvs --unbalanced_csv=$DATASET_DIR/metadata/unbalanced_train_segments.csv --unbalanced_partial_csvs_dir=$DATASET_DIR"/metadata/unbalanced_partial_csvs"

echo "------ Download wavs ------"
# Download evaluation wavs
python3 utils/dataset.py download_wavs --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR"/audios/eval_segments"

# Download balanced train wavs
python3 utils/dataset.py download_wavs --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" --audios_dir=$DATASET_DIR"/audios/balanced_train_segments"

# Download unbalanced train wavs. Users may consider executing the following
# commands in parallel. One simple way is to open 41 terminals and execute
# one command in one terminal.
for IDX in {00..40}; do
  echo $IDX
  python utils/dataset.py download_wavs --csv_path=$DATASET_DIR"/metadata/unbalanced_csvs/unbalanced_train_segments_part$IDX.csv" --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX"
done

# ============ Pack waveform and target to hdf5 ============
# Pack evaluation waveforms to a single hdf5 file
python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR"/audios/eval_segments" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5"

# Pack balanced training waveforms to a single hdf5 file
python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" --audios_dir=$DATASET_DIR"/audios/balanced_train_segments" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5"

# Pack unbalanced training waveforms to hdf5 files. Users may consider 
# executing the following commands in parallel to speed up. One simple 
# way is to open 41 terminals and execute one command in one terminal.
for IDX in {00..40}; do
    echo $IDX
    python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5"
done

# ============ Prepare training indexes ============
# Balanced training indexes
python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/balanced_train.h5"

# Unbalanced training indexes
for IDX in {00..40}; do
    echo $IDX
    python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

# Combine balanced and unbalanced training indexes to a full training indexes hdf5
python3 utils/create_indexes.py combine_full_indexes --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5"

# ============ Blacklist for training (optional) ============
# Audios in the balck list will not be used in training
python3 utils/create_black_list.py dcase2017task4 --workspace=$WORKSPACE

# ============ Train & Inference ============
python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

# Plot statistics
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_aug
