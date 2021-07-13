#!/bin/bash
DATASET_DIR=${1:-"./datasets/audioset201906"}   # Default first argument.
WORKSPACE=${2:-"./workspaces/audioset_tagging"}   # Default second argument.

# Pack evaluation waveforms to a single hdf5 file
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" \
    --audios_dir=$DATASET_DIR"/audios/eval_segments" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5"

# Pack balanced training waveforms to a single hdf5 file
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" \
    --audios_dir=$DATASET_DIR"/audios/balanced_train_segments" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5"

# Pack unbalanced training waveforms to hdf5 files. Users may consider 
# executing the following commands in parallel to speed up. One simple 
# way is to open 41 terminals and execute one command in one terminal.
for IDX in {00..40}; do
    echo $IDX
    python3 utils/dataset.py pack_waveforms_to_hdf5 \
        --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" \
        --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" \
        --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5"
done