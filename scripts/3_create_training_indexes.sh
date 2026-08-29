#!/bin/bash
WORKSPACE=${1:-"./workspaces/audioset_tagging"}   # Default argument.

# evaluation indexes
python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5" \
    --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/eval.h5"

# Balanced training indexes
python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" \
    --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/balanced_train.h5"

# Unbalanced training indexes
for IDX in {00..40}; do
    echo $IDX
    python3 utils/create_indexes.py create_indexes \
        --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" \
        --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

# Combine balanced and unbalanced training indexes to a full training indexes hdf5
python3 utils/create_indexes.py combine_full_indexes \
    --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" \
    --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5"

# ============ Blacklist for training (optional) ============
# Audios in the balck list will not be used in training
python3 utils/create_black_list.py dcase2017task4 --workspace=$WORKSPACE