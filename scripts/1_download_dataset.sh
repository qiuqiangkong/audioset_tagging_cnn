#!/bin/bash
DATASET_DIR=${1:-"./datasets/audioset201906"}   # Default argument.

echo "------ Download metadata ------"
mkdir -p $DATASET_DIR"/metadata"

# Download ideo list csv.
wget -O $DATASET_DIR"/metadata/eval_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
wget -O $DATASET_DIR"/metadata/balanced_train_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
wget -O $DATASET_DIR"/metadata/unbalanced_train_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"

# Download class labels indices.
wget -O $DATASET_DIR"/metadata/class_labels_indices.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

# Download quality of counts.
wget -O $DATASET_DIR"/metadata/qa_true_counts.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv"

echo "Download metadata to $DATASET_DIR/metadata"

# Split large unbalanced csv file (2,041,789) to 41 partial csv files. 
# Each csv file contains at most 50,000 audio info.
echo "------ Split unbalanced csv to csvs ------"
python3 utils/dataset.py split_unbalanced_csv_to_partial_csvs \
    --unbalanced_csv=$DATASET_DIR/metadata/unbalanced_train_segments.csv \
    --unbalanced_partial_csvs_dir=$DATASET_DIR"/metadata/unbalanced_partial_csvs"

echo "------ Download wavs ------"
# Download evaluation wavs
python3 utils/dataset.py download_wavs \
    --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" \
    --audios_dir=$DATASET_DIR"/audios/eval_segments"

# Download balanced train wavs
python3 utils/dataset.py download_wavs \
    --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" \
    --audios_dir=$DATASET_DIR"/audios/balanced_train_segments"

# Download unbalanced train wavs. Users may consider executing the following
# commands in parallel. One simple way is to open 41 terminals and execute
# one command in one terminal.
for IDX in {00..40}; do
  echo $IDX
  python utils/dataset.py download_wavs \
    --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" \
    --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX"
done