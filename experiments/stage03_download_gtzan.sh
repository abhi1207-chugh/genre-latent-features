#!/bin/bash
# Stage 3: download GTZAN from Kaggle into data/raw/.
#
# Needs a Kaggle API token in ~/.kaggle/access_token
# (kaggle.com -> Settings -> API -> Create New Token).
#
# Run from the project root:
#     bash experiments/stage03_download_gtzan.sh
set -e

DATASET="andradaolteanu/gtzan-dataset-music-genre-classification"
RAW_DIR="data/raw"

# Kaggle's zip has an extra "Data/" folder on top.
if [ -d "$RAW_DIR/genres_original" ] || [ -d "$RAW_DIR/Data/genres_original" ]; then
    echo "GTZAN already downloaded, skipping."
else
    .venv/bin/kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip
fi

# Move the audio folder up to data/raw/genres_original.
if [ -d "$RAW_DIR/Data/genres_original" ]; then
    mv "$RAW_DIR/Data/genres_original" "$RAW_DIR/genres_original"
fi

# Remove the spectrogram images and the engineered-feature CSVs.
# The project must learn from raw audio only.
rm -rf "$RAW_DIR/Data" "$RAW_DIR/images_original" "$RAW_DIR"/features_*.csv

echo "Done. Contents of $RAW_DIR/genres_original:"
ls "$RAW_DIR/genres_original"
