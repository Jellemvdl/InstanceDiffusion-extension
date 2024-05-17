#!/bin/bash

# Define the target directory
TARGET_DIR="src/lib/instancediffusion/datasets/"

# Create the target directory if it doesn'\''t exist
mkdir -p "$TARGET_DIR"

# Change to the target directory
cd "$TARGET_DIR"

# URLs for the MSCOCO dataset
BASE_URL="http://images.cocodataset.org/zips/"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Files to download
FILES=("train2017.zip" "val2017.zip" "test2017.zip")

# Download and unzip the dataset files
for FILE in "${FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "Downloading $FILE..."
        wget "${BASE_URL}${FILE}"
        echo "Unzipping $FILE..."
        unzip "$FILE"
        rm "$FILE"
    else
        echo "$FILE already exists, skipping download."
    fi
done

# Download and unzip the annotations
ANNOTATIONS_FILE="annotations_trainval2017.zip"
if [ ! -f "$ANNOTATIONS_FILE" ]; then
    echo "Downloading $ANNOTATIONS_FILE..."
    wget "$ANNOTATIONS_URL"
    echo "Unzipping $ANNOTATIONS_FILE..."
    unzip "$ANNOTATIONS_FILE"
    rm "$ANNOTATIONS_FILE"
else
    echo "$ANNOTATIONS_FILE already exists, skipping download."
fi

# Create a symbolic link from src/data to the new dataset location
cd ../../..
ln -sfn lib/instancediffusion/dataset src/data

echo "Download and extraction complete. Symbolic link created from src/data to src/lib/instancediffusion/datasets."
