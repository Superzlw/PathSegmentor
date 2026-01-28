#!/bin/bash

# Define the root directory for raw data
RAW_DATA_DIR="./raw_data_directory"
mkdir -p "$RAW_DATA_DIR"

# Counters
AUTO_COUNT=0
MANUAL_COUNT=9 # CoNIC, DigestPath, MIDOG, MoNuSAC, WSSS4LUAD, RINGS, CoNSeP, CRAG, SegPath

# Function to check and install gdown if missing
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing via pip..."
    pip install gdown
fi

echo "-------------------------------------------------------"
echo "Starting automated download for PathSeg datasets..."
echo "-------------------------------------------------------"

# --- 1. Google Drive Datasets (Using gdown) ---
echo "[1/3] Downloading Google Drive datasets..."

# BCSS (Option 1)
gdown --folder 1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss -O "$RAW_DATA_DIR/BCSS/" && ((AUTO_COUNT++))
# CPM15
gdown --folder 11ko-GcDsPpA9GBHuCtl_jNzWQl6qY_-I -O "$RAW_DATA_DIR/CPM15/" && ((AUTO_COUNT++))
# CPM17
gdown --folder 1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA -O "$RAW_DATA_DIR/CPM17/" && ((AUTO_COUNT++))
# Kumar
gdown --folder 1bI3RyshWej9c4YoRW-_q7lh7FOFDFUrJ -O "$RAW_DATA_DIR/Kumar/" && ((AUTO_COUNT++))
# NuCLS
gdown --folder 1gvjlC9lPYUyf03X2ddWKscqUcEMEsIFX -O "$RAW_DATA_DIR/NuCLS/" && ((AUTO_COUNT++))

# --- 2. Direct Web Downloads (wget) ---
echo "[2/3] Downloading datasets via direct links..."

# CoCaHis
mkdir -p "$RAW_DATA_DIR/CoCaHis"
wget -q --show-progress -O "$RAW_DATA_DIR/CoCaHis/CoCaHis.hdf5" http://cocahis.irb.hr/wp-content/uploads/2020/12/CoCaHis.hdf5 && ((AUTO_COUNT++))

# Janowczyk
mkdir -p "$RAW_DATA_DIR/Janowczyk"
wget -q --show-progress -O "$RAW_DATA_DIR/Janowczyk/nuclei.tgz" https://andrewjanowczyk.com/wp-static/nuclei.tgz && {
    tar -xzf "$RAW_DATA_DIR/Janowczyk/nuclei.tgz" -C "$RAW_DATA_DIR/Janowczyk/"
    ((AUTO_COUNT++))
}

# TNBC
mkdir -p "$RAW_DATA_DIR/TNBC"
wget -q --show-progress -O "$RAW_DATA_DIR/TNBC/TNBC_NucleiSegmentation.zip" "https://zenodo.org/records/2579118/files/TNBC_NucleiSegmentation.zip?download=1" && {
    unzip -q "$RAW_DATA_DIR/TNBC/TNBC_NucleiSegmentation.zip" -d "$RAW_DATA_DIR/TNBC/"
    ((AUTO_COUNT++))
}

# GlaS
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi
python3 -c "import kagglehub" &> /dev/null
if [ $? -ne 0 ]; then
    echo "kagglehub not found. Installing..."
    pip install kagglehub
fi
mkdir -p "$RAW_DATA_DIR/GlaS"
echo "Downloading GlaS via kagglehub..."
python3 -c "
import kagglehub
import shutil
import os
try:
    download_path = kagglehub.dataset_download('sani84/glasmiccai2015-gland-segmentation')
    target_dir = '$RAW_DATA_DIR/GlaS'
    for item in os.listdir(download_path):
        s = os.path.join(download_path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
except Exception as e:
    print(f'Error downloading GlaS: {e}')
" && ((AUTO_COUNT++))

# --- 3. Kaggle Datasets ---
if command -v kaggle &> /dev/null; then
    echo "[3/3] Downloading Kaggle datasets..."
    kaggle datasets download -d andrewmvd/cancer-inst-segmentation-and-classification -p "$RAW_DATA_DIR/PanNuke" --unzip && ((AUTO_COUNT++))
    kaggle datasets download -d shridharspol/sicapv2 -p "$RAW_DATA_DIR/SICAPv2" --unzip && ((AUTO_COUNT++))
    kaggle datasets download -d aadimator/lizard-dataset -p "$RAW_DATA_DIR/Lizard" --unzip && ((AUTO_COUNT++))
else
    echo "[SKIP] Kaggle CLI not found. Skipping PanNuke, SICAPv2, and Lizard."
fi

echo "-------------------------------------------------------"
echo "Automated Download Summary:"
echo "Successfully downloaded: $AUTO_COUNT datasets."
echo "Requires manual action:  $MANUAL_COUNT datasets."
echo "-------------------------------------------------------"
echo "PLEASE MANUALLY DOWNLOAD THE FOLLOWING DUE TO ACCESS RESTRICTIONS:"
echo "1. CoNIC2022      (Grand-Challenge Login)"
echo "2. DigestPath2019 (DUA/Login Required)"
echo "3. MIDOG          (Google Drive Manual Access)"
echo "4. MoNuSAC        (Grand-Challenge/Google Drive Login)"
echo "5. WSSS4LUAD      (Grand-Challenge Login)"
echo "6. RINGS          (Mendeley Data Download)"
echo "7. CoNSeP         (OpenDataLab Login)"
echo "8. CRAG           (OpenDataLab Login)"
echo "9. SegPath        (Manual Zenodo multi-part download recommended)"
echo "-------------------------------------------------------"