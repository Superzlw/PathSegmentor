# Data Preprocessing Pipeline for PathSeg
This pipeline aims to convert raw images and masks into a unified **PathSeg** standard format. The format supports multimodal prompt-based segmentation tasks.

## 1. Overview of the Data Processing Pipeline

This pipeline automates the conversion from raw high-resolution whole slide images (WSI) to standardized $1024 \times 1024$ training samples through five core steps: [!IMPORTANT] Path configuration note: Before running the script, please make sure to manually check and modify [RAW_DATA_DIR] (raw data path) and [PROCESS_DATA_DIR] (preprocessing output path) in the code according to your server environment.

## 2. Step-by-Step Details

### Step 1: Image Tiling (Tiling & Patching)

Since raw pathology images are extremely high in resolution, they are first split into fixed-size patches.

* **Input**: Raw color images (`rgbs_colorNormalized`) and raw masks (`masks`).
* **Process**: Sliding window algorithm with overlap.
* **Output**: Save $1024 \times 1024$ patches and generate `patchinfo.txt` to record each patch’s coordinates in the original image.

### Step 2: Dataset Splitting and JSON Generation (Data Splitting)

Randomly split the processed patches into training and test sets.

* **Logic**: By default, shuffle and split with **80% training / 20% testing**.
* **Output**: `train.json` and `test.json`, storing index lists containing all valid image IDs.

### Step 3: Label Cleaning and Remapping (Relabeling) (Optional)

Clean dataset categories by removing invalid annotations and making IDs contiguous.

* **Output**: Cleaned grayscale images stored under the `mask_relabel` directory.

### Step 4: Binary Mask Decomposition (Binary Mask Decomposition)

* **Naming rule**: `[CleanID]_[Modality]_[Site]_[TargetName].png`
* **Target mapping**:
* For example, extract ID 1 as `tissue-tumor`.
* All masks are resized to $1024 \times 1024$ using **Nearest Neighbor** interpolation to preserve label accuracy.

### Step 5: Image Standardization (Image Standardization)

Final standardized naming and resizing of original images.

* **Process**: Uniformly resize to $1024 \times 1024$ using **Bilinear** interpolation to maintain image quality.
* **Naming**: `[CleanID]_[Modality]_[Site].png`

## 3. Standardized Directory Structure

After processing, your dataset directory should look like:

```text
/biomedparse_datasets/BCSS/
├── train/              # Standardized training images (1024x1024)
│   └── ID_pathology_breast.png
├── train_mask/         # Corresponding binary masks (0/1)
│   ├── ID_pathology_breast_tissue+tumor.png
│   └── ID_pathology_breast_tissue+stroma.png
├── test/               # Test images
└── test_mask/          # Test binary masks

```

