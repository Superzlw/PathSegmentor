<!-- # PathSegmentor -->
[![header](https://capsule-render.vercel.app/api?type=waving&height=180&color=gradient&text=PathSegmentor&reversal=false&desc=Segment%20Anything%20in%20Pathology%20Images%20with%20Natural%20Language&fontSize=40&fontAlign=19&fontAlignY=22&descSize=28&descAlignY=47&descAlign=47)](https://arxiv.org/abs/2407.18449)

[![Arxiv Page](https://img.shields.io/badge/Arxiv-2407.18449-red?style=flat-square)](https://arxiv.org/abs/2407.18449)
<!-- [![GitHub Stars](https://img.shields.io/github/stars/zhi-xuan-chen/PathSegmentor?style=social)](https://github.com/zhi-xuan-chen/PathSegmentor) -->
---
This repository hosts the official implementation of the research paper titled *"Segment Anything in Pathology Images with Natural Language".* It provides code, models, and resources to support natural language-guided semantic segmentation in pathology imaging.

# 🛠 Environment Setup
First, create and activate a Conda environment with Python 3.9.18:

```bash
conda create -n pathsegmentor python=3.9.18
conda activate pathsegmentor
```

Install PyTorch matching your CUDA version from [here](https://pytorch.org/get-started/locally/). For basic setups:
```bash
pip3 install torch torchvision
```

Clone the repository and install dependencies:
```bash
git clone https://github.com/zhi-xuan-chen/PathSegmentor
cd PathSegmentor
pip install -r requirements.txt
```

Install MPI support for distributed training:
```bash
conda install -c conda-forge mpi4py openmpi
```

Finally, install the custom Detectron2 version:
```bash
pip install git+https://github.com/MaureenZOU/detectron2-xyz.git
```


# 🗂️ PathSeg Dataset

The PathSeg dataset was created by preprocessing publicly available pathology image segmentation datasets. For details on the source datasets, please refer to the corresponding documentation: [PathSeg Dataset](assets/readmes/Dataset.md).

To give a general overview of the dataset, we provide a sample subset in the `pathseg_datasets/Demo` directory. You can use this subset to train and evaluate the model, helping you understand how the code functions. Additionally, you may create your own dataset by following the same organizational structure as the `Demo` dataset and fine-tune PathSegmentor on it. Detailed instructions can be found in the [Fine-tuning](#fine-tuning-your-pathsegmentor) section.

# 💾 PathSegmentor Checkpoints
You can download the pre-trained PathSegmentor checkpoints from this Google Drive link: [PathSegmentor Checkpoints](https://drive.google.com/drive/folders/1HxFdA9KXfS6CKgMX-i6TEVfflHiO5Tmf?usp=sharing).
 
# 🚀 How to Use PathSegmentor
The inference workflow using our pre-trained PathSegmentor has been streamlined in the [inference notebook](inference_examples.ipynb). You can directly run inference on sample images provided in the `examples` directory.

>💡 **Note**: Remember to change the checkpoint path `pretrained_pth` in the notebook according to your environment configuration.

We provide four types of examples in the `examples` directory: `1_irregular`, `2_tiny`, `3_dense`, and `4_others`. These directories may contain further subcategories, but each final-level folder always includes two files: an image and a mask.

To run inference on a specific image, simply replace the `image_path` and `mask_path` variables in the notebook with the paths to your chosen image and mask files.

The `text_prompt` used for segmentation can be derived from the end of the mask filename. For example, the mask file named `Part_2_1211_pathology_bile+duct_nuclei+connective.png` corresponds to the prompt:
“nuclei-level connective in bile duct pathology”.

>💡 **Note:** Remove any `+` signs when constructing the prompt.

Now, run each cell in the notebook. A figure containing the image, predicted mask, and ground truth (GT) mask will be generated at the end, allowing you to compare the results.

# 🔧 Fine-tuning Your PathSegmentor
To fine-tune PathSegmentor on your own dataset, follow the instructions below to organize your data according to the required format and begin the fine-tuning process.

## 📁 Dataset Organization
Before you begin, please create a folder named `pathseg_datasets` in a directory with sufficient storage space. This folder will be used to store the preprocessed datasets required by PathSegmentor.

The structure of each dataset within `pathseg_datasets` is organized as follows:
<pre>
pathseg_datasets  
├── [DATASET_NAME]  
│   ├── train  
│   ├── train_mask  
│   ├── test  
│   └── test_mask
</pre>
The `train` and `train_mask` folders contain the images and corresponding mask files used for training, while the `test` and `test_mask` folders hold the images and mask files used for testing.

The `pathseg_datasets/Demo` directory serves as a clear example of how to properly structure a dataset named `Demo`.

## ⚙️ Preprocessing Dataset
### Image Processing
PathSegmentor requires `1024×1024` image patches extracted from whole slide images (WSIs) at `40x` magnification. Therefore, ensure your WSIs are scaled to `40x`, then crop them into `1024×1024` patches. Save the resulting images into either the `train` or `test` directories based on their designated split.

Please ensure that image filenames adhere to the following format:
```
[IMAGE-NAME]_pathology_[REGION].png
```
Here, `[IMAGE-NAME]` refers to a unique identifier for each image, while `[REGION]` indicates the anatomical region to which the pathology image belongs.

### Mask Processing
Each image can have multiple binary mask files of the same size (1024×1024), with each mask representing a distinct object within the image. In these masks, pixels with a value of 1 indicate the foreground region.

The mask filenames follow a similar format but include an additional field:
```
[IMAGE-NAME]_pathology_[REGION]_[OBJECT].png
```
The initial part of the mask filename matches its corresponding image. The `[OBJECT]` segment indicates the specific object depicted by the mask within the image. In `[OBJECT]`, spaces are replaced with `+` signs for easier parsing. The first component of `[OBJECT]` indicates the histological structure the object belongs to. For example, `nuclei+neoplastic` refers to a `neoplastic` object belonging to the `nuclei` structure.

## 📝 Generating Dataset Info JSON
After preparing the preprocessed image and mask files, you can generate the dataset info JSON files, which are used for both training and testing.

You can generate the JSON files for a single dataset by running the following command:
```bash
cd pathseg_datasets
python create_single_dataset.py
```
>💡 **Note:** Be sure to set the `targetpath` parameter to the path of your dataset.

If you have multiple datasets that require JSON file generation, run the following command:
```bash
cd pathseg_datasets
python create_multiple_dataset.py
```
>💡 **Note:** Similarly, make sure to update the `dataset_paths` list with the paths to all your datasets.

After the JSON files are generated, your dataset directory will look like this:
<pre>
pathseg_datasets  
├── [DATASET_NAME]  
│   ├── train  
│   ├── train_mask  
│   ├── train.json  
│   ├── test  
│   ├── test_mask  
│   └── test.json  
</pre>

## 📋 Register Dataset
To use these datasets, you need to register them appropriately. Open the file [register_pathseg_datasets.py](datasets/registration/register_pathseg_datasets.py), and add your `[DATASET_NAME]` to the `datasets` list. The training dataset will be registered as `pathseg_[DATASET_NAME]_train`, and the testing dataset as `pathseg_[DATASET_NAME]_test`.

Finally, add these registered dataset to the configuration file [pathsegmentor.yaml](configs/pathsegmentor.yaml), around line 233.

Once completed, your datasets will be ready for training and evaluation.

## 💪 Training
You can start training by running the following command:
```bash
bash scripts/train.sh
```
>💡 **Note:** You may modify the `SAVE_DIR` and `RESUME_FROM` variables in [pathsegmentor.yaml](configs/pathsegmentor.yaml) as needed. These variables specify the output path for training checkpoints and the path to the pre-trained checkpoint, respectively.

## 🧪 Evaluation
After completing training, you can evaluate your fine-tuned PathSegmentor on your dataset's test set by running:
```bash
bash scripts/eval.sh
```
>💡 **Note:** You can customize the output directory for result JSON files in [default_trainer.py](trainer/default_trainer.py), around line 79.

# Citation
