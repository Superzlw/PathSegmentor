# PathSegmentor
This repository hosts the official implementation of the research paper titled *"Segment Anything in Pathology Images with Natural Language".* It provides code, models, and resources to support natural language-guided semantic segmentation in pathology imaging.

# Installation
To set up the environment, run the following command:
```bash
git clone https://github.com/zhi-xuan-chen/PathSegmentor
conda create -n pathsegmentor python=3.9.18
conda activate pathsegmentor
pip install -r requirements.txt
```

# PathSeg Dataset

PathSeg dataset was constructed from preprocessing publicly available pathology image segmentation datasets. For the source datasets, please check the details here: [PathSeg Dataset](assets/readmes/Dataset.md). 

To provide a general impression of the dataset, we showcase a subset of PathSeg dataset in [PathSeg-Demo](pathseg_datasets/Demo). You can train and evaluate the model on this dataset to understand how the code works. You may also create your own dataset by following the organizational structure of the demo dataset. 

Detailed instructions can be found in the [Fine-tuning](#fine-tuning) section.

# PathSegmentor Checkpoints
You can download the pre-trained PathSegmentor checkpoints from the following Google Drive link: [PathSegmentor Checkpoints](https://drive.google.com/drive/folders/1HxFdA9KXfS6CKgMX-i6TEVfflHiO5Tmf?usp=sharing).
 
# How to Use PathSegmentor 
We have pipelined the inference process using our pre-trained PathSegmentor in the [inference notebook](inference_examples.ipynb). You can directly perform inference on the example images located in the `examples` directory. 

Remember to change the checkpoint path `pretrained_pth` in the notebook according your environment configuration.  

We provide four types of examples in the examples directory: `1_irregular`, `2_tiny`, `3_dense`, and `4_others`. These directories may contain further subcategories, but each final-level folder always includes two files: an image and a mask.

To run inference on a specific image, simply replace the `image_path` and `mask_path` variables in the notebook with the paths to your chosen image and mask files.

The `text_prompt` used for segmentation can be derived from the end of the mask filename. For example, the mask file named `Part_2_1211_pathology_bile+duct_nuclei+connective.png` corresponds to the prompt:
“nuclei-level connective in bile duct pathology”.

> **Note:** Be sure to remove any `+` signs when constructing the prompt.

Now, run each cell in the notebook. A figure containing the image, predicted mask, and ground truth (GT) mask will be generated at the end, allowing you to compare the results.

# Fine-tuning Your PathSegmentor
If you want to fine-tune PathSegmentor on your own dataset, refer to these instructions to organize your data according to PathSegmentor’s requirements and begin fine-tuning.

## Dataset Organization
Before you begin, please create a folder named `pathseg_datasets` in a directory with sufficient storage space. This folder will be used to store the preprocessed datasets required by PathSegmentor.

The structure of each dataset within this directory is organized as follows:
<pre>
pathseg_datasets  
├── [DATASET_NAME]  
│   ├── train  
│   ├── train_mask  
│   ├── test  
│   └── test_mask
</pre>
The `train` and `train_mask` folders contain the images and corresponding mask files used for training, while the `test` and `test_mask` folders hold the images and mask files used for testing.

The [PathSeg-Demo](pathseg_datasets/Demo) provides a clear example demonstrating how a dataset with [DATASET_NAME] set to `Demo` should be organized.

## Preprocessing Dataset
### Image Processing
PathSegmentor takes as input `1024×1024` images cropped from whole slide images (WSIs) at `40x` magnification. Therefore, please downsample or upsample your WSIs to `40x` and crop them into `1024×1024` patches. Depending on their designated split, save the images into the `train` or `test` folders accordingly.

Note that image names must be formatted according to the following requirements:
```
[IMAGE-NAME]_pathology_[REGION].png
```
Here, `[IMAGE-NAME]` refers to a unique identifier for each image, while `[REGION]` indicates the anatomical region to which the pathology image belongs.

### Mask Processing
Each image can have multiple binary mask files of the same size (1024×1024), with each mask representing a distinct object within the image. In these masks, pixels with a value of 1 indicate the foreground region.

The mask filenames follow a similar format but include additional fields:
```
[IMAGE-NAME]_pathology_[REGION]_[OBJECT].png
```
The initial part of the mask filename matches its corresponding image. The `[OBJECT]` segment specifies which object within the image the mask represents. In `[OBJECT]`, spaces are replaced with `+` signs for easier parsing. The first component of `[OBJECT]` indicates the histological structure the object belongs to. For example, `nuclei+neoplastic` refers to a `neoplastic` object within the `nuclei` structure.

## Generating Dataset Info JSON
After preparing the preprocessed image and mask files, you can generate the dataset info JSON file, which is used for both training and testing.

You can generate the JSON file for a single dataset by running the following command:
```
cd pathseg_datasets
python create_single_dataset.py
```
> **Note:** Be sure to set the `targetpath` parameter to the path of your dataset.

If you have multiple datasets that require JSON file generation, run the following command:
```
cd pathseg_datasets
python create_multiple_dataset.py
```
Similarly, make sure to update the `dataset_paths` list with the paths to your datasets.

After the JSON files generation, your dataset directory will look like this:
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

## Register Dataset
To use these datasets, you need to register them appropriately. Open the file [register_pathseg_datasets.py](datasets/registration/register_pathseg_datasets.py), and add your `[DATASET_NAME]` to the `datasets` list. The training dataset will be registered as `pathseg_[DATASET_NAME]_train`, and the testing dataset as `pathseg_[DATASET_NAME]_test`.

Finally, add these dataset names to the configuration file [pathsegmentor.yaml](configs/pathsegmentor.yaml), around line 233.
> **Note:** Only the registered datasets in the configuration file will be used.

Once completed, your datasets will be ready for training and evaluation.

## Training
You can start trainig by runing:
```
bash scripts/train.sh
```
You may modify the `SAVE_DIR` and `RESUME_FROM` variables in [pathsegmentor.yaml](configs/pathsegmentor.yaml) as needed. These variables specify the output path for training checkpoints and the path to the pre-trained checkpoint, respectively.

## Evaluation
After training, evaluate your fine-tuned PathSegmentor by running:
```
bash scripts/eval.sh
```
You can modify the output results path in [default_trainer.py](trainer/default_trainer.py), around line 79. This is where the result JSON files will be saved.

# Citation
