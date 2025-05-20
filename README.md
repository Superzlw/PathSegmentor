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

PathSeg dataset was constructed from preprocessing publicly available pathology image segmentation datasets. For the source datasets, please check the details here: [PathSeg Dataset](assets/readmes/DATASET.md). 

To provide a general impression of the dataset, we showcase a subset of PathSeg dataset in [PathSeg-Demo](/home/chenzhixuan/Workspace/PathSegmentor/pathseg_datasets/Demo). You can train and evaluate the model on this dataset to understand how the code works. You may also create your own dataset by following the organizational structure of the demo dataset. 

Detailed instructions can be found in the [Fine-tuning](#fine-tuning) section.

# PathSegmentor Checkpoints
You can download the pre-trained PathSegmentor checkpoints from the following Google Drive link: [PathSegmentor Checkpoints](https://drive.google.com/drive/folders/1HxFdA9KXfS6CKgMX-i6TEVfflHiO5Tmf?usp=sharing).
 
# Inference
We have pipelined the inference process using our pre-trained PathSegmentor in the [inference notebook](inference_examples.ipynb). You can directly perform inference on the example images located in the `examples` directory. 

Remember to change the checkpoint path `pretrained_pth` in the notebook according your environment configuration.  

We provide four types of examples in the examples directory: `1_irregular`, `2_tiny`, `3_dense`, and `4_others`. These directories may contain further subcategories, but each final-level folder always includes two files: an image and a mask.

To run inference on a specific image, simply replace the `image_path` and `mask_path` variables in the notebook with the paths to your chosen image and mask files.

The `text_prompt` used for segmentation can be derived from the end of the mask filename. For example, the mask file named `Part_2_1211_pathology_bile+duct_nuclei+connective.png` corresponds to the prompt:
“nuclei-level connective in bile duct pathology”.
Be sure to remove any `+` signs when constructing the prompt.

# Fine-tuning
If you want to fine-tune the model on your own dataset, see this instructions to help you organize your dataset adhere the requirement of PathSegmentor.

# Citation
