<!-- # PathSegmentor -->
[![header](https://capsule-render.vercel.app/api?type=waving&height=180&color=gradient&text=PathSegmentor&reversal=false&desc=Segment%20Anything%20in%20Pathology%20Images%20with%20Natural%20Language&fontSize=40&fontAlign=19&fontAlignY=22&descSize=28&descAlignY=47&descAlign=47)](https://arxiv.org/pdf/2506.20988)

<!-- [![Arxiv Page](https://img.shields.io/badge/Arxiv-2407.18449-red?style=flat-square)](https://arxiv.org/abs/2407.18449) -->
<!-- [![GitHub Stars](https://img.shields.io/github/stars/zhi-xuan-chen/PathSegmentor?style=social)](https://github.com/zhi-xuan-chen/PathSegmentor) -->
---
This repository hosts the official implementation of the research paper titled *"Segment Anything in Pathology Images with Natural Language".* It provides code, models, and resources to support natural language-guided semantic segmentation in pathology imaging.

**Try the demo:** complete [Environment Setup](#-environment-setup), then follow [Run the notebook](#-how-to-use-pathsegmentor). A sample image is included; you only need to download the model weights. No training or dataset download is required.

# 🛠 Environment Setup

The tested environment uses Python 3.11, PyTorch 2.1.0, torchvision 0.16.0, and CUDA 12.1 on Linux with an NVIDIA GPU. The notebook requires CUDA; CPU-only and Apple Silicon execution are not supported by this demo. Before installing, make sure `nvidia-smi`, `nvcc --version` (CUDA 12.1), and `g++ --version` work in your terminal. The custom Detectron2 extension needs the CUDA toolkit and a C++ compiler; the Conda CUDA runtime alone does not provide `nvcc`. Create the environment with Conda:

```bash
git clone https://github.com/zhi-xuan-chen/PathSegmentor.git
cd PathSegmentor
conda env create -f environment.yml
conda activate pathsegmentor
```

Install the model dependencies. Build isolation is disabled because the pinned custom Detectron2 build imports PyTorch during compilation:

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

Download the NLTK resources used by the language encoder:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger
```

The custom Detectron2 extension is compiled during installation, so a C++ compiler and a CUDA toolkit providing `nvcc` are required. The NVIDIA driver must support CUDA 12.1 or newer.


# 🗂️ PathSeg Dataset

The PathSeg dataset is constructed by preprocessing a collection of publicly available pathology image segmentation datasets. For detailed information about the source datasets, please refer to the corresponding documentation: [PathSeg Dataset](assets/readmes/Datasets_acquisition.md).

We also plan to open-source the processed versions of the datasets we have curated and release them in the following repository: [PathSeg on Hugging Face](https://huggingface.co/datasets/Trusure/PathSeg/). We are currently requesting redistribution permissions from the original authors/maintainers of each dataset. Once permission is granted for a dataset, we will promptly upload its processed images and masks to the [PathSeg](https://huggingface.co/datasets/Trusure/PathSeg/) repository.

We provide a one-click download script at `pathseg_datasets/download.sh`, which can retrieve most datasets, except those that require user login or additional access verification.

In addition, `assets/readmes/Datasets_processing.md` provides detailed instructions on converting raw data into the required format and directory structure.

To give a general overview of the dataset, we provide a sample subset in the `pathseg_datasets/Demo` directory. You can use this subset to train and evaluate the model, helping you understand how the code functions. Additionally, you may create your own dataset by following the same organizational structure as the `Demo` dataset and fine-tune PathSegmentor on it. Detailed instructions can be found in the [Fine-tuning](#fine-tuning-your-pathsegmentor) section.

# 💾 PathSegmentor Checkpoints
You can download the pre-trained PathSegmentor checkpoints from this Google Drive link: [PathSegmentor Checkpoints](https://drive.google.com/drive/folders/1HxFdA9KXfS6CKgMX-i6TEVfflHiO5Tmf?usp=sharing).
 
# 🚀 How to Use PathSegmentor

### 1. Prepare the checkpoint

After completing Environment Setup, run these commands from the repository root:

```bash
conda activate pathsegmentor
mkdir -p checkpoints
```

Download the pretrained `model_state_dict.pt` from [PathSegmentor Checkpoints](https://drive.google.com/drive/folders/1HxFdA9KXfS6CKgMX-i6TEVfflHiO5Tmf?usp=sharing) and save it as `checkpoints/model_state_dict.pt`. If it is already stored elsewhere, change `pretrained_pth` in the notebook's first code cell instead.

### 2. Open the notebook

Install Jupyter and register the environment once:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name pathsegmentor --display-name "PathSegmentor"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The check must print `CUDA available: True`. From the repository root, launch:

```bash
python -m jupyterlab inference_examples.ipynb
```

Select the **PathSegmentor** kernel, then choose **Run > Run All Cells**. With the checkpoint in the default location, no notebook edits are needed. The notebook uses the first visible GPU and displays the input image and predicted gland mask overlay. Model loading and the first prediction may take some time. The first load also needs internet access to download the `openai/clip-vit-base-patch32` tokenizer from Hugging Face unless it is already cached.

To choose a GPU on a machine with several GPUs, set its index before launching Jupyter, for example:

```bash
CUDA_VISIBLE_DEVICES=1 python -m jupyterlab inference_examples.ipynb
```

For VS Code, open the repository folder, open `inference_examples.ipynb`, select the **PathSegmentor** kernel, and click **Run All**. For a remote GPU server, use VS Code Remote SSH or Jupyter through an SSH tunnel; the notebook kernel must run on the GPU server.

### 3. Try your own image (optional)

All settings are together in the first code cell:

```python
pretrained_pth = 'checkpoints/model_state_dict.pt'
image_path = 'examples/1_irregular/2/test181_pathology_colon.png'
text_prompt = ['tissue-level gland in colon pathology']
```

| Setting | What to change |
| --- | --- |
| `pretrained_pth` | Path to the downloaded weights; keep the default if you used step 1. |
| `image_path` | Path to a pathology image. Relative paths are resolved from the repository root. |
| `text_prompt` | A nonempty list of English prompts describing the structure and organ to segment. Keep the square brackets even for one prompt. |

Use a 1024 × 1024 pathology patch at 40× magnification for the intended input setting. The demo pads and resizes other image sizes to 1024 × 1024; it does not infer or correct magnification. Whole-slide files must first be converted into patches.

More examples are under `examples/1_irregular`, `examples/2_tiny`, `examples/3_dense`, and `examples/4_others`. Mask filenames indicate suitable prompts: `Part_2_1211_pathology_bile+duct_nuclei+connective.png` corresponds to `['nuclei-level connective in bile duct pathology']`. Replace `+` separators with spaces. Ground-truth masks are optional reference annotations; the demo does not require a `mask_path` or display a GT panel by default.

After loading the model, you can assign a new `image_path` and `text_prompt` in a new cell and rerun the final prediction cell. If you change the checkpoint, restart the kernel and run all cells again.

### Troubleshooting

| Symptom | What to do |
| --- | --- |
| Missing checkpoint or image | Read the path in the error and update the settings cell, or place the file at that path. |
| Config file missing / `No module named modeling` | Launch Jupyter from the repository root, not its parent directory. |
| CUDA unavailable | Select the PathSegmentor kernel, check `nvidia-smi`, and check `CUDA_VISIBLE_DEVICES`. This demo requires an NVIDIA GPU. |
| Detectron2 installation fails | Check `nvcc`, the CUDA toolkit version, and the C++ compiler. Activate `pathsegmentor` and use the installation command with `--no-build-isolation` above. |
| Missing Python package or MPI library | Select the registered kernel and complete both `environment.yml` and `requirements.txt` installation. |
| Tokenizer download fails | Allow access to Hugging Face for the first run, or set `MODEL.TEXT.PRETRAINED_TOKENIZER` in `configs/pathsegmentor_inference.yaml` to a local directory containing the CLIP tokenizer files. |
| NLTK resource missing | Rerun `python -m nltk.downloader punkt averaged_perceptron_tagger` in the activated environment. |
| CUDA out of memory | Restart the kernel to release its model allocations; use a GPU with more free memory and close other GPU jobs you own. |

# Fine-tuning Your PathSegmentor
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

<!-- # Citation -->
