import os
import scipy.io as sio
import numpy as np
import cv2
from tqdm import tqdm
import json
import shutil
from PIL import Image
import re

def convert_type_map_to_png(input_folder: str, output_folder: str):
    """
    Converts the 'type_map' from .mat files in a specified folder to grayscale PNG format and saves them.

    Args:
        input_folder (str): Path to the folder containing the input .mat files.
        output_folder (str): Path to the folder where the output PNG files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    mat_files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]

    for mat_file in tqdm(mat_files, desc="Converting masks"):
        mat_path = os.path.join(input_folder, mat_file)
        png_filename = os.path.splitext(mat_file)[0] + ".png"
        png_path = os.path.join(output_folder, png_filename)

        data = sio.loadmat(mat_path)

        if 'type_map' not in data:
            print(f"Warning: 'type_map' not found in {mat_file}. Skipping...")
            continue

        type_map = data['type_map'].astype(np.uint8)

        cv2.imwrite(png_path, type_map)

    print("All masks have been converted and saved.")


def generate_json_from_images(input_dir, output_json):
    """
    Generate a JSON index file from PNG images in the specified directory.
    If the output file or its parent directory does not exist, they will be created.
    """
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_ids = []

    if not os.path.exists(input_dir):
        print(f"Warning: Input directory {input_dir} not found.")
        return

    for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
        if filename.endswith(".png"):
            image_ids.append(os.path.splitext(filename)[0])

    with open(output_json, 'w') as f:
        json.dump({"image_ids": image_ids}, f, indent=4)
    
    print(f"Successfully saved {len(image_ids)} IDs to {output_json}")

def remap_mask_labels(input_dir, output_dir):
    """
    Remaps category label IDs in mask images and saves the results as grayscale PNGs.

    This function shifts label values (e.g., converting 1 to background 0, and decrementing 
    other tissue categories) to align the dataset with specific model class requirements.

    Args:
        input_dir (str): Path to the directory containing the original mask images.
        output_dir (str): Path to the directory where remapped masks will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            mask = Image.open(input_path).convert("L")
            mask_array = np.array(mask)

            mask_array_new = np.zeros_like(mask_array)
            mask_array_new[mask_array == 1] = 0
            mask_array_new[mask_array == 2] = 1
            mask_array_new[mask_array == 3] = 2
            mask_array_new[mask_array == 4] = 3
            mask_array_new[mask_array == 5] = 4
            mask_array_new[mask_array == 6] = 5
            mask_array_new[mask_array == 7] = 6

            new_mask = Image.fromarray(mask_array_new)
            new_mask.save(output_path)

    print(f"Processing complete. Remapped masks saved to: {output_dir}")


def process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping):
    """
    Processes mask image files by extracting specific label IDs to generate and save binary masks.

    Args:
        mask_dir (str): Directory where the original mask image files are located.
        json_file (str): Path to the JSON file containing the list of image names (IDs).
        target_dir (str): Directory where the processed binary masks will be saved.
        modality (str): Imaging modality (e.g., "pathology").
        site (str): Anatomical site or organ (e.g., "breast" or "multiple").
        prompt_name_mapping (dict): A dictionary mapping target names to their corresponding label IDs.
    """
    with open(json_file, 'r') as f:
        mask_names = json.load(f)["image_ids"]

    for mask_name in tqdm(mask_names, desc="Processing masks", ncols=100):
        mask_file = f"{mask_name}.png"
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path): 
            mask = Image.open(mask_path)

            mask_array = np.array(mask)

            for target, label_id in prompt_name_mapping.items():
                if label_id in np.unique(mask_array):  
                    binary_mask = np.where(mask_array == label_id, 1, 0)

                    clean_mask_name = re.sub(r'[^A-Za-z0-9]+', '', mask_name)  
                    clean_target = re.sub(r'[^A-Za-z0-9]+', '+', target) 

                    binary_mask_file_name = f"{clean_mask_name}_{modality}_{site}_{clean_target}.png"

                    mask_output_path = os.path.join(target_dir, binary_mask_file_name)

                    os.makedirs(target_dir, exist_ok=True)

                    binary_mask_image = Image.fromarray(binary_mask.astype(np.uint8)) 

                    if binary_mask_image.size != (1024, 1024):
                        binary_mask_image = binary_mask_image.resize((1024, 1024), Image.NEAREST)

                    binary_mask_image.save(mask_output_path)

        else:
            print(f"Warning: Mask file {mask_file} does not exist in {mask_dir}.")

    print("Mask processing completed.")

def process_images(image_dir, json_file, target_dir, modality, site):
    """
    Processes image files and saves them according to specified naming conventions.

    Args:
        image_dir (str): Directory where the source image files are located.
        json_file (str): Path to the JSON file containing the list of image names (IDs).
        target_dir (str): Directory where the processed images will be saved.
        modality (str): Imaging modality (e.g., "pathology").
        site (str): Anatomical site or organ (e.g., "breast" or "multiple").
    """
    with open(json_file, 'r') as f:
        image_names = json.load(f)["image_ids"]

    for image_name in tqdm(image_names, desc="Processing images", ncols=100):
        img_file = f"{image_name}.png"
        img_path = os.path.join(image_dir, img_file)

        if os.path.exists(img_path):  
            image = Image.open(img_path)

            clean_image_name = re.sub(r'[^A-Za-z0-9]+', '', image_name)

            image_file_name = f"{clean_image_name}_{modality}_{site}.png"

            image_output_path = os.path.join(target_dir, image_file_name)

            if image.size != (1024, 1024):
                image = image.resize((1024, 1024), Image.BILINEAR)

            os.makedirs(target_dir, exist_ok=True)

            image.save(image_output_path)
        else:
            print(f"Warning: Image file {img_file} does not exist in {image_dir}.")

    print("Image processing completed.")


if __name__ == "__main__":

    # 1、 Converts the 'type_map' from .mat files in a specified folder to grayscale PNG format and saves them.
    input_folder = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Test/Labels"
    output_folder = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Test/mask"
    convert_type_map_to_png(input_folder, output_folder)

    input_folder = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Train/Labels"
    output_folder = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Train/mask"
    convert_type_map_to_png(input_folder, output_folder)

    # 2、 The original images are 1000x1000, so patching is not required. 
    # The dataset already has predefined training and testing splits; 
    # direct recording of these splits is sufficient.
    input_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Test/Images"
    output_json = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/test.json"
    generate_json_from_images(input_dir, output_json)

    input_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Train/Images"
    output_json = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/train.json"
    generate_json_from_images(input_dir, output_json)

    # 3、 Remaps category label IDs in mask images and saves the results as grayscale PNGs.
    input_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Test/mask"
    output_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/test_mask"
    remap_mask_labels(input_dir, output_dir)

    input_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Train/mask"
    output_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/train_mask"
    remap_mask_labels(input_dir, output_dir)

    # 4. Mask processing: Read index: Get image IDs from train.json. Extract category: Read a raw PNG image containing multiple label_ids (e.g., 1 for tumor, 2 for stroma).
    # Split layers: For each target category (e.g., tissue-tumor), set pixels belonging to that category to 1, and others to 0. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    # Modality, organ
    modality = "pathology"
    site = "colon"
    prompt_name_mapping = {
        'cell-inflammatory': 1,
        'cell-healthy_epithelial': 2,
        'cell-dysplastic/malignant_epithelial': 3,
        'cell-fibroblast': 4,
        'cell-muscle': 5,
        'cell-endothelial': 6
    }

    mask_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/train_mask"
    json_file = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/train.json"
    target_dir = "[PROCESS_DATA_DIR]/CoNSeP/train_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    mask_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/test_mask"
    json_file = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/test.json"
    target_dir = "[PROCESS_DATA_DIR]/CoNSeP/test_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    # 5. Image processing: Read index: Get image IDs from train.json. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    modality = "pathology"
    site = "colon"
    image_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Train/Images"
    json_file = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/train.json"
    target_dir = "[PROCESS_DATA_DIR]/CoNSeP/train"
    process_images(image_dir, json_file, target_dir, modality, site)

    image_dir = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/raw/CoNSeP/Test/Images"
    json_file = "[RAW_DATA_DIR]/CoNSeP-HoVer-Net/CoNSeP/data/OpenDataLab___CoNSeP/test.json"
    target_dir = "[PROCESS_DATA_DIR]/CoNSeP/test"
    process_images(image_dir, json_file, target_dir, modality, site)