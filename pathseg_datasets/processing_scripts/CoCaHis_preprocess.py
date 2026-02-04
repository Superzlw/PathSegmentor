import h5py
import numpy as np
from PIL import Image
import os
import json
import math
from tqdm import tqdm
import re
import shutil


def split_image_into_patches(input_dir, output_dir, patch_info_path, patch_size=1024, threshold=1500):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    with open(patch_info_path, "w") as patch_info_file:
        for filename in tqdm(files, desc="Processing images", unit="image"):
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(img_path) as img:
                width, height = img.size

                if width <= threshold and height <= threshold:
                    img.save(output_path)
                    patch_info_file.write(
                        f"{filename},{filename},(0, 0),(0, {height}),({width}, 0),({width}, {height})\n")
                else:
                    patch_num = 1

                    def calculate_patches_and_overlap(dim, patch_size):
                        patches = math.ceil(dim / patch_size)
                        total_overlap = patches * patch_size - dim
                        overlap_per_patch = total_overlap // (patches - 1) if patches > 1 else 0
                        return patches, overlap_per_patch

                    x_patches, x_overlap = calculate_patches_and_overlap(width, patch_size)
                    y_patches, y_overlap = calculate_patches_and_overlap(height, patch_size)

                    for i in range(x_patches):
                        for j in range(y_patches):
                            x_start = i * (patch_size - x_overlap)
                            y_start = j * (patch_size - y_overlap)

                            x_end = min(x_start + patch_size, width)
                            y_end = min(y_start + patch_size, height)

                            patch = img.crop((x_start, y_start, x_end, y_end))

                            patch_name = f"{filename.split('.')[0]}_{patch_num}.png"
                            patch.save(os.path.join(output_dir, patch_name))

                            patch_info_file.write(
                                f"{patch_name},{filename},({x_start}, {y_start}),({x_start}, {y_end}),({x_end}, {y_start}),({x_end}, {y_end})\n")

                            patch_num += 1


def save_image(image_data, save_dir, filename):
    """
    Saves a single numpy array image as a PNG file to the specified directory.

    Args:
        image_data (np.ndarray): Single image data as a numpy array.
        save_dir (str): Target directory to save the image.
        filename (str): Name of the file to be saved.
    """
    os.makedirs(save_dir, exist_ok=True)  
    img = Image.fromarray(image_data.astype(np.uint8))
    img_path = os.path.join(save_dir, f"{filename}.png")
    img.save(img_path)


def process_and_save(hdf5_file_path, base_save_dir):
    """
    Processes HDF5 files and saves images to specified directories based on the 'train_test_split' attribute.
    """
    with h5py.File(hdf5_file_path, 'r') as f:
        split = f['HE'].attrs['train_test_split']
        
        raw_data = f['HE/raw'][:]
        mask_data = f['GT/GT_majority_vote'][:]
        
        for i in tqdm(range(len(split)), desc="Processing datasets"):
            if split[i] == 'train':
                img_type = 'train'
            else:
                img_type = 'test'
            
            raw_save_dir = os.path.join(base_save_dir, f"{img_type}_origin", 'img_raw')
            mask_save_dir = os.path.join(base_save_dir, f"{img_type}_origin", 'mask')
            
            save_image(raw_data[i], raw_save_dir, f'{i}')
            save_image(mask_data[i], mask_save_dir, f'{i}')


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
    # 1. Convert HDF5 files to PNG images
    hdf5_file_path = '[RAW_DATA_DIR]/CoCaHis/CoCaHis.hdf5'
    base_save_dir = '[RAW_DATA_DIR]/CoCaHis'
    process_and_save(hdf5_file_path, base_save_dir)

    # 2. Split patches and save patch information
    input_dir = "[RAW_DATA_DIR]/CoCaHis/train_origin/img_raw"
    output_dir = "[RAW_DATA_DIR]/CoCaHis/patch/train"
    patch_info_path = "[RAW_DATA_DIR]/CoCaHis/patch/patchinfo.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "[RAW_DATA_DIR]/CoCaHis/train_origin/mask"
    output_dir = "[RAW_DATA_DIR]/CoCaHis/patch/train_mask"
    patch_info_path = "[RAW_DATA_DIR]/CoCaHis/patch/patchinfo.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "[RAW_DATA_DIR]/CoCaHis/test_origin/img_raw"
    output_dir = "[RAW_DATA_DIR]/CoCaHis/patch/test"
    patch_info_path = "[RAW_DATA_DIR]/CoCaHis/patch/patchinfo.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "[RAW_DATA_DIR]/CoCaHis/test_origin/mask"
    output_dir = "[RAW_DATA_DIR]/CoCaHis/patch/test_mask"
    patch_info_path = "[RAW_DATA_DIR]/CoCaHis/patch/patchinfo.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    # 3、Write training and testing set information into JSON files.
    # The original dataset is already pre-split into training and testing sets; no further splitting is required.
    input_dir = "[RAW_DATA_DIR]/CoCaHis/patch/test"
    output_json = "[RAW_DATA_DIR]/CoCaHis/test/test.json"
    generate_json_from_images(input_dir, output_json)

    input_dir = "[RAW_DATA_DIR]/CoCaHis/patch/train"
    output_json = "[RAW_DATA_DIR]/CoCaHis/train/train.json"
    generate_json_from_images(input_dir, output_json)

    # 4. Mask processing: Read index: Get image IDs from train.json. Extract category: Read a raw PNG image containing multiple label_ids (e.g., 1 for tumor, 2 for stroma).
    # Split layers: For each target category (e.g., tissue-tumor), set pixels belonging to that category to 1, and others to 0. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    # Modality, organ
    modality = "pathology"
    site = "liver"
    prompt_name_mapping = {
        'tissue-metastatic_colon_cancer_in_a_liver': 1
    }

    mask_dir = "[RAW_DATA_DIR]/CoCaHis/patch/train_mask"
    json_file = "[RAW_DATA_DIR]/CoCaHis/train/train.json"
    target_dir = "[PROCESS_DATA_DIR]/CoCaHis/train_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    mask_dir = "[RAW_DATA_DIR]/CoCaHis/patch/test_mask"
    json_file = "[RAW_DATA_DIR]/CoCaHis/test/test.json"
    target_dir = "[PROCESS_DATA_DIR]/CoCaHis/test_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    # 5. Image processing: Read index: Get image IDs from train.json. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    modality = "pathology"
    site = "liver"
    image_dir = "[RAW_DATA_DIR]/CoCaHis/patch/train"
    json_file = "[RAW_DATA_DIR]/CoCaHis/train/train.json"
    target_dir = "[PROCESS_DATA_DIR]/CoCaHis/train"
    process_images(image_dir, json_file, target_dir, modality, site)

    image_dir = "[RAW_DATA_DIR]/CoCaHis/patch/test"
    json_file = "[RAW_DATA_DIR]/CoCaHis/test/test.json"
    target_dir = "[PROCESS_DATA_DIR]/CoCaHis/test"
    process_images(image_dir, json_file, target_dir, modality, site)