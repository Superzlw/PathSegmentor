import os
import json
from PIL import Image
import math
from tqdm import tqdm
import shutil
import random
import numpy as np
import re


def split_and_generate_json(input_dir, train_json, test_json, train_ratio=0.8):
    os.makedirs(os.path.dirname(train_json), exist_ok=True)
    os.makedirs(os.path.dirname(test_json), exist_ok=True)
    for path in [train_json, test_json]:
        with open(path, 'w') as f:
            json.dump({"image_ids": []}, f) 

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    random.shuffle(all_files)

    train_size = int(len(all_files) * train_ratio)

    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    def get_image_ids(file_list):
        return [os.path.splitext(f)[0] for f in file_list]

    with open(train_json, 'w') as f:
        json.dump({"image_ids": get_image_ids(train_files)}, f, indent=4)

    with open(test_json, 'w') as f:
        json.dump({"image_ids": get_image_ids(test_files)}, f, indent=4)


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

def process_mask_label(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            mask = Image.open(input_path).convert("L")  
            mask_array = np.array(mask)

            mask_array_new = np.zeros_like(mask_array)
            mask_array_new[mask_array == 0] = 0
            mask_array_new[mask_array == 1] = 1
            mask_array_new[mask_array == 2] = 2
            mask_array_new[mask_array == 3] = 3
            mask_array_new[mask_array == 4] = 4
            mask_array_new[mask_array == 5] = 5
            mask_array_new[mask_array == 6] = 6
            mask_array_new[mask_array == 7] = 0
            mask_array_new[mask_array == 8] = 7
            mask_array_new[mask_array == 9] = 8
            mask_array_new[mask_array == 10] = 9
            mask_array_new[mask_array == 11] = 10
            mask_array_new[mask_array == 12] = 11
            mask_array_new[mask_array == 13] = 12
            mask_array_new[mask_array == 14] = 13
            mask_array_new[mask_array == 15] = 0
            mask_array_new[mask_array == 16] = 14
            mask_array_new[mask_array == 17] = 15
            mask_array_new[mask_array == 18] = 16
            mask_array_new[mask_array == 19] = 17
            mask_array_new[mask_array == 20] = 18
            mask_array_new[mask_array == 21] = 0

            new_mask = Image.fromarray(mask_array_new)
            new_mask.save(output_path)


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

    # 1. First, split images into patches and save patch information
    input_dir = "[RAW_DATA_DIR]/BCSS/rgbs_colorNormalized"
    output_dir = "[PROCESS_DATA_DIR]/BCSS/image"
    patch_info_path = "[PROCESS_DATA_DIR]/BCSS/patchinfo.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "[RAW_DATA_DIR]/BCSS/masks"
    output_dir = "[PROCESS_DATA_DIR]/BCSS/mask"
    patch_info_path = "[PROCESS_DATA_DIR]/BCSS/patchinfo_mask.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    # 2. Then, randomly split the dataset into training and testing sets
    input_dir = "[PROCESS_DATA_DIR]/BCSS/mask"
    train_json = "[PROCESS_DATA_DIR]/BCSS/train.json"
    test_json = "[PROCESS_DATA_DIR]/BCSS/test.json"
    with tqdm(total=len(os.listdir(input_dir)), desc="Processing images") as pbar:
        split_and_generate_json(input_dir, train_json, test_json)
        pbar.update(len(os.listdir(input_dir)))

    # 3. Process mask labels, setting outside_roi, exclude, undetermined, and other to background 0, then save as grayscale
    input_dir = "[PROCESS_DATA_DIR]/BCSS/mask"
    output_dir = "[PROCESS_DATA_DIR]/BCSS/mask_relabel"
    process_mask_label(input_dir, output_dir)

    # 4. Mask processing: Read index: Get image IDs from train.json. Extract category: Read a raw PNG image containing multiple label_ids (e.g., 1 for tumor, 2 for stroma).
    # Split layers: For each target category (e.g., tissue-tumor), set pixels belonging to that category to 1, and others to 0. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    # Modality, organ
    modality = "pathology"
    site = "breast"
    prompt_name_mapping = {
        'tissue-tumor': 1,
        'tissue-stroma': 2,
        'tissue-lymphocytic_infiltrate': 3,
        'tissue-necrosis_or_debris': 4,
        'tissue-glandular_secretions': 5,
        'tissue-blood': 6,
        'tissue-metaplasia_NOS' : 7,
        'tissue-fat' : 8,
        'tissue-plasma_cells' : 9,
        'tissue-other_immune_infiltrate' : 10,
        'tissue-mucoid_material' : 11,
        'tissue-normal_acinus_or_duct' : 12,
        'tissue-lymphatics' : 13,
        'tissue-nerve' : 14,
        'tissue-skin_adnexa' : 15,
        'tissue-blood_vessel' : 16,
        'tissue-angioinvasion' : 17,
        'tissue-dcis' : 18
    }

    mask_dir = "[PROCESS_DATA_DIR]/BCSS/mask_relabel"
    json_file = "[PROCESS_DATA_DIR]/BCSS/train.json"
    target_dir = "[PROCESS_DATA_DIR]/BCSS/train_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    mask_dir = "[PROCESS_DATA_DIR]/BCSS/mask_relabel"
    json_file = "[PROCESS_DATA_DIR]/BCSS/test.json"
    target_dir = "[PROCESS_DATA_DIR]/BCSS/test_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    # 5. Image processing: Read index: Get image IDs from train.json. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    modality = "pathology"
    site = "breast"
    image_dir = "[PROCESS_DATA_DIR]/BCSS/image"
    json_file = "[PROCESS_DATA_DIR]/BCSS/train.json"
    target_dir = "[PROCESS_DATA_DIR]/BCSS/train"
    process_images(image_dir, json_file, target_dir, modality, site)

    image_dir = "[PROCESS_DATA_DIR]/BCSS/image"
    json_file = "[PROCESS_DATA_DIR]/BCSS/test.json"
    target_dir = "[PROCESS_DATA_DIR]/BCSS/test"
    process_images(image_dir, json_file, target_dir, modality, site)