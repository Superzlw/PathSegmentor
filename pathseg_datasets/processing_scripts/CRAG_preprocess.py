import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import math
import shutil
import re

def convert_to_binary_mask(input_folder, output_folder):
    """
    遍历输入文件夹，将图像转成二值灰度mask，并保存到输出文件夹。
    Args:
    - input_folder: 原始Annotation文件夹路径。
    - output_folder: 转换后的mask文件保存路径。
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有文件，并显示进度条
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for file_name in tqdm(files, desc=f"Processing {os.path.basename(input_folder)}"):
        # 构造输入和输出路径
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 打开原始图像，并将其转换为灰度模式
        img = Image.open(input_path).convert('L')  # 'L'表示灰度模式
        img_array = np.array(img)

        # 将非零像素值设为1，保持0区域为0
        binary_mask = np.where(img_array > 0, 1, 0).astype(np.uint8)

        # 保存为PNG格式的mask，只包含0和1
        mask_img = Image.fromarray(binary_mask, mode='L')  # 保持0/1的灰度值
        mask_img.save(output_path)

def upscale_mask(input_dir, output_dir, scale_factor=2):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有png文件
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    # 遍历输入目录中的所有文件，并显示进度条
    for filename in tqdm(files, desc=f"Processing {os.path.basename(input_dir)}", ncols=80):
        # 构建完整的文件路径
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开图像并进行上采样
        with Image.open(img_path) as img:
            # 计算新的尺寸
            new_size = (img.width * scale_factor, img.height * scale_factor)
            # 使用最近邻插值进行上采样
            upscaled_img = img.resize(new_size, Image.NEAREST)
            # 保存上采样后的图像
            upscaled_img.save(output_path)

def upscale_image(input_dir, output_dir, scale_factor=2):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有png文件
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    # 遍历输入目录中的所有文件，并显示进度条
    for filename in tqdm(files, desc=f"Processing {os.path.basename(input_dir)}", ncols=80):
        # 构建完整的文件路径
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开图像并进行上采样
        with Image.open(img_path) as img:
            # 计算新的尺寸
            new_size = (img.width * scale_factor, img.height * scale_factor)
            # 使用最近邻插值进行上采样
            upscaled_img = img.resize(new_size, Image.BILINEAR)
            # 保存上采样后的图像
            upscaled_img.save(output_path)


def split_image_into_patches(input_dir, output_dir, patch_info_path, patch_size=1024, threshold=1500):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中所有文件的列表
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    # 打开保存patch信息的txt文件
    with open(patch_info_path, "w") as patch_info_file:
        # 使用tqdm遍历输入目录中的所有png文件，并显示进度条
        for filename in tqdm(files, desc="Processing images", unit="image"):
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 打开图像
            with Image.open(img_path) as img:
                width, height = img.size

                # 如果长宽都小于等于阈值，直接复制图像，不切patch
                if width <= threshold and height <= threshold:
                    img.save(output_path)
                    # 写入patch信息
                    patch_info_file.write(
                        f"{filename},{filename},(0, 0),(0, {height}),({width}, 0),({width}, {height})\n")
                else:
                    patch_num = 1

                    # 计算在每个方向需要的patch数量和重叠部分
                    def calculate_patches_and_overlap(dim, patch_size):
                        patches = math.ceil(dim / patch_size)
                        total_overlap = patches * patch_size - dim
                        overlap_per_patch = total_overlap // (patches - 1) if patches > 1 else 0
                        return patches, overlap_per_patch

                    x_patches, x_overlap = calculate_patches_and_overlap(width, patch_size)
                    y_patches, y_overlap = calculate_patches_and_overlap(height, patch_size)

                    # 生成patch
                    for i in range(x_patches):
                        for j in range(y_patches):
                            x_start = i * (patch_size - x_overlap)
                            y_start = j * (patch_size - y_overlap)

                            x_end = min(x_start + patch_size, width)
                            y_end = min(y_start + patch_size, height)

                            # 切出patch
                            patch = img.crop((x_start, y_start, x_end, y_end))

                            # 生成patch名称
                            patch_name = f"{filename.split('.')[0]}_{patch_num}.png"
                            patch.save(os.path.join(output_dir, patch_name))

                            # 写入patch信息
                            patch_info_file.write(
                                f"{patch_name},{filename},({x_start}, {y_start}),({x_start}, {y_end}),({x_end}, {y_start}),({x_end}, {y_end})\n")

                            patch_num += 1


def generate_json_for_images(input_dir, train_json, test_json):
    train_image_ids = []  # 存储包含"train"的文件名（去掉扩展名）
    test_image_ids = []   # 存储包含"test"的文件名（去掉扩展名）

    # 遍历目录中的 PNG 文件，并使用进度条显示进度
    for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
        if filename.endswith(".png"):
            # 获取文件名（不带扩展名）
            image_id = os.path.splitext(filename)[0]

            # 根据文件名中的关键字分类到对应列表
            if "train" in filename.lower():
                train_image_ids.append(image_id)
            elif "test" in filename.lower():
                test_image_ids.append(image_id)

    # 将结果写入对应的 JSON 文件
    with open(train_json, 'w') as f:
        json.dump({"image_ids": train_image_ids}, f, indent=4)

    with open(test_json, 'w') as f:
        json.dump({"image_ids": test_image_ids}, f, indent=4)

def generate_json_for_images(input_dirs, train_json, test_json):
    """
    input_dirs: 传入一个列表，例如 ["path/to/dir1", "path/to/dir2"]
    """
    train_image_ids = []
    test_image_ids = []

    # 如果只传入了一个字符串路径，自动转成列表处理，增加兼容性
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    for directory in input_dirs:
        # 针对每个目录显示进度
        for filename in tqdm(os.listdir(directory), desc=f"Processing {os.path.basename(directory)}"):
            if filename.endswith(".png"):
                image_id = os.path.splitext(filename)[0]

                if "train" in filename.lower():
                    train_image_ids.append(image_id)
                elif "test" in filename.lower():
                    test_image_ids.append(image_id)

    # 写入 JSON (保持不变)
    with open(train_json, 'w') as f:
        json.dump({"image_ids": train_image_ids}, f, indent=4)
    with open(test_json, 'w') as f:
        json.dump({"image_ids": test_image_ids}, f, indent=4)


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
    # 1、将CRAG数据集的Annotation转换为二值mask
    train_annotation = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/Annotation"
    train_mask = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/mask"
    valid_annotation = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/Annotation"
    valid_mask = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/mask"
    convert_to_binary_mask(train_annotation, train_mask)
    convert_to_binary_mask(valid_annotation, valid_mask)
    
    # 2、对生成的mask进行最近邻差值上采样
    upscale_mask("/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/mask","/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/mask")
    upscale_mask("/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/mask","/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/mask")

    # 3、对原始图像进行双线性插值上采样
    upscale_image("/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/Images","/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/image")
    upscale_image("/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/Images","/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/image")

    # 4、将上采样后的图像和mask切成1024x1024的patch
    input_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/image"
    output_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/image"
    patch_info_path = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/patchinfo_train.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/train/mask"
    output_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/mask"
    patch_info_path = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/patchinfo_train.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/image"
    output_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/valid/image"
    patch_info_path = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/valid/patchinfo_test.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    input_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/valid/mask"
    output_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/valid/mask"
    patch_info_path = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/valid/patchinfo_test.txt"
    split_image_into_patches(input_dir, output_dir, patch_info_path)

    # 5、生成train和test的json文件
    dirs_to_process = [
    "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/image",
    "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/valid/image" 
    ]
    generate_json_for_images(
        dirs_to_process,
        "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train.json",
        "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/test.json"
    )   

    # 6. Mask processing: Read index: Get image IDs from train.json. Extract category: Read a raw PNG image containing multiple label_ids (e.g., 1 for tumor, 2 for stroma).
    # Split layers: For each target category (e.g., tissue-tumor), set pixels belonging to that category to 1, and others to 0. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    # Modality, organ
    modality = "pathology"
    site = "colon"
    prompt_name_mapping = {
        'tissue-gland': 1
    }

    mask_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/mask"
    json_file = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train.json"
    target_dir = "/jhcnas5/chenzhixuan/data/PathSeg/Preprocessed/rebattle_test/CRAG/train_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    mask_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/test/mask"
    json_file = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/test.json"
    target_dir = "/jhcnas5/chenzhixuan/data/PathSeg/Preprocessed/rebattle_test/CRAG/test_mask"
    process_binary_masks(mask_dir, json_file, target_dir, modality, site, prompt_name_mapping)

    # 7. Image processing: Read index: Get image IDs from train.json. Standardization and renaming: Resize to 1024x1024 and rename according to specific rules (including modality and organ information).
    modality = "pathology"
    site = "colon"
    image_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train/image"
    json_file = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/train.json"
    target_dir = "/jhcnas5/chenzhixuan/data/PathSeg/Preprocessed/rebattle_test/CRAG/train"
    process_images(image_dir, json_file, target_dir, modality, site)

    image_dir = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/patch/test/image"
    json_file = "/jhcnas5/chenzhixuan/data/PathSeg/14-27/CRAG-MILD-Net/CRAG/CRAG/test.json"
    target_dir = "/jhcnas5/chenzhixuan/data/PathSeg/Preprocessed/rebattle_test/CRAG/test"
    process_images(image_dir, json_file, target_dir, modality, site)