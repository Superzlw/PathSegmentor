import os
import glob
import json
from tqdm import tqdm
import argparse
from create_annotations import *

image_size = 1024

# Load PathSeg Label Base
with open('/home/chenzhixuan/Workspace/PathSegmentor/pathseg_datasets/region_label_base_new.json', 'r') as f:
    label_base = json.load(f)

# Get parent class for the names
parent_class = {}
for i in label_base:
    subnames = label_base[i].get('child', [])
    for label in subnames:
        parent_class[label] = int(i)

# Label ids of the dataset
category_ids = {label_base[i]['parent']: int(i) for i in label_base if 'parent' in label_base[i]}

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    imagepath = maskpath.replace('_mask', '')
    annotation_id = 0  # Record the annotation id for each image
    sent_id = 0
    ref_id = 0

    annotations = []  # Record the annotations info for each image
    images = []  # Record the image info for each image
    image_to_id = {}  # Record the mapping between image name and image id
    n_total = len(glob.glob(maskpath + "*.png"))
    n_errors = 0  # Record the number of errors in conversion

    def extra_annotation(ann, file_name, target):
        nonlocal sent_id, ref_id
        ann['file_name'] = file_name
        ann['split'] = keyword

        mod = file_name.split('.')[0].split('_')[-2]
        site = file_name.split('.')[0].split('_')[-1]
        site = site.replace("+", " ")
        site = site.replace("-", " ")

        task = {'target': target, 'modality': mod, 'site': site}
        if 'T1' in mod or 'T2' in mod or 'FLAIR' in mod or 'ADC' in mod:
            task['modality'] = 'MRI'
            if 'MRI' not in mod:
                task['sequence'] = mod
            else:
                task['sequence'] = mod[4:]
        
        # extract the first word of the target name
        scale = target.split(' ')[0]
        target = target[len(scale)+1:]
        
        prompts = [f'{scale}-level {target} in {site} {mod}']  # Fixed prompt
        # prompts = [f'{target} in {site} {mod}']  # Fixed prompt

        ann['sentences'] = []
        for p in prompts:
            ann['sentences'].append({'raw': p, 'sent': p, 'sent_id': sent_id})
            sent_id += 1
        ann['sent_ids'] = [s['sent_id'] for s in ann['sentences']]

        ann['ann_id'] = ann['id']
        ann['ref_id'] = ref_id
        ref_id += 1

        return ann

    for mask_image in tqdm(glob.glob(maskpath + "*.png")):
        filename_parsed = os.path.basename(mask_image).split("_")
        site = filename_parsed[-2]
        site = site.replace("+", " ")
        site = site.replace("-", " ")
        target_name = filename_parsed[-1].split(".")[0].replace("+", " ")

        original_file_name = "_".join(filename_parsed[:-1]) + ".png"

        if original_file_name not in os.listdir(imagepath):
            print("Original file not found: {}".format(original_file_name))
            n_errors += 1
            continue

        if original_file_name not in image_to_id:
            image_to_id[original_file_name] = len(image_to_id)
            image_id = image_to_id[original_file_name]
            image = create_image_annotation(original_file_name, image_size, image_size, image_id)
            images.append(image)

        annotation = {
            "mask_file": os.path.basename(mask_image),
            "iscrowd": 0,
            "image_id": image_to_id[original_file_name],
            "category_id": parent_class[site + " " + target_name], #NOTE: The target name should be use the site as prefix, to ensure the uniqueness of the target name
            "id": annotation_id,
        }

        annotation = extra_annotation(annotation, original_file_name, target_name)
        annotations.append(annotation)
        annotation_id += 1

    return images, annotations, annotation_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset annotations.")
    parser.add_argument('--targetpath', type=str, help="Path to the dataset folder", 
                        default="/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/WSSS4LUAD")
    args = parser.parse_args()

    targetpath = args.targetpath

    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    for keyword in ['train', 'test']:
        mask_path = os.path.join(targetpath, f"{keyword}_mask/")
        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)

        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        # Post-process file
        images_with_ann = set()
        for ann in coco_format['annotations']:
            images_with_ann.add(ann['file_name'])
        for im in coco_format['images']:
            if im["file_name"] not in images_with_ann:
                coco_format['images'].remove(im)

        with open(os.path.join(targetpath, f"{keyword}_new.json"), "w") as outfile:
            json.dump(coco_format, outfile)

        print(f"Created {annotation_cnt} annotations for {len(coco_format['images'])} images in folder: {mask_path}")