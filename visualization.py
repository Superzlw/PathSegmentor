import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

def get_color_map(num_colors):
    """
    提取调色板的前 num_colors 种颜色。
    
    :param num_colors: 要提取的颜色数量。
    :return: 包含前 num_colors 种颜色的字典。
    """
    cmap = plt.get_cmap('Set1')  # 获取 Set1 调色板
    # Set1 调色板最多支持 9 种颜色
    max_colors = cmap.N  # 获取调色板支持的最大颜色数
    if num_colors > max_colors:
        raise ValueError(f"Set1 调色板最多支持 {max_colors} 种颜色，请减少 num_colors。")
    
    # 直接提取前 num_colors 种颜色
    colors = [cmap(i)[:3] for i in range(num_colors)]
    return {i: colors[i] for i in range(num_colors)}

def visualize_results(image_masks, vis_num, results_folder, dataset_label):
    """
    Visualize the results of the test set.
    image_masks: list of numpy arrays
    vis_num: number of images to visualize
    """
    img_names = list(image_masks.keys())
    # sort the img names
    img_names.sort()
    print(f"{dataset_label} dataset has {len(img_names)} images.")

    # uniform sample vis_num images
    if len(img_names) > vis_num:
        step = max(1, len(img_names) // vis_num)  # 确保步长至少为1
        img_names = img_names[::step]
        img_names = img_names[:vis_num]  # 确保结果数量刚好为 vis_num
    else:
        vis_num = len(img_names)
        
    save_dir = os.path.join(results_folder, 'vis', dataset_label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Visualizing {len(img_names)} images on {dataset_label} dataset...")
    
    for img_name in img_names:
        prompts = [key for key in image_masks[img_name].keys() if key != 'image']
        categories = [re.search(r'level (.+) in .*$', prompt).group(1).strip() for prompt in prompts]
        
        categories, prompts = zip(*sorted(zip(categories, prompts)))
        print(img_name)
        print(categories)
        
        num_masks = len(prompts)
        color_map = get_color_map(num_masks)
        
        image = image_masks[img_name]['image']
        height, width, _ = image.shape
        
        # get muti-label masks
        pred_masks = [image_masks[img_name][prompt]['pred'] for prompt in prompts]
        gt_masks = [image_masks[img_name][prompt]['gt'] for prompt in prompts]
        
        # 创建全零的 mask
        zero_mask = np.zeros((height, width), dtype=pred_masks[0].dtype)

        # 在列表最前面插入全零的 mask
        pred_masks.insert(0, zero_mask)
        gt_masks.insert(0, zero_mask)
        
        pred_masks = np.stack(pred_masks, axis=0)
        gt_masks = np.stack(gt_masks, axis=0)
        
        # argmax to get the predicted mask
        pred_mask = np.argmax(pred_masks, axis=0)
        gt_mask = np.argmax(gt_masks, axis=0)
        
        # 创建预测掩码的热图
        heatmap_pred = np.zeros_like(image)
        height, width = pred_mask.shape
        for i in range(height):
            for j in range(width):
                category = pred_mask[i, j]
                if category > 0: # 0 is the background (non-object)
                    heatmap_pred[i, j] = tuple(int(x * 255) for x in color_map[category-1])
                else:
                    heatmap_pred[i, j] = image[i, j] 

        # 创建真实掩码的热图
        heatmap_gt = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                category = gt_mask[i, j]
                if category > 0:
                    heatmap_gt[i, j] = tuple(int(x * 255) for x in color_map[category-1])
                else:
                    heatmap_gt[i, j] = image[i, j]

        # 混合因子
        alpha = 0.5  # 原始图像的权重
        beta = 1 - alpha

        # 将原始图像与预测掩码混合
        blended_pred = cv2.addWeighted(image, alpha, heatmap_pred, beta, 0)

        # 将原始图像与真实掩码混合
        blended_gt = cv2.addWeighted(image, alpha, heatmap_gt, beta, 0)
        
        # 创建一个包含三个子图的图形
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))

        # 绘制原始图像
        ax[0].imshow(image)
        ax[0].set_title('Original Image', fontsize=30, pad=15)
        ax[0].axis('off')

        # 绘制带预测掩码的图像
        ax[1].imshow(blended_pred)
        ax[1].set_title('Predicted Mask', fontsize=30, pad=15)
        ax[1].axis('off')

        # 绘制带真实掩码的图像
        ax[2].imshow(blended_gt)
        ax[2].set_title('Ground Truth Mask', fontsize=30, pad=15)
        ax[2].axis('off')
        
        plt.subplots_adjust(wspace=0.4)  #增加子图之间的水平间距
        
        # filtered_class_names = [gt_codes_dict[id] for id in range(num_classes) if id not in ignored_class_id]
        # filtered_color_map = {i: color_map[i] for i in range(num_classes) if i not in ignored_class_id}
        # filtered_class_ids = [id for id in range(num_classes) if id not in ignored_class_id]

        # num_colors = len(filtered_class_names)
        
        # 添加图例
        patches = []
        for i in range(0, num_masks):
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=color_map[i]))
        
        plt.figlegend(patches, categories, loc='lower center', ncol=num_masks, 
                    bbox_to_anchor=(0.5, -0.1), bbox_transform=plt.gcf().transFigure, fontsize=25)

        # 调整子图之间的间距
        plt.tight_layout()

        # 保存组合图形
        save_path = os.path.join(save_dir, img_name + '.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
                
        
        