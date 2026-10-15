import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
#from utils.visualizer import Visualizer
# from detectron2.utils.colormap import random_color
# from detectron2.data import MetadataCatalog
# from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utilities.constants import PATHSEG_CLASSES
from inference_utils.utils import format_string
#from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
# import os
# import glob
# import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize(1024, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
#metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = ['background'] + [name.replace('-other','').replace('-merged','') 
                                for name in PATHSEG_CLASSES] + ["others"]
# colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

# use color list from matplotlib
import matplotlib.colors as mcolors
colors = dict(mcolors.TABLEAU_COLORS, **mcolors.BASE_COLORS)
colors_list = [list(colors.values())[i] for i in range(16)] 

    

@torch.no_grad()
def interactive_infer_image(model, image, prompts):

    image_ori = transform(image)
    #mask_ori = image['mask']
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    image = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    # Normalize text prompts.
    prompts = [format_string(prompt) for prompt in prompts]

    data = {"image": image, 'text': prompts, "height": height, "width": width}
    
    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = True
    model.model.task_switch['audio'] = False
    model.model.task_switch['grounding'] = True

    batch_inputs = [data]
    results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0] #NOTE: just the class embed
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
    
    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id,:,:]
    
    pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    # interpolate mask to ori size
    pred_mask_prob = F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']].sigmoid().cpu().numpy()
    pred_masks_pos = (1*(pred_mask_prob > 0.5)).astype(np.uint8)
    
    return pred_masks_pos

