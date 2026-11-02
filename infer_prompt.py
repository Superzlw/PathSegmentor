#!/usr/bin/env python
"""PathSegmentor 无 GT 推理：对单张图或一个目录里的图，用任意 prompt 分割，保存 mask / 概率图 / 叠加图。

在 PathSegmentor 仓库根目录、pathsegmentor 环境里运行：
  CUDA_VISIBLE_DEVICES=1 python infer_prompt.py <图或目录> <输出目录> "<prompt1>" ["<prompt2>" ...]
  CUDA_VISIBLE_DEVICES=1 python infer_prompt.py <图或目录> <输出目录>      # 不给 prompt = 你的 8 个组织类
                                                                          # （与 test_batch.py 相同的官方 prompt）
prompt 按官方格式写 "<结构>-level <对象> in <部位> pathology"，结构是 tissue / cell / nuclei，
例如 "tissue-level gland in colon pathology"。

处理方式与 test_batch.py 相同（官方用法），模型加载和推理直接调用 test_batch.py 里的函数：
  - 模型要求 40x 输入：--upscale 2（默认）= 输入是 20x（约 0.5 um/px），双线性放大到 40x；40x 的图用 --upscale 1。
  - 放大后按 1024x1024 滑窗推理（最后一块贴边，重叠处取平均）；短边不足 1024 时像官方 demo 一样先把短边缩放到 1024。
  - 每个 prompt 单独推理，阈值 0.5；概率图用区域平均缩回原图尺寸。
  1024 的 20x tile 用默认参数，处理与 test_batch.py 评测时完全一样。

输出（每张图）：
  <名>_<prompt>_mask.png     二值 mask（0/255）
  <名>_<prompt>_prob.png     概率 x255（能看到阈值以下的弱响应：PathSegmentor 常常整张都低于 0.5、输出空 mask）
  <名>_<prompt>_overlay.png  红色半透明叠加
  <名>_all.png               多个 prompt 时：原图 | 每个像素取概率最高的 prompt 的彩色合成（颜色见 legend.txt）
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

from test_batch import CONFIG, COLORS, DEFAULT_CKPT, IMG_EXT, NAMES, Segmenter, build_prompts, load_model

CROP = 1024
MAX_PX = 100_000_000      # 放大后超过约 1 万 x 1 万像素就不跑（整张 WSI 请先切块）
COLOR_NAMES = ["红", "绿", "黄", "蓝", "橙", "紫", "青", "品红"]


def starts(n, crop=CROP):
    """滑窗起点：步长 crop，最后一块贴边（n <= crop 时只有 0）。"""
    if n <= crop:
        return [0]
    return list(range(0, n - crop, crop)) + [n - crop]


def infer(seg, image, prompts, upscale):
    """image (H, W, 3) uint8 -> (len(prompts), H, W) 概率。"""
    torch = seg.torch
    import torch.nn.functional as F
    H, W = image.shape[:2]
    big = image if upscale == 1 else np.array(
        Image.fromarray(image).resize((W * upscale, H * upscale), Image.BILINEAR))
    bh, bw = big.shape[:2]
    if min(bh, bw) < CROP:      # 官方 demo 的 transforms.Resize(1024)：短边缩放到 1024（bicubic）
        s = CROP / min(bh, bw)
        bh, bw = max(CROP, round(bh * s)), max(CROP, round(bw * s))
        big = np.array(Image.fromarray(big).resize((bw, bh), Image.BICUBIC))
    if bh * bw > MAX_PX:
        raise ValueError(f"放大后 {bw}x{bh} 太大，像是整张 WSI；请先切成 tile")
    acc = torch.zeros((len(prompts), bh, bw))
    cnt = torch.zeros((bh, bw))
    for y in starts(bh):
        for x in starts(bw):
            acc[:, y:y + CROP, x:x + CROP] += seg.crop_probs(big[y:y + CROP, x:x + CROP], prompts).cpu()
            cnt[y:y + CROP, x:x + CROP] += 1
    acc /= cnt
    if (bh, bw) != (H, W):
        acc = F.interpolate(acc[None], size=(H, W), mode="area")[0]
    return acc.numpy()


def list_images(src):
    if os.path.isfile(src):
        return [src]
    if os.path.isdir(src):
        return sorted(os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT)
    sys.exit(f"找不到 {src}")


def tag(prompt):
    return prompt.replace(" ", "+").replace("/", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="一张图，或一个装着图的目录（不递归）")
    ap.add_argument("out", help="输出目录")
    ap.add_argument("prompts", nargs="*", help="不给 = 你的 8 个组织类（--template / --objects 决定写法）")
    ap.add_argument("--upscale", type=int, default=2, choices=[1, 2, 3, 4],
                    help="2 = 输入是 20x，放大到官方要求的 40x（默认）；1 = 输入已是 40x")
    ap.add_argument("--template", default="tissue-level {name} in colon pathology",
                    help="不给 prompt 时 8 个组织类用的模板（同 test_batch.py）")
    ap.add_argument("--objects", choices=["names", "pathseg"], default="names",
                    help="不给 prompt 时的对象词：names = 你的类名；pathseg = PathSeg 训练集用词（同 test_batch.py）")
    ap.add_argument("--thr", type=float, default=0.5, help="概率阈值（官方 0.5）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="PathSegmentor 权重 model_state_dict.pt")
    ap.add_argument("--limit", type=int, default=0, help="目录里只跑按文件名排序的前 N 张，0 = 全部")
    a = ap.parse_args()

    if not os.path.exists(CONFIG):
        sys.exit("请在 PathSegmentor 仓库根目录运行")
    files = list_images(a.src)
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"{a.src} 里没有图（支持 {sorted(IMG_EXT)}）")
    if a.prompts:
        prompts = list(dict.fromkeys(a.prompts))        # 去重，保持顺序
    else:
        d = build_prompts(a.template, False, a.objects)
        prompts = [d[c] for c in NAMES]
    for p in prompts:
        if "-level " not in p or not p.endswith(" pathology"):
            print(f"⚠ 不是官方格式 '<结构>-level <对象> in <部位> pathology'，结果可能不可靠: {p}")

    os.makedirs(a.out, exist_ok=True)
    legend = [f"{COLOR_NAMES[i % len(COLORS)]}\t{p}" for i, p in enumerate(prompts)]
    with open(os.path.join(a.out, "legend.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(legend) + "\n")
    print(f"{len(files)} 张图；upscale {a.upscale}；阈值 {a.thr}；prompt（颜色用于 _all.png）:")
    for line in legend:
        print("  " + line.replace("\t", "  "))

    seg = Segmenter(load_model(a.ckpt), prompts)
    for k, f in enumerate(files, 1):
        try:
            image = np.array(Image.open(f).convert("RGB"))
            probs = infer(seg, image, prompts, a.upscale)
        except (OSError, ValueError) as e:
            print(f"[{k}/{len(files)}] 跳过 {os.path.basename(f)}: {e}")
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        print(f"[{k}/{len(files)}] {os.path.basename(f)}  {image.shape[1]}x{image.shape[0]}")
        for i, p in enumerate(prompts):
            pr = probs[i]
            m = pr > a.thr
            base = os.path.join(a.out, f"{stem}_{tag(p)}")
            Image.fromarray(m.astype(np.uint8) * 255).save(base + "_mask.png")
            Image.fromarray(np.clip(pr * 255, 0, 255).round().astype(np.uint8)).save(base + "_prob.png")
            ovl = image.copy()
            ovl[m] = (0.5 * ovl[m] + [127, 0, 0]).astype(np.uint8)
            Image.fromarray(ovl).save(base + "_overlay.png")
            print(f"    前景 {m.mean():6.1%}  最大概率 {pr.max():.3f}  {p}")
        if len(prompts) > 1:
            comb = image.copy()
            hit, best = probs.max(0) > a.thr, probs.argmax(0)
            for i in range(len(prompts)):
                sel = hit & (best == i)
                comb[sel] = (0.5 * comb[sel] + 0.5 * COLORS[i % len(COLORS)]).astype(np.uint8)
            Image.fromarray(np.concatenate([image, comb], axis=1)).save(os.path.join(a.out, f"{stem}_all.png"))
    print(f"\n结果在 {a.out}/")


if __name__ == "__main__":
    main()
