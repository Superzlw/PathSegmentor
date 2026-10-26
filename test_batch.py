#!/usr/bin/env python
"""PathSegmentor 组织分割批量评测（UKK_Col 数据集）。

在 PathSegmentor 仓库根目录、pathsegmentor 环境里运行：
  python test_batch.py --selftest                          # 只测 Dice 计算逻辑，几秒，不用 GPU
  python test_batch.py --check                             # 只检查数据（配对/尺寸/标签分布），不加载模型
  CUDA_VISIBLE_DEVICES=1 python test_batch.py --sanity     # 用仓库自带的示例图(带 GT)确认模型加载和推理正常
  CUDA_VISIBLE_DEVICES=1 python test_batch.py --limit 50   # 随机抽 50 张试跑
  CUDA_VISIBLE_DEVICES=1 python test_batch.py              # 全量

评测协议与 BiomedParse 仓库里的 test_batch.py 相同，两边结果可以逐行对照：
每张图只对 GT 中出现的类别发 prompt，Dice 只在该类别出现的图上统计，
衡量的是"已知目标存在时分得准不准"，不衡量"能否判断目标不存在"。
加 --all-prompts 则每张图都发全部 8 个 prompt，额外报告把误检也算进去的 Dice（all 列）。
每个 prompt 单独推理（同一张图只跑一次图像编码器），prompt 之间互不影响，
所以 --all-prompts 时 mean/pooled 两列与不加时完全相同。

像素规则：未标注(0) 永远不参与计算；背景(11) 不作为类别评测，但模型在背景上的预测默认算误报；
加 --ignore-background 则背景像素也完全不参与计算。

倍率：PathSegmentor 训练用的是 40x（约 0.25 um/px）的 1024x1024 patch。UKK_Col 的 tile 文件名里
d=2.16945、w=2222 表示从约 0.23 um/px 的原图降采样到约 0.5 um/px，所以默认 --upscale 2：
把 1024 的 tile 放大到 2048，切成 4 块 1024 分别推理，概率图拼回后再缩回 1024 计算 Dice。
--upscale 1 = 直接按 1024 输入（与 BiomedParse 那边的做法相同）。

输出（--out 目录）：config.json（参数与 prompt）、per_tile.csv（每张图每个类别的像素数与 Dice）、
summary.csv（每类与各组的汇总）、vis/（--save-vis 时的对比图：原图 | GT | 预测）。
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image

DEFAULT_DATA = "/data/02_ZW/Raw_Dataset/VPR/Test_DS/UKK_Col/07_Tiles"
DEFAULT_CKPT = "/data/02_ZW/ckpts/PathSegmentor/model_state_dict.pt"
CONFIG = "configs/pathsegmentor_inference.yaml"
VOCAB_CSV = "pathseg_datasets/hierachical_labels_and_prompts.csv"   # PathSeg 的 160 条训练 prompt
CROP = 1024

# 类别名: (mask 标签值, PathSeg 训练集里同一概念的原文 prompt；None = 训练标签里没有这个概念)
# 训练集的 colon 组织级只有 gland / malignant lesion，所以多数对应来自别的器官（BCSS 乳腺、SegPath 不分器官），
# "in-training" 指概念出现过，不代表在结肠上训练过。可按需替换成下面注释里的其他训练原文：
#   epithelial tissue: "tissue-level malignant lesion in colon pathology" / "tissue-level gland in colon pathology"
#   connective tissue: "tissue-level tumor associated stroma tissue in lung pathology"
#   erythrocytes:      "cell-level red blood cells in unspecified pathology"
CLASSES = {
    "epithelial tissue":        ((1,),      "tissue-level epithelial in unspecified pathology"),
    "colorectal mucosa":        ((2,),      None),
    "connective tissue":        ((3, 4, 6), "tissue-level stroma in breast pathology"),
    "muscle tissue":            ((5,),      "tissue-level smooth muscle cells in unspecified pathology"),
    "dense lymphoid aggregate": ((7,),      "tissue-level lymphocytic infiltrate in breast pathology"),
    "necrotic debris":          ((8,),      "tissue-level necrosis or debris in breast pathology"),
    "erythrocytes":             ((9,),      "tissue-level blood in breast pathology"),
    "mucin":                    ((10,),     "tissue-level mucoid material in breast pathology"),
}
UNANNOTATED, BACK = 0, 11

NAMES = list(CLASSES)
IN_TRAIN = [c for c in NAMES if CLASSES[c][1] is not None]
ABSENT = [c for c in NAMES if CLASSES[c][1] is None]
GROUPS = {"in-training": IN_TRAIN, "absent-from-training": ABSENT, "ALL": NAMES}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLORS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                   [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.uint8)
NAN = float("nan")


# ---------------- Dice 计算（--selftest 测的就是这几个函数，主流程也调用同样的函数） ----------------

def dice(tp, p, g):
    return 2.0 * tp / (p + g) if (p + g) > 0 else NAN


def valid_region(mask, ignore_bg):
    """参与计算的像素：去掉未标注；--ignore-background 时再去掉背景。"""
    v = mask != UNANNOTATED
    if ignore_bg:
        v &= mask != BACK
    return v


def tile_counts(mask, pred, labels, valid):
    """返回 (tp, 预测像素数, GT 像素数)，只在 valid 区域内计数。"""
    gt = np.isin(mask, labels) & valid
    pr = pred & valid
    return int((pr & gt).sum()), int(pr.sum()), int(gt.sum())


def new_stat():
    return {"n": 0, "dsum": 0.0, "tp": 0, "p": 0, "g": 0, "tp_all": 0, "p_all": 0, "g_all": 0}


def add(stat, tp, p, g):
    stat["tp_all"] += tp
    stat["p_all"] += p
    stat["g_all"] += g
    if g > 0:                       # 只有 GT 里有这个类别的图才进入 mean / pooled
        stat["n"] += 1
        stat["dsum"] += dice(tp, p, g)
        stat["tp"] += tp
        stat["p"] += p
        stat["g"] += g


def summarize(stats, all_prompts):
    per_class = {}
    for c in NAMES:
        s = stats[c]
        per_class[c] = {
            "tiles": s["n"],
            "mean": s["dsum"] / s["n"] if s["n"] else NAN,
            "pooled": dice(s["tp"], s["p"], s["g"]),
            "all": dice(s["tp_all"], s["p_all"], s["g_all"]) if all_prompts else NAN,
        }
    groups = {}
    for g, members in GROUPS.items():
        cs = [c for c in members if stats[c]["n"] > 0]      # 测试集里没出现过的类别不进入总体

        def tot(k):
            return sum(stats[c][k] for c in cs)

        groups[g] = {
            "classes": len(cs),
            "macro": float(np.mean([per_class[c]["mean"] for c in cs])) if cs else NAN,
            "micro": dice(tot("tp"), tot("p"), tot("g")),
            "micro_all": dice(tot("tp_all"), tot("p_all"), tot("g_all")) if all_prompts else NAN,
        }
    return per_class, groups


def example_prompt(mask_file):
    """示例 mask 文件名 -> 训练格式的 prompt，如 ..._bile+duct_nuclei+connective.png ->
    'nuclei-level connective in bile duct pathology'（与 README 的说明一致）。"""
    parts = os.path.splitext(os.path.basename(mask_file))[0].split("_")
    site = parts[-2].replace("+", " ")
    words = parts[-1].split("+")
    return f"{words[0]}-level {' '.join(words[1:])} in {site} pathology"


def selftest():
    m = np.array([[1, 1, 2, 0],
                  [3, 4, 6, 11],
                  [5, 5, 11, 11]])
    # epithelial：GT 2 个像素；预测命中 1 个、漏 1 个，另在未标注(0)和背景(11)上各误报 1 个
    pred = np.zeros(m.shape, bool)
    pred[0, 0] = pred[0, 3] = pred[1, 3] = True
    assert tile_counts(m, pred, (1,), valid_region(m, False)) == (1, 2, 2)   # 未标注不计，背景算误报
    assert tile_counts(m, pred, (1,), valid_region(m, True)) == (1, 1, 2)    # 背景也不计
    # connective = 标签 3/4/6 合并；整张图都预测成它
    allp = np.ones(m.shape, bool)
    assert tile_counts(m, allp, (3, 4, 6), valid_region(m, False)) == (3, 11, 3)
    assert tile_counts(m, allp, (3, 4, 6), valid_region(m, True)) == (3, 8, 3)

    st = {c: new_stat() for c in NAMES}
    add(st["epithelial tissue"], 4, 5, 5)    # 单张 Dice 0.8
    add(st["epithelial tissue"], 4, 5, 5)    # 0.8
    add(st["epithelial tissue"], 0, 3, 0)    # GT 里没有：不进 mean/pooled，只进 all（误检）
    add(st["colorectal mucosa"], 1, 5, 5)    # 0.2
    pc, gr = summarize(st, all_prompts=True)

    def close(a, b):
        return abs(a - b) < 1e-9

    e = pc["epithelial tissue"]
    assert e["tiles"] == 2 and close(e["mean"], 0.8) and close(e["pooled"], 0.8)
    assert close(e["all"], 16 / 23)                          # 2*8 / (13+10)
    assert close(gr["ALL"]["macro"], 0.5)                    # (0.8 + 0.2) / 2
    assert close(gr["ALL"]["micro"], 18 / 30)                # 2*(8+1) / ((10+5)+(10+5))
    assert gr["in-training"]["classes"] == 1 and gr["absent-from-training"]["classes"] == 1
    assert close(gr["in-training"]["macro"], 0.8) and close(gr["absent-from-training"]["macro"], 0.2)
    assert pc["muscle tissue"]["tiles"] == 0 and pc["muscle tissue"]["mean"] != pc["muscle tissue"]["mean"]

    assert example_prompt("Part_2_1211_pathology_bile+duct_nuclei+connective.png") == \
        "nuclei-level connective in bile duct pathology"
    assert example_prompt("x/67test188_pathology_lung_tissue+tumor+associated+stroma+tissue.png") == \
        "tissue-level tumor associated stroma tissue in lung pathology"
    vocab = load_vocab()
    for c in IN_TRAIN:
        assert CLASSES[c][1] in vocab, f"不是训练原文: {CLASSES[c][1]}"
    print("selftest OK")


# ---------------- 数据 ----------------

def load_mask(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:   # 用 RGB(A) 存的标签图：前三个通道必须相同
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError(f"mask 是彩色图，无法直接读成标签: {path}")
        a = a[..., 0]
    return a.astype(np.int64)


def list_by_stem(d):
    out = {}
    for f in os.listdir(d):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT:
            out[stem] = os.path.join(d, f)
    return out


def pair_files(img_dir, mask_dir):
    imgs, masks = list_by_stem(img_dir), list_by_stem(mask_dir)
    common = sorted(imgs.keys() & masks.keys())
    return [(imgs[k], masks[k]) for k in common], imgs.keys() - masks.keys(), masks.keys() - imgs.keys()


def label_names():
    names = {UNANNOTATED: "unannotated", BACK: "background"}
    for c, (labels, _) in CLASSES.items():
        for v in labels:
            names[v] = c
    return names


def check(pairs, only_img, only_mask, n=50):
    print(f"配对成功 {len(pairs)} 对；只有图像 {len(only_img)}；只有 mask {len(only_mask)}")
    for title, s in (("只有图像", only_img), ("只有 mask", only_mask)):
        if s:
            print(f"  {title} 示例: {sorted(s)[:3]}")
    if not pairs:
        return
    sample = random.Random(0).sample(pairs, min(n, len(pairs)))
    sizes, pix, tiles = Counter(), Counter(), Counter()
    for ip, mp in sample:
        im, m = Image.open(ip), load_mask(mp)
        sizes[f"image {im.size[0]}x{im.size[1]} {im.mode} / mask {m.shape[1]}x{m.shape[0]}"] += 1
        u, cnt = np.unique(m, return_counts=True)
        for v, k in zip(u.tolist(), cnt.tolist()):
            pix[v] += k
            tiles[v] += 1
    print(f"随机抽 {len(sample)} 张的尺寸:")
    for s, k in sizes.most_common():
        print(f"  {s}: {k}")
    names, total = label_names(), sum(pix.values())
    print("标签分布（像素占比 / 出现在几张图里）:")
    for v in sorted(pix):
        print(f"  {v:>3} {names.get(v, '??? 未定义'):<26}{pix[v] / total:8.2%}  {tiles[v]}/{len(sample)}")
    unknown = sorted(v for v in pix if v not in names)
    if unknown:
        print(f"⚠ 发现未定义的标签值 {unknown}，先确认 mask 格式再跑评测")


def load_vocab():
    with open(VOCAB_CSV, newline="", encoding="utf-8") as f:
        return {r["Text Prompts"].strip().lower() for r in csv.DictReader(f)}


# ---------------- 模型 ----------------

def load_model(ckpt):
    import torch
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import PATHSEG_CLASSES
    from utilities.model import align_and_update_state_dicts

    opt = init_distributed(load_opt_from_config_files([CONFIG]))
    model = BaseModel(opt, build_model(opt))

    # 等同于 BaseModel.from_pretrained（strict=False），但把逐个张量的日志换成一行汇总，并在明显加载失败时退出
    model_sd = model.model.state_dict()
    ckpt_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    logging.getLogger("utilities.model").setLevel(logging.ERROR)
    aligned = align_and_update_state_dicts(model_sd, ckpt_sd)
    missing = [k for k in model_sd if k not in aligned]
    unused = [k for k in ckpt_sd if k not in aligned]
    model.model.load_state_dict(aligned, strict=False)
    print(f"权重 {ckpt}: 加载 {len(aligned)}/{len(model_sd)} 个张量；"
          f"模型中未加载 {len(missing)} 个；ckpt 中未使用 {len(unused)} 个")
    for title, keys in (("未加载", missing), ("未使用", unused)):
        prefixes = Counter(".".join(k.split(".")[:4]) for k in keys)
        for pfx, k in prefixes.most_common(6):
            print(f"  {title} {pfx}.* : {k}")
    if not any(k.startswith("backbone.") for k in aligned):
        sys.exit("backbone 权重一个都没加载上，检查 --ckpt 是否是 PathSegmentor 的 model_state_dict.pt")

    model = model.eval().cuda()
    m = model.model
    for k in ("spatial", "visual", "audio"):      # 与 inference_utils/inference.py 的设置相同：只用文本 grounding
        m.task_switch[k] = False
    m.task_switch["grounding"] = True
    with torch.no_grad():
        m.sem_seg_head.predictor.lang_encoder.get_text_embeddings(PATHSEG_CLASSES + ["background"], is_eval=True)
    return model


class Segmenter:
    """复现 inference_utils/inference.py 的 interactive_infer_image（单个 prompt），
    区别只是同一张图的图像特征只算一次、各 prompt 分别跑解码器。--sanity 会和官方函数逐像素比对。"""

    def __init__(self, model, prompts):
        import torch
        from inference_utils.utils import format_string
        self.torch = torch
        self.model = model
        self.m = model.model
        self.lang = self.m.sem_seg_head.predictor.lang_encoder
        self.text = {}
        with torch.no_grad():
            for p in set(prompts):
                g = self.lang.get_text_token_embeddings([format_string(p)], name="grounding", token=False, norm=False)
                tok = g["token_emb"][g["tokens"]["attention_mask"].bool()]          # (n_token, C)
                extra = {"grounding_tokens": tok[:, None],
                         "grounding_nonzero_mask": torch.zeros(tok.shape[0], 1, dtype=torch.bool,
                                                               device=tok.device).t()}
                cls = g["class_emb"]
                self.text[p] = (extra, cls / (cls.norm(dim=-1, keepdim=True) + 1e-7))

    def crop_probs(self, crop, prompts):
        """crop: (h, w, 3) uint8 -> (len(prompts), h, w) 概率（GPU 张量）。"""
        torch = self.torch
        import torch.nn.functional as F
        from detectron2.structures import ImageList
        from modeling.language.loss import vl_similarity
        m = self.m
        h, w = crop.shape[:2]
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).to(m.device).float()
            x = (x - m.pixel_mean) / m.pixel_std
            images = ImageList.from_tensors([x], m.size_divisibility)
            feats = m.backbone(images.tensor)
            mask_features, _, multi_scale = m.sem_seg_head.pixel_decoder.forward_features(feats)
            out = []
            for p in prompts:
                extra, t_emb = self.text[p]
                o = m.sem_seg_head.predictor(multi_scale, mask_features, target_queries=None,
                                             extra=dict(extra), task="demo")
                v = o["pred_captions"][0]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-7)
                q = vl_similarity(v, t_emb, temperature=self.lang.logit_scale)[:, 0].argmax()
                mk = F.interpolate(o["pred_masks"][0, q][None, None], size=images.tensor.shape[-2:],
                                   mode="bilinear")[0, 0, :h, :w]
                out.append(mk.sigmoid())
            return torch.stack(out)


def predict(seg, image, prompts, upscale):
    """返回 (len(prompts), H, W) 概率图。upscale>1 时先放大、按 1024 切块推理、拼回后用区域平均缩回原尺寸。"""
    torch = seg.torch
    import torch.nn.functional as F
    H, W = image.shape[:2]
    big = image if upscale == 1 else np.array(
        Image.fromarray(image).resize((W * upscale, H * upscale), Image.BILINEAR))
    BH, BW = big.shape[:2]
    acc = torch.zeros((len(prompts), BH, BW), device=seg.m.device)
    for y in range(0, BH, CROP):
        for x in range(0, BW, CROP):
            acc[:, y:y + CROP, x:x + CROP] = seg.crop_probs(big[y:y + CROP, x:x + CROP], prompts)
    if upscale > 1:
        acc = F.interpolate(acc[None], size=(H, W), mode="area")[0]
    return acc.cpu().numpy()


def build_prompts(template, trained):
    out = {}
    for c in NAMES:
        tp = CLASSES[c][1]
        out[c] = tp if (trained and tp) else template.format(name=c)
    return out


def example_pairs(root="examples"):
    """仓库自带示例：每个叶子目录里最短的文件名是原图，其余以它为前缀的是对应 mask。"""
    pairs = []
    for d, _, files in os.walk(root):
        pngs = sorted(f for f in files if f.lower().endswith(".png"))
        if len(pngs) < 2:
            continue
        img = min(pngs, key=len)
        stem = os.path.splitext(img)[0]
        for f in pngs:
            if f != img and f.startswith(stem + "_"):
                pairs.append((os.path.join(d, img), os.path.join(d, f), example_prompt(f)))
    return sorted(pairs)


def sanity(seg, thr):
    from inference_utils.inference import interactive_infer_image
    pairs = example_pairs()
    if not pairs:
        sys.exit("examples/ 下没有找到示例图，确认在仓库根目录运行")
    print(f"\n{len(pairs)} 个示例（仓库自带，40x、1024，直接输入，不放大）:")
    by_level = {}
    for k, (ip, mp, prompt) in enumerate(pairs):
        image = np.array(Image.open(ip).convert("RGB"))
        gt = np.array(Image.open(mp).convert("L")) > 0
        prob = predict(seg, image, [prompt], upscale=1)[0]
        pred = prob > thr
        d = dice(int((pred & gt).sum()), int(pred.sum()), int(gt.sum()))
        by_level.setdefault(prompt.split("-level")[0], []).append(d)
        print(f"  {d:6.4f}  {prompt}")
        if k == 0:   # 自己的实现与官方推理函数逐像素比对（阈值 0.5，与官方相同）
            ref = interactive_infer_image(seg.model, Image.fromarray(image), [prompt])[0] > 0
            diff = int((ref != (prob > 0.5)).sum())
            print(f"  ↑ 与官方 interactive_infer_image 的结果相差 {diff} 个像素（应为 0 或极少）")
    for lvl, ds in sorted(by_level.items()):
        print(f"{lvl:>8}: 平均 Dice {np.nanmean(ds):.4f}  ({len(ds)} 个)")
    print("这些示例来自 PathSeg 的数据集；平均 Dice 很低（例如 < 0.3）说明权重或环境有问题，先别跑正式评测。")


def save_vis(path, image, mask, valid, probs, prompted, thr):
    gt = np.full(image.shape, 128, np.uint8)      # 灰 = 不参与计算
    gt[valid] = 0                                  # 黑 = 参与计算的背景
    for i, c in enumerate(NAMES):
        gt[np.isin(mask, CLASSES[c][0])] = COLORS[i]
    pr = np.zeros_like(image)
    if prompted:
        hit, best = probs.max(0) > thr, probs.argmax(0)
        for j, c in enumerate(prompted):
            pr[hit & (best == j)] = COLORS[NAMES.index(c)]
    Image.fromarray(np.concatenate([image, gt, pr], axis=1)).save(path)


# ---------------- 报告 ----------------

def fmt(x):
    return f"{x:8.4f}" if x == x else f"{'-':>8}"


def num(x):
    return f"{x:.4f}" if x == x else ""


def report(per_class, groups, prompts, in_vocab, all_prompts, out_dir):
    pw = max(len(p) for p in prompts.values()) + 2
    cols = ["mean", "pooled"] + (["all"] if all_prompts else [])
    head = f"{'class':<26}{'group':<10}{'prompt':<{pw}}{'tiles':>6}" + "".join(f"{c:>8}" for c in cols)
    print("\n" + head + "\n" + "-" * len(head))
    rows = []
    for c in NAMES:
        r = per_class[c]
        grp = "in-train" if c in IN_TRAIN else "absent"
        vals = [r["mean"], r["pooled"]] + ([r["all"]] if all_prompts else [])
        print(f"{c:<26}{grp:<10}{prompts[c]:<{pw}}{r['tiles']:>6}" + "".join(fmt(v) for v in vals))
        rows.append(["class", c, grp, prompts[c], r["tiles"], num(r["mean"]), num(r["pooled"]), num(r["all"]),
                     int(in_vocab[c])])
    print("-" * len(head))
    for g, r in groups.items():
        vals = [r["macro"], r["micro"]] + ([r["micro_all"]] if all_prompts else [])
        label = f"({r['classes']} classes)"
        print(f"{g:<26}{'':<10}{label:<{pw}}{'':>6}" + "".join(fmt(v) for v in vals))
        rows.append(["group", g, "", "", r["classes"], num(r["macro"]), num(r["micro"]), num(r["micro_all"]), ""])
    print("\nmean   = 每张图 Dice 的平均；组那几行是各类 mean 再取平均（macro，每类权重相同）")
    print("pooled = 所有图的像素合并后算一个 Dice；组那几行是跨类别合并（micro，大类权重大）")
    if all_prompts:
        print("all    = 同 pooled，但把 GT 里没有该类别的图也算进去（误检会拉低它）")
    print("tiles  = GT 里出现该类别的图数；没出现过的类别不进入总体")
    print("in-train = PathSeg 训练标签里有这个组织概念（多数来自乳腺 BCSS 或不分器官的 SegPath）；absent = 没有")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "name", "group", "prompt", "tiles_or_classes",
                    "mean_or_macro", "pooled_or_micro", "all_pooled", "prompt_in_training_vocab"])
        w.writerows(rows)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="PathSegmentor 组织分割批量评测")
    ap.add_argument("--data", default=DEFAULT_DATA, help="包含 image/ 和 mask/ 的目录")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="PathSegmentor 权重 model_state_dict.pt")
    ap.add_argument("--out", default="out/ukk_col_tissue", help="结果输出目录（不同设置请用不同目录）")
    ap.add_argument("--upscale", type=int, default=2, choices=[1, 2, 3, 4],
                    help="输入放大倍数：2 = 0.5 um/px 的 tile 放大到训练用的 40x；1 = 直接输入")
    ap.add_argument("--template", default="tissue-level {name} in colon pathology",
                    help="用自己类别名时的 prompt 模板（PathSeg 训练格式：<结构>-level <对象> in <部位> pathology）")
    ap.add_argument("--trained-prompts", action="store_true",
                    help="in-training 组改用 PathSeg 训练集里同一概念的原文 prompt（见 CLASSES）")
    ap.add_argument("--all-prompts", action="store_true", help="每张图都发全部 8 个 prompt，额外报告含误检的 Dice")
    ap.add_argument("--ignore-background", action="store_true", help="背景(11)像素也不参与计算")
    ap.add_argument("--thr", type=float, default=0.5, help="概率阈值")
    ap.add_argument("--limit", type=int, default=0, help="随机抽 N 张（固定种子），0 = 全部")
    ap.add_argument("--save-vis", type=int, default=0, help="保存前 N 张的对比图（原图 | GT | 预测）")
    ap.add_argument("--check", action="store_true", help="只检查数据，不加载模型")
    ap.add_argument("--sanity", action="store_true", help="只在仓库自带示例上验证模型，不读评测数据")
    ap.add_argument("--selftest", action="store_true", help="只测 Dice 计算逻辑")
    a = ap.parse_args()

    if not os.path.exists(CONFIG):
        sys.exit("请在 PathSegmentor 仓库根目录运行")
    if a.selftest:
        selftest()
        return
    if a.sanity:
        model = load_model(a.ckpt)
        sanity(Segmenter(model, [p for _, _, p in example_pairs()]), a.thr)
        return
    pairs, only_img, only_mask = pair_files(os.path.join(a.data, "image"), os.path.join(a.data, "mask"))
    if a.check:
        check(pairs, only_img, only_mask)
        return
    if not pairs:
        sys.exit("没有配对成功的图像 / mask，先跑 --check")
    if "{name}" not in a.template:
        sys.exit("--template 里必须包含 {name}")
    if a.limit:
        pairs = sorted(random.Random(0).sample(pairs, min(a.limit, len(pairs))))

    prompts = build_prompts(a.template, a.trained_prompts)
    vocab = load_vocab()
    in_vocab = {c: prompts[c].lower() in vocab for c in NAMES}
    os.makedirs(a.out, exist_ok=True)
    if a.save_vis:
        os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({**vars(a), "prompts": prompts, "prompt_in_training_vocab": in_vocab, "n_pairs": len(pairs),
                   "unpaired_images": len(only_img), "unpaired_masks": len(only_mask)},
                  f, ensure_ascii=False, indent=2)
    print(f"{len(pairs)} 张图；upscale {a.upscale}；prompt（* = PathSeg 训练原文）:")
    for c in NAMES:
        print(f"  {'*' if in_vocab[c] else ' '} {c:<26}{prompts[c]}")

    seg = Segmenter(load_model(a.ckpt), list(prompts.values()))
    from tqdm import tqdm

    stats = {c: new_stat() for c in NAMES}
    n_eval, skipped, n_vis = 0, [], 0
    with open(os.path.join(a.out, "per_tile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "prompt", "present", "gt_px", "pred_px", "tp", "dice"])
        for ip, mp in tqdm(pairs, dynamic_ncols=True):
            try:
                image = np.array(Image.open(ip).convert("RGB"))
                mask = load_mask(mp)
            except (OSError, ValueError) as e:
                skipped.append((ip, f"读取失败: {e}"))
                continue
            if image.shape[:2] != mask.shape:
                skipped.append((ip, f"尺寸不一致 image {image.shape[:2]} / mask {mask.shape}"))
                continue
            valid = valid_region(mask, a.ignore_background)
            present = [c for c in NAMES if np.isin(mask, CLASSES[c][0]).any()]
            todo = NAMES if a.all_prompts else present
            if not todo:
                skipped.append((ip, "没有任何评测类别（只有未标注/背景）"))
                continue
            probs = predict(seg, image, [prompts[c] for c in todo], a.upscale)
            for i, c in enumerate(todo):
                tp, p, g = tile_counts(mask, probs[i] > a.thr, CLASSES[c][0], valid)
                add(stats[c], tp, p, g)
                w.writerow([os.path.basename(ip), c, prompts[c], int(g > 0), g, p, tp,
                            f"{dice(tp, p, g):.4f}" if g > 0 else ""])
            n_eval += 1
            if n_vis < a.save_vis:
                stem = os.path.splitext(os.path.basename(ip))[0]
                save_vis(os.path.join(a.out, "vis", f"{stem}.png"), image, mask, valid, probs, todo, a.thr)
                n_vis += 1

    per_class, groups = summarize(stats, a.all_prompts)
    report(per_class, groups, prompts, in_vocab, a.all_prompts, a.out)
    print(f"\n评测 {n_eval} 张；跳过 {len(skipped)} 张；结果在 {a.out}/")
    for ip, why in skipped[:5]:
        print(f"  跳过 {os.path.basename(ip)}: {why}")


if __name__ == "__main__":
    main()
