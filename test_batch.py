#!/usr/bin/env python
"""PathSegmentor 组织分割批量评测（结肠 5 个队列 + UKK 肺 LUAD/LUSC + UKK 乳腺 BRCA）。

--cohort 选类别表（默认 COAD = 结肠，与以前完全相同）：COAD / LUAD / LUSC / BRCA，标签值与
VPR_LISA_BASED/scripts/test_batch.py 的 COHORTS 相同（结肠背景 11，肺/乳腺背景 12）。肺/乳腺要同时给 --data；
不给 --template 时模板里的部位自动取该测试集的器官（colon / lung / breast）。

在 PathSegmentor 仓库根目录、pathsegmentor 环境里运行：
  python test_batch.py --selftest                          # 只测 Dice 计算逻辑，几秒，不用 GPU
  python test_batch.py --check                             # 只检查数据（配对/尺寸/标签分布），不加载模型
  CUDA_VISIBLE_DEVICES=1 python test_batch.py --sanity     # 用仓库自带的示例图(带 GT)确认模型加载和推理正常
  CUDA_VISIBLE_DEVICES=1 python test_batch.py --limit 50   # 随机抽 50 张试跑
  CUDA_VISIBLE_DEVICES=1 python test_batch.py              # 全量

评测协议与 BiomedParse 仓库里的 test_batch.py 相同，两边结果可以逐行对照：
每张图只对 GT 中出现的类别发 prompt，Dice 只在该类别出现的图上统计，
衡量的是"已知目标存在时分得准不准"，不衡量"能否判断目标不存在"。
加 --all-prompts 则每张图都发全部类别的 prompt，额外报告把误检也算进去的 Dice（all 列）。
每个 prompt 单独推理（同一张图只跑一次图像编码器），prompt 之间互不影响，
所以 --all-prompts 时 mean/pooled 两列与不加时完全相同。

像素规则：未标注(0) 永远不参与计算；背景（结肠 11，肺/乳腺 12）不作为类别评测，但模型在背景上的预测默认算误报；
加 --ignore-background 则背景像素也完全不参与计算。

默认按官方推荐用法（README + 论文的数据处理）：
  - 输入：40x 的 1024x1024 patch。UKK_Col 是 20x（约 0.5 um/px），按论文"非 40x 数据双线性放大到 40x、
    边长超过 1500 按 1024 切块"的规则：1024 的 tile 放大到 2048，切成 4 块 1024 分别推理（--upscale 2 --crop 1024），
    概率图拼回后用区域平均缩回 1024，与你自己模型的结果在同一张 tile、同一分辨率上算 Dice。
  - prompt：论文模板 "<结构>-level <对象> in <部位> pathology"，结构 tissue、部位 colon、对象用 OBJECTS 里的名字
    （去掉与 "tissue-level" 重复的 "tissue"）。官方只规定模板，不规定对象用什么词：
    --objects names（默认）= 你的类名，与你的模型问同一个问题；
    --objects pathseg = 同一模板、部位仍是 colon，对象换成 PathSeg 训练集里同一概念的组织级用词（stroma、
    smooth muscle cells、blood、mucoid material…），作敏感性分析。两组都报告，不按测试结果挑。
    --trained-prompts 照搬训练原文（部位是 breast/unspecified），不是官方用法，只作分析。
  - 阈值 0.5；每个 prompt 单独推理（与论文评测一致：一个样本一条 prompt）。
--gate 决定 prompt 发给哪几块 patch：
  tile（默认）= 类别在整张 tile 的 GT 里出现，就发给全部 4 块。模型拿到的信息与你的模型相同
               （只知道"这张 tile 里有"），不含该类的 patch 也会被迫给出预测。
  patch       = 只发给 GT 里含有该类的 patch（论文以 patch 为样本的评测方式），模型额外获知类别在哪块里。
50 张试跑（非官方设置，仅供参考）：--upscale 1 比官方的放大切块更好
（--trained-prompts、gate tile 的 mean Dice：epithelial 0.67 vs 0.54，mucosa 0.41 vs 0.15，connective 0.14 vs 0.04）。

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
# 标签值 = VPR_LISA_BASED/scripts/test_batch.py 的 COHORTS（结肠 TISSUE_CLASSES / LUNG_CLASSES / BRCA_CLASSES）。
# 结肠：训练集的 colon 组织级只有 gland / malignant lesion，所以多数对应来自别的器官（BCSS 乳腺、SegPath 不分器官），
# "in-training" 指概念出现过，不代表在结肠上训练过。可按需替换成下面注释里的其他训练原文：
#   epithelial tissue: "tissue-level malignant lesion in colon pathology" / "tissue-level gland in colon pathology"
#   connective tissue: "tissue-level tumor associated stroma tissue in lung pathology"
#   erythrocytes:      "cell-level red blood cells in unspecified pathology"
COAD_CLASSES = {
    "epithelial tissue":        ((1,),      "tissue-level epithelial in unspecified pathology"),
    "colorectal mucosa":        ((2,),      None),
    "connective tissue":        ((3, 4, 6), "tissue-level stroma in breast pathology"),
    "muscle tissue":            ((5,),      "tissue-level smooth muscle cells in unspecified pathology"),
    "dense lymphoid aggregate": ((7,),      "tissue-level lymphocytic infiltrate in breast pathology"),
    "necrotic debris":          ((8,),      "tissue-level necrosis or debris in breast pathology"),
    "erythrocytes":             ((9,),      "tissue-level blood in breast pathology"),
    "mucin":                    ((10,),     "tissue-level mucoid material in breast pathology"),
}
# 肺（LUAD 和 LUSC 共用）：肺的组织级训练标签只有 WSSS4LUAD 的 3 类，其余对应来自乳腺 BCSS
LUNG_CLASSES = {
    "epithelial tissue":        ((1,),      "tissue-level tumor epithelial tissue in lung pathology"),
    "connective tissue":        ((2, 6),    "tissue-level tumor associated stroma tissue in lung pathology"),
    "necrotic debris":          ((3,),      "tissue-level necrosis or debris in breast pathology"),
    "mucin":                    ((4,),      "tissue-level mucoid material in breast pathology"),
    "lung tissue":              ((5,),      "tissue-level normal tissue in lung pathology"),
    "erythrocytes":             ((7,),      "tissue-level blood in breast pathology"),
    "bronchial mucosa":         ((8,),      None),
    "cartilage":                ((9,),      None),
    "mucous glands":            ((10,),     None),
    "dense lymphoid aggregate": ((11,),     "tissue-level lymphocytic infiltrate in breast pathology"),
}
# 乳腺：BCSS 是乳腺组织级训练数据，除表皮外都有同器官的对应
#   epithelial tissue 的原位癌部分（标签 3）也可对应 "tissue-level dcis in breast pathology"
BRCA_CLASSES = {
    "epithelial tissue":        ((1, 3),    "tissue-level tumor in breast pathology"),
    "connective tissue":        ((2, 8),    "tissue-level stroma in breast pathology"),
    "necrotic debris":          ((4,),      "tissue-level necrosis or debris in breast pathology"),
    "mucin":                    ((5,),      "tissue-level mucoid material in breast pathology"),
    "dense lymphoid aggregate": ((6,),      "tissue-level lymphocytic infiltrate in breast pathology"),
    "fatty tissue":             ((7,),      "tissue-level fat in breast pathology"),
    "erythrocytes":             ((9,),      "tissue-level blood in breast pathology"),
    "epidermis":                ((10,),     None),
    "breast parenchyma":        ((11,),     "tissue-level normal acinus or duct in breast pathology"),
}

# 官方模板里的 <对象>（"tissue-level" 已表明是组织级，所以去掉类名里重复的 "tissue"；
# lung tissue / fatty tissue 去掉后不成词，保留全名，PathSeg 自己的标签里也有 "normal tissue" 这样的写法）
COAD_OBJECTS = {
    "epithelial tissue":        "epithelial",
    "colorectal mucosa":        "colorectal mucosa",
    "connective tissue":        "connective",
    "muscle tissue":            "muscle",
    "dense lymphoid aggregate": "dense lymphoid aggregate",
    "necrotic debris":          "necrotic debris",
    "erythrocytes":             "erythrocytes",
    "mucin":                    "mucin",
}
SHARED_OBJECTS = {"epithelial tissue": "epithelial", "connective tissue": "connective"}
LUNG_OBJECTS = {c: SHARED_OBJECTS.get(c, c) for c in LUNG_CLASSES}
BRCA_OBJECTS = {c: SHARED_OBJECTS.get(c, c) for c in BRCA_CLASSES}

# --cohort -> (类别表, 背景标签, 模板里的对象名, 器官词)
COHORTS = {"COAD": (COAD_CLASSES, 11, COAD_OBJECTS, "colon"), "LUAD": (LUNG_CLASSES, 12, LUNG_OBJECTS, "lung"),
           "LUSC": (LUNG_CLASSES, 12, LUNG_OBJECTS, "lung"), "BRCA": (BRCA_CLASSES, 12, BRCA_OBJECTS, "breast")}
UNANNOTATED = 0



def pathseg_object(c):
    """--objects pathseg：训练原文里的对象词，如 "tissue-level stroma in breast pathology" -> "stroma"；
    训练集里没有的概念（colorectal mucosa）仍用 OBJECTS。"""
    tp = CLASSES[c][1]
    return tp.split("-level ", 1)[1].rsplit(" in ", 1)[0] if tp else OBJECTS[c]


def set_cohort(name):
    """选测试集的类别表；其余函数都读这些模块级变量。"""
    global CLASSES, BACK, OBJECTS, NAMES, IN_TRAIN, ABSENT, GROUPS
    CLASSES, BACK, OBJECTS, _ = COHORTS[name]
    NAMES = list(CLASSES)
    IN_TRAIN = [c for c in NAMES if CLASSES[c][1] is not None]
    ABSENT = [c for c in NAMES if CLASSES[c][1] is None]
    GROUPS = {"in-training": IN_TRAIN, "absent-from-training": ABSENT, "ALL": NAMES}


def default_template(cohort):
    return f"tissue-level {{name}} in {COHORTS[cohort][3]} pathology"


def organ_mismatch(template, cohort):
    """模板里出现了别的测试集的器官词（如 --cohort LUAD 却写 in colon pathology）就返回那个词。"""
    own = COHORTS[cohort][3]
    others = {o for _, _, _, o in COHORTS.values()} - {own}
    return next((o for o in sorted(others) if f"in {o} pathology" in template), None)


set_cohort("COAD")
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLORS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                   [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
                   [210, 245, 60], [170, 110, 40]], np.uint8)    # 至少与最大的类别表（肺 10 类）一样多
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


def is_present(g, tile_px, min_frac):
    """该 (tile, 类别) 是否提问、计分：GT 面积 >= min_frac x 整张 tile 像素（与自己模型的
    VPR_LISA_BASED/scripts/test_batch.py 的 present_classes 相同，默认 0.02）；min_frac=0 即 GT 里有就算。"""
    return g > 0 and g >= min_frac * tile_px


def add(stat, tp, p, g, present):
    stat["tp_all"] += tp
    stat["p_all"] += p
    stat["g_all"] += g
    if present:                     # 只有计分的 (tile, 类别) 才进入 mean / pooled
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

        nq = sum(stats[c]["n"] for c in cs)

        def macro(k):
            return float(np.mean([per_class[c][k] for c in cs])) if cs else NAN

        groups[g] = {   # 与自己模型脚本的 macro_mean_dice / macro_pooled_dice / all_questions_mean_dice 同义
            "classes": len(cs),
            "questions": nq,
            "macro_mean": macro("mean"),
            "macro_pooled": macro("pooled"),
            "macro_all": macro("all") if all_prompts else NAN,
            "all_questions": sum(stats[c]["dsum"] for c in cs) / nq if nq else NAN,
        }
    return per_class, groups


def crop_prompt_ids(mask, labels_list, box):
    """--gate patch 用：box = 原图坐标 (y0, y1, x0, x1)，返回在这块区域的 GT 中出现过的 prompt 下标。"""
    y0, y1, x0, x1 = box
    sub = mask[y0:y1, x0:x1]
    return [i for i, labels in enumerate(labels_list) if np.isin(sub, labels).any()]


def example_prompt(mask_file):
    """示例 mask 文件名 -> 训练格式的 prompt，如 ..._bile+duct_nuclei+connective.png ->
    'nuclei-level connective in bile duct pathology'（与 README 的说明一致）。"""
    parts = os.path.splitext(os.path.basename(mask_file))[0].split("_")
    site = parts[-2].replace("+", " ")
    words = parts[-1].split("+")
    return f"{words[0]}-level {' '.join(words[1:])} in {site} pathology"


def selftest():
    set_cohort("COAD")
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
    add(st["epithelial tissue"], 4, 5, 5, True)     # 单张 Dice 0.8
    add(st["epithelial tissue"], 4, 5, 5, True)     # 0.8
    add(st["epithelial tissue"], 0, 3, 0, False)    # 不计分（GT 没有）：不进 mean/pooled，只进 all（误检）
    add(st["colorectal mucosa"], 1, 5, 5, True)     # 0.2
    pc, gr = summarize(st, all_prompts=True)

    def close(a, b):
        return abs(a - b) < 1e-9

    e = pc["epithelial tissue"]
    assert e["tiles"] == 2 and close(e["mean"], 0.8) and close(e["pooled"], 0.8)
    assert close(e["all"], 16 / 23)                          # 2*8 / (13+10)
    assert close(gr["ALL"]["macro_mean"], 0.5)               # (0.8 + 0.2) / 2
    assert close(gr["ALL"]["macro_pooled"], 0.5)             # 两类 pooled 分别 0.8、0.2
    assert close(gr["ALL"]["all_questions"], 1.8 / 3)        # 3 个问题 (0.8 + 0.8 + 0.2) / 3
    assert gr["ALL"]["questions"] == 3
    assert gr["in-training"]["classes"] == 1 and gr["absent-from-training"]["classes"] == 1
    assert close(gr["in-training"]["macro_mean"], 0.8) and close(gr["absent-from-training"]["macro_mean"], 0.2)
    # min_frac 与自己模型的脚本一致：gt.sum() >= min_frac * mask.size
    assert is_present(20972, 1024 * 1024, 0.02) and not is_present(20971, 1024 * 1024, 0.02)
    assert is_present(1, 100, 0.0) and not is_present(0, 100, 0.0)
    assert pc["muscle tissue"]["tiles"] == 0 and pc["muscle tissue"]["mean"] != pc["muscle tissue"]["mean"]

    # --gate patch：只给 GT 里含有该类的区域发 prompt
    labels_list = [(1,), (5,), (9,)]
    assert crop_prompt_ids(m, labels_list, (0, 1, 0, 4)) == [0]        # 第一行只有标签 1/2/0
    assert crop_prompt_ids(m, labels_list, (1, 3, 0, 2)) == [1]        # 3,4 / 5,5 → 只有 muscle
    assert crop_prompt_ids(m, labels_list, (0, 3, 0, 4)) == [0, 1]     # 整张：没有 9
    assert set(OBJECTS) == set(NAMES)
    assert build_prompts("tissue-level {name} in colon pathology", False)["epithelial tissue"] == \
        "tissue-level epithelial in colon pathology"
    ps = build_prompts("tissue-level {name} in colon pathology", False, "pathseg")
    assert ps["connective tissue"] == "tissue-level stroma in colon pathology"
    assert ps["muscle tissue"] == "tissue-level smooth muscle cells in colon pathology"
    assert ps["necrotic debris"] == "tissue-level necrosis or debris in colon pathology"
    assert ps["mucin"] == "tissue-level mucoid material in colon pathology"
    assert ps["colorectal mucosa"] == "tissue-level colorectal mucosa in colon pathology"

    assert example_prompt("Part_2_1211_pathology_bile+duct_nuclei+connective.png") == \
        "nuclei-level connective in bile duct pathology"
    assert example_prompt("x/67test188_pathology_lung_tissue+tumor+associated+stroma+tissue.png") == \
        "tissue-level tumor associated stroma tissue in lung pathology"
    vocab = load_vocab()
    # 各测试集的类别表：训练原文都在 PathSeg 词表里、对象名齐全、标签不重复且不含未标注/背景、颜色够用
    for name, (cls, back, objs, _) in COHORTS.items():
        set_cohort(name)
        for c in IN_TRAIN:
            assert CLASSES[c][1] in vocab, f"{name} 不是训练原文: {CLASSES[c][1]}"
        assert set(OBJECTS) == set(NAMES), name
        labels = [v for ls, _ in cls.values() for v in ls]
        assert len(labels) == len(set(labels)) and UNANNOTATED not in labels and back not in labels, name
        assert len(cls) <= len(COLORS), name
    set_cohort("LUAD")
    assert BACK == 12 and ABSENT == ["bronchial mucosa", "cartilage", "mucous glands"]
    lp = build_prompts(default_template("LUAD"), False)
    assert lp["epithelial tissue"] == "tissue-level epithelial in lung pathology"
    assert lp["lung tissue"] == "tissue-level lung tissue in lung pathology"
    lps = build_prompts(default_template("LUAD"), False, "pathseg")
    assert lps["epithelial tissue"] == "tissue-level tumor epithelial tissue in lung pathology"
    assert lps["cartilage"] == "tissue-level cartilage in lung pathology"
    lm = np.array([[1, 2, 6, 12], [11, 0, 5, 12]])
    assert tile_counts(lm, np.ones(lm.shape, bool), (2, 6), valid_region(lm, False)) == (2, 7, 2)
    assert tile_counts(lm, np.ones(lm.shape, bool), (2, 6), valid_region(lm, True)) == (2, 5, 2)   # 背景 12 不计
    set_cohort("BRCA")
    assert ABSENT == ["epidermis"] and CLASSES["epithelial tissue"][0] == (1, 3)
    assert build_prompts(default_template("BRCA"), False, "pathseg")["fatty tissue"] == "tissue-level fat in breast pathology"
    assert organ_mismatch("tissue-level {name} in colon pathology", "BRCA") == "colon"
    assert organ_mismatch(default_template("BRCA"), "BRCA") is None
    set_cohort("COAD")
    assert default_template("COAD") == "tissue-level {name} in colon pathology"   # 与以前的默认相同

    # --stain-ref：LAB 往返、白/黑的 LAB 值、匹配到自己的统计量 = 原图、匹配后组织统计量 = 参考
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert np.abs(lab2rgb(rgb2lab(img)).astype(int) - img).max() <= 1
    w, k = rgb2lab(np.array([[[255, 255, 255], [0, 0, 0]]], np.uint8))[0]
    assert abs(w[0] - 100) < 0.1 and abs(w[1]) < 0.1 and abs(w[2]) < 0.1 and abs(k[0]) < 0.1
    tis = rng.integers(90, 200, (64, 64, 3), dtype=np.uint8)
    tis[..., 1] //= 2                                        # 偏紫的"组织"，全部是组织像素
    assert np.abs(reinhard(tis, lab_stats([tis])).astype(int) - tis).max() <= 2
    ref_img = (tis.astype(np.float32) * 0.85).astype(np.uint8)
    out = reinhard(tis, lab_stats([ref_img]))
    assert np.allclose(rgb2lab(out).mean((0, 1)), lab_stats([ref_img])[0], atol=1.0)
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


def predict(seg, image, prompts, upscale, crop=CROP, gate=None):
    """返回 (len(prompts), H, W) 概率图。upscale>1 时先放大、按 crop 切块推理、拼回后用区域平均缩回原尺寸。
    crop=0：不切块，（放大后的）整张图一次推理。
    gate=(mask, 每个 prompt 的标签)：只把 prompt 发给 GT 里含有该类的块，其余块该 prompt 的概率为 0（--gate patch）。"""
    torch = seg.torch
    import torch.nn.functional as F
    H, W = image.shape[:2]
    big = image if upscale == 1 else np.array(
        Image.fromarray(image).resize((W * upscale, H * upscale), Image.BILINEAR))
    BH, BW = big.shape[:2]
    step = crop if crop > 0 else max(BH, BW)
    acc = torch.zeros((len(prompts), BH, BW), device=seg.m.device)
    for y in range(0, BH, step):
        for x in range(0, BW, step):
            ids = list(range(len(prompts)))
            if gate is not None:
                box = (y // upscale, (y + step) // upscale, x // upscale, (x + step) // upscale)
                ids = crop_prompt_ids(gate[0], gate[1], box)
            if ids:
                acc[ids, y:y + step, x:x + step] = seg.crop_probs(big[y:y + step, x:x + step],
                                                                  [prompts[i] for i in ids])
    if upscale > 1:
        acc = F.interpolate(acc[None], size=(H, W), mode="area")[0]
    return acc.cpu().numpy()


# ---------------- 颜色归一化（--stain-ref，诊断用，非官方流程） ----------------
# Reinhard：把 tile 组织像素的 LAB 均值/标准差匹配到参考图（如 examples/ 里同器官的训练样本）。sRGB(D65) <-> CIELAB。
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)
_D = 6 / 29


def rgb2lab(rgb):
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (c @ _M.T) / _WHITE
    f = np.where(t > _D ** 3, np.cbrt(t), t / (3 * _D ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    t = np.where(f > _D, f ** 3, 3 * _D ** 2 * (f - 4 / 29)) * _WHITE
    c = np.clip(t @ np.linalg.inv(_M).T, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def tissue_pixels(rgb):
    """组织像素：饱和度 (max-min)/max > 20/255（去掉白色背景）；组织太少就用全部像素。"""
    mx, mn = rgb.max(-1).astype(np.float32), rgb.min(-1).astype(np.float32)
    t = (mx - mn) / np.maximum(mx, 1) > 20 / 255
    return t if t.mean() >= 0.01 else np.ones_like(t)


def lab_stats(images):
    """多张参考图的组织像素合并后的 LAB 均值、标准差。"""
    px = np.concatenate([rgb2lab(im)[tissue_pixels(im)] for im in images])
    return px.mean(0), px.std(0)


def reinhard(rgb, ref):
    lab = rgb2lab(rgb)
    src = lab[tissue_pixels(rgb)]
    mu, sd = src.mean(0), np.maximum(src.std(0), 1e-6)
    return lab2rgb((lab - mu) / sd * ref[1] + ref[0])


def build_prompts(template, trained, objects="names"):
    out = {}
    for c in NAMES:
        tp = CLASSES[c][1]
        obj = pathseg_object(c) if objects == "pathseg" else OBJECTS[c]
        out[c] = tp if (trained and tp) else template.format(name=obj)
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
        if image.shape[:2] != gt.shape:   # 如 1224_pathology_colon.png 原图 512、mask 1024；官方 demo 同样会把输入 bicubic 缩放到 1024
            image = np.array(Image.fromarray(image).resize(gt.shape[::-1], Image.BICUBIC))
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


def report(per_class, groups, prompts, in_vocab, all_prompts, min_frac, out_dir, csv_name="summary.csv"):
    pw = max(len(p) for p in prompts.values()) + 2
    cols = ["mean", "pooled"] + (["all"] if all_prompts else []) + ["allq"]
    head = f"{'class':<26}{'group':<10}{'prompt':<{pw}}{'n':>7}" + "".join(f"{c:>8}" for c in cols)
    print("\n" + head + "\n" + "-" * len(head))
    rows = []
    for c in NAMES:
        r = per_class[c]
        grp = "in-train" if c in IN_TRAIN else "absent"
        vals = [r["mean"], r["pooled"]] + ([r["all"]] if all_prompts else []) + [NAN]
        print(f"{c:<26}{grp:<10}{prompts[c]:<{pw}}{r['tiles']:>7}" + "".join(fmt(v) for v in vals))
        rows.append(["class", c, grp, prompts[c], r["tiles"], num(r["mean"]), num(r["pooled"]), num(r["all"]), "",
                     int(in_vocab.get(c, False))])
    print("-" * len(head))
    for g, r in groups.items():
        vals = [r["macro_mean"], r["macro_pooled"]] + ([r["macro_all"]] if all_prompts else []) + [r["all_questions"]]
        label = f"({r['classes']} classes)"
        print(f"{g:<26}{'':<10}{label:<{pw}}{r['questions']:>7}" + "".join(fmt(v) for v in vals))
        rows.append(["group", g, "", "", r["questions"], num(r["macro_mean"]), num(r["macro_pooled"]),
                     num(r["macro_all"]), num(r["all_questions"]), ""])
    print(f"\nn      = 计分的 (tile, 类别) 数：GT 面积 >= {min_frac:g} × 整张 tile 才提问、才计分；组那几行是问题总数")
    print("mean   = 每个问题 Dice 的平均；组那几行 = 各类 mean 的平均（macro，= 自己模型脚本的 macro_mean_dice）")
    print("pooled = 所有问题的像素合并后算一个 Dice；组那几行 = 各类 pooled 的平均（= macro_pooled_dice）")
    if all_prompts:
        print("all    = 同 pooled，但每张图都问全部类别，GT 里没有的图上的误检也算进去；组那几行 = 各类 all 的平均")
    print("allq   = 只有组那几行：组内所有问题 Dice 的平均，问题多的类别权重大（= all_questions_mean_dice）")
    print("in-train = PathSeg 训练标签里有这个组织概念（多数来自乳腺 BCSS 或不分器官的 SegPath）；absent = 没有")
    with open(os.path.join(out_dir, csv_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "name", "group", "prompt", "n", "mean_or_macro_mean", "pooled_or_macro_pooled",
                    "all_or_macro_all", "all_questions_mean", "prompt_in_training_vocab"])
        w.writerows(rows)
    print(f"汇总写入 {os.path.join(out_dir, csv_name)}")


def resummarize(out_dir, min_frac):
    """不跑模型：按 min_frac 重新汇总已有的 per_tile.csv。旧版输出没有 tile_px 列时按 1024x1024 算面积占比。"""
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    set_cohort(cfg.get("cohort", "COAD"))       # 旧版输出没有 cohort 字段 = 结肠
    all_prompts = cfg.get("all_prompts", False)
    stats = {c: new_stat() for c in NAMES}
    n_rows = no_px = 0
    with open(os.path.join(out_dir, "per_tile.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("tile_px"):
                no_px += 1
            g = int(r["gt_px"])
            add(stats[r["class"]], int(r["tp"]), int(r["pred_px"]), g,
                is_present(g, int(r.get("tile_px") or 1024 * 1024), min_frac))
            n_rows += 1
    print(f"重新汇总 {out_dir}/per_tile.csv（{n_rows} 行），min_frac {min_frac:g}")
    if no_px:
        print(f"  其中 {no_px} 行没有 tile_px 列（旧版输出），按 1024x1024 计算面积占比")
    per_class, groups = summarize(stats, all_prompts)
    report(per_class, groups, cfg["prompts"], cfg.get("prompt_in_training_vocab", {}), all_prompts, min_frac,
           out_dir, f"summary_minfrac{min_frac:g}.csv")


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="PathSegmentor 组织分割批量评测")
    ap.add_argument("--cohort", choices=sorted(COHORTS), default="COAD",
                    help="类别表：COAD = 结肠（默认）；LUAD / LUSC = 肺；BRCA = 乳腺")
    ap.add_argument("--data", default=DEFAULT_DATA, help="包含 image/ 和 mask/ 的目录")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="PathSegmentor 权重 model_state_dict.pt")
    ap.add_argument("--out", default="out/ukk_col_tissue", help="结果输出目录（不同设置请用不同目录）")
    ap.add_argument("--upscale", type=int, default=2, choices=[1, 2, 3, 4],
                    help="输入放大倍数：2 = 20x 的 tile 放大到官方要求的 40x（默认）；1 = 直接输入（非官方）")
    ap.add_argument("--crop", type=int, default=CROP,
                    help="推理切块边长（放大后的像素），默认 1024 = 官方 patch 大小；0 = 不切块（非官方）")
    ap.add_argument("--gate", choices=["tile", "patch"], default="tile",
                    help="tile = 类别在 tile 里出现就发给全部 patch（与你的模型信息相同，默认）；"
                         "patch = 只发给 GT 含该类的 patch（论文按 patch 评测的方式）")
    ap.add_argument("--template", default=None,
                    help="官方 prompt 模板 <结构>-level <对象> in <部位> pathology；{name} 换成 OBJECTS 里的对象名。"
                         "默认 tissue-level {name} in <--cohort 的器官> pathology")
    ap.add_argument("--objects", choices=["names", "pathseg"], default="names",
                    help="模板里的 <对象>：names = 你的类名（默认，与你的模型同一个问题）；"
                         "pathseg = PathSeg 训练集里同一概念的组织级用词，部位仍是该测试集的器官（敏感性分析）")
    ap.add_argument("--trained-prompts", action="store_true",
                    help="in-training 组改用 PathSeg 训练原文（非官方用法，只作分析；见 CLASSES）")
    ap.add_argument("--all-prompts", action="store_true", help="每张图都发全部类别的 prompt，额外报告含误检的 Dice")
    ap.add_argument("--ignore-background", action="store_true", help="背景（结肠 11，肺/乳腺 12）像素也不参与计算")
    ap.add_argument("--stain-ref", nargs="+", default=[], metavar="PNG",
                    help="诊断用（非官方流程）：推理前把每张 tile 的颜色按 Reinhard（LAB 均值/标准差）匹配到这些参考图，"
                         "例如仓库 examples/ 里同器官的训练样本；GT 和计分不变")
    ap.add_argument("--min-frac", type=float, default=0.02,
                    help="类别 GT 面积 >= 该比例 × 整张 tile 才提问、才计分；0.02 = 自己模型 scripts/test_batch.py 的默认；"
                         "0 = GT 里有就算")
    ap.add_argument("--resummarize", metavar="OUT_DIR",
                    help="不跑模型：按 --min-frac 重新汇总 OUT_DIR/per_tile.csv，写入 summary_minfrac<值>.csv")
    ap.add_argument("--tile-size", type=int, default=1024,
                    help="只评测这个尺寸的 tile，其他尺寸跳过（与自己模型脚本的 --tile_size 相同）；0 = 不限")
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
    if a.resummarize:
        resummarize(a.resummarize, a.min_frac)
        return
    if a.sanity:
        model = load_model(a.ckpt)
        sanity(Segmenter(model, [p for _, _, p in example_pairs()]), a.thr)
        return
    set_cohort(a.cohort)
    if a.cohort != "COAD" and a.data == DEFAULT_DATA:
        sys.exit(f"--cohort {a.cohort} 需要同时给 --data（默认数据是结肠 UKK_Col）")
    if a.template is None:
        a.template = default_template(a.cohort)
    wrong = organ_mismatch(a.template, a.cohort)
    if wrong:
        sys.exit(f"--template 写的是 {wrong}，但 --cohort 是 {a.cohort}（器官应为 {COHORTS[a.cohort][3]}）")
    img_dir, msk_dir = os.path.join(a.data, "image"), os.path.join(a.data, "mask")
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        sys.exit(f"{a.data} 下没有 image/ 和 mask/ 子目录。QuPath 导出的是 <切片>/ALL/*.jpg|png，需要先像 UKK_Col 那样"
                 "汇总（VPR_LISA_BASED/scripts/ukk_col/move_qupath2train_val.py）")
    pairs, only_img, only_mask = pair_files(img_dir, msk_dir)
    if a.check:
        check(pairs, only_img, only_mask)
        return
    if not pairs:
        sys.exit("没有配对成功的图像 / mask，先跑 --check")
    if "{name}" not in a.template:
        sys.exit("--template 里必须包含 {name}")
    if a.gate == "patch" and a.all_prompts:
        sys.exit("--all-prompts 会把所有 prompt 发给所有 patch，不能和 --gate patch 一起用")
    if a.limit:
        pairs = sorted(random.Random(0).sample(pairs, min(a.limit, len(pairs))))

    prompts = build_prompts(a.template, a.trained_prompts, a.objects)
    vocab = load_vocab()
    in_vocab = {c: prompts[c].lower() in vocab for c in NAMES}
    ref = None
    if a.stain_ref:
        missing = [p for p in a.stain_ref if not os.path.isfile(p)]
        if missing:
            sys.exit(f"--stain-ref 找不到：{missing}")
        ref = lab_stats([np.array(Image.open(p).convert("RGB")) for p in a.stain_ref])
        print(f"颜色归一化（诊断用，非官方）：参考 {len(a.stain_ref)} 张，组织 LAB 均值 {np.round(ref[0], 1).tolist()} "
              f"标准差 {np.round(ref[1], 1).tolist()}")
    os.makedirs(a.out, exist_ok=True)
    if a.save_vis:
        os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({**vars(a), "prompts": prompts, "prompt_in_training_vocab": in_vocab, "n_pairs": len(pairs),
                   "unpaired_images": len(only_img), "unpaired_masks": len(only_mask),
                   "stain_ref_lab": None if ref is None else [ref[0].tolist(), ref[1].tolist()]},
                  f, ensure_ascii=False, indent=2)
    print(f"{len(pairs)} 张图；upscale {a.upscale}；crop {a.crop}；gate {a.gate}；objects {a.objects}；"
          "prompt（* = PathSeg 训练原文）:")
    for c in NAMES:
        print(f"  {'*' if in_vocab[c] else ' '} {c:<26}{prompts[c]}")

    seg = Segmenter(load_model(a.ckpt), list(prompts.values()))
    from tqdm import tqdm

    stats = {c: new_stat() for c in NAMES}
    n_eval, skipped, n_vis = 0, [], 0
    with open(os.path.join(a.out, "per_tile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "prompt", "present", "gt_px", "pred_px", "tp", "dice", "tile_px"])
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
            if a.tile_size and image.shape[:2] != (a.tile_size, a.tile_size):
                skipped.append((ip, f"不是 {a.tile_size}x{a.tile_size}"))
                continue
            if ref is not None:                  # 只改模型的输入（对比图也显示归一化后的图），GT 不变
                image = reinhard(image, ref)
            valid = valid_region(mask, a.ignore_background)
            present = [c for c in NAMES
                       if is_present(int(np.isin(mask, CLASSES[c][0]).sum()), mask.size, a.min_frac)]
            todo = NAMES if a.all_prompts else present
            if not todo:
                skipped.append((ip, f"没有 GT 面积 >= {a.min_frac:g} 的评测类别"))
                continue
            gate = (mask, [CLASSES[c][0] for c in todo]) if a.gate == "patch" else None
            probs = predict(seg, image, [prompts[c] for c in todo], a.upscale, a.crop, gate)
            for i, c in enumerate(todo):
                tp, p, g = tile_counts(mask, probs[i] > a.thr, CLASSES[c][0], valid)
                counted = c in present
                add(stats[c], tp, p, g, counted)
                w.writerow([os.path.basename(ip), c, prompts[c], int(counted), g, p, tp,
                            f"{dice(tp, p, g):.4f}" if counted else "", mask.size])
            n_eval += 1
            if n_vis < a.save_vis:
                stem = os.path.splitext(os.path.basename(ip))[0]
                save_vis(os.path.join(a.out, "vis", f"{stem}.png"), image, mask, valid, probs, todo, a.thr)
                n_vis += 1

    per_class, groups = summarize(stats, a.all_prompts)
    report(per_class, groups, prompts, in_vocab, a.all_prompts, a.min_frac, a.out)
    print(f"\n评测 {n_eval} 张；跳过 {len(skipped)} 张；结果在 {a.out}/")
    for why, k in Counter(why for _, why in skipped).most_common(5):
        print(f"  跳过 {k} 张：{why}")
    for ip, why in skipped[:3]:
        print(f"  例: {os.path.basename(ip)}: {why}")


if __name__ == "__main__":
    main()
