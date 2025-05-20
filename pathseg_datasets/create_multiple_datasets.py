import os
import subprocess

# # 定义基础路径
# base_path = '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets'

# # 获取所有子文件夹路径
# subdirectories = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

subdirectories = [
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/BCSS',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/CoCaHis',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/CoNIC2022',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/CoNSeP',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/CRAG',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/CryoNuSeg',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/DigestPath2019_signe_ring_cell',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/DigestPath2019_colonoscopy_tissue_segment',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/GlaS',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/Janowczyk',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/MoNuSAC2020',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/MoNuSeg',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/NuCLS',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/TNBC',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/WSSS4LUAD',
    # '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/MIDOG_breast',
    '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/MIDOG_neuroendocrine',
    '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/RINGS',
    '/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets/SICAPv2',
]

# 遍历每个子文件夹路径并调用另一个脚本
for subdir in subdirectories:
    print(f"Processing dataset in: {subdir}")
    # 调用 process_annotations.py 并传递 targetpath 参数
    subprocess.run(['python', 'create_single_dataset_name_label.py', '--targetpath', subdir], check=True)