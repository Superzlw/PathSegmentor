# **PathSeg Dataset**

The PathSeg dataset was constructed by preprocessing publicly available pathology image segmentation datasets. For more information regarding the source datasets or their specific licenses, please reach out to the respective dataset maintainers.

---

## List of Datasets and Links

| Dataset        | Official Link                                                                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BCSS           | [https://github.com/PathologyDataScience/BCSS](https://github.com/PathologyDataScience/BCSS)                                                                                   |
| CoCaHis        | [https://cocahis.irb.hr/](https://cocahis.irb.hr/)                                                                                                                             |
| CoNIC2022      | [https://conic-challenge.grand-challenge.org/](https://conic-challenge.grand-challenge.org/)                                                                                   |
| CRAG           | [https://opendatalab.com/OpenDataLab/CRAG/](https://opendatalab.com/OpenDataLab/CRAG/)                                                                                         |
| DigestPath2019 | [https://digestpath2019.grand-challenge.org/](https://digestpath2019.grand-challenge.org/)                                                                                     |
| GlaS           | [https://datasetninja.com/gland-segmentation](https://datasetninja.com/gland-segmentation)                                                                                     |
| Janowczyk      | [https://andrewjanowczyk.com/use-case-1-nuclei-segmentation/](https://andrewjanowczyk.com/use-case-1-nuclei-segmentation/)                                                     |
| MIDOG          | [https://midog.deepmicroscopy.org/download-dataset/](https://midog.deepmicroscopy.org/download-dataset/)                                                                       |
| MoNuSAC        | [https://monusac-2020.grand-challenge.org/Data/](https://monusac-2020.grand-challenge.org/Data/)                                                                               |
| NuCLS          | [https://sites.google.com/view/nucls/home](https://sites.google.com/view/nucls/home)                                                                                           |
| PanNuke        | [https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification) |
| RINGS          | [https://data.mendeley.com/datasets/h8bdwrtnr5/1](https://data.mendeley.com/datasets/h8bdwrtnr5/1)                                                                             |
| SegPath        | [https://dakomura.github.io/SegPath/](https://dakomura.github.io/SegPath/)                                                                                                     |
| SICAPv2        | [https://www.kaggle.com/datasets/shridharspol/sicapv2](https://www.kaggle.com/datasets/shridharspol/sicapv2)                                                                   |
| TNBC           | [https://peterjacknaylor.github.io/data/](https://peterjacknaylor.github.io/data/)                                                                                             |
| WSSS4LUAD      | [https://wsss4luad.grand-challenge.org/](https://wsss4luad.grand-challenge.org/)                                                                                               |
| CoNSeP         | [https://opendatalab.com/OpenDataLab/CoNSeP](https://opendatalab.com/OpenDataLab/CoNSeP)                                                                                       |
| CPM15          | [https://drive.google.com/drive/folders/11ko-GcDsPpA9GBHuCtl_jNzWQl6qY_-I](https://drive.google.com/drive/folders/11ko-GcDsPpA9GBHuCtl_jNzWQl6qY_-I)                                                                                |
| CPM17          | [https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA](https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA)                                                                                |
| Kumar          | [https://drive.google.com/drive/folders/1bI3RyshWej9c4YoRW-_q7lh7FOFDFUrJ](https://drive.google.com/drive/folders/1bI3RyshWej9c4YoRW-_q7lh7FOFDFUrJ)                                                                                |
| Lizard         | [https://www.kaggle.com/datasets/aadimator/lizard-dataset](https://www.kaggle.com/datasets/aadimator/lizard-dataset)                                                           |

---

## Dataset Download Instructions

To facilitate the acquisition of the various source datasets used in PathSeg, we provide detailed instructions for obtaining each raw dataset below.

---

### 1. BCSS
Choose one of the two independent methods to download the dataset:
* **Option 1: Direct Download (Google Drive)**
    1.  Access the [Google Drive Folder](https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss).
* **Option 2: Command Line (GitHub)**
    1.  Follow the instructions in the "Section Download (command line - flexible)" section of the [BCSS GitHub repository](https://github.com/PathologyDataScience/BCSS).
* **Storage:** Extract/Save to `<raw_data_directory>/BCSS/`.

---

### 2. CoCaHis
1.  Visit the [CoCaHis website](https://cocahis.irb.hr/).
2.  Download the [HDF5 file](http://cocahis.irb.hr/wp-content/uploads/2020/12/CoCaHis.hdf5).
3.  (Optional) Refer to the [file structure documentation](http://cocahis.irb.hr/wp-content/uploads/2020/12/cocahis_structure.txt) for details.
* **Storage:** Save to `<raw_data_directory>/CoCaHis/`.

---

### 3. CoNIC2022
1.  Open the [CoNIC2022 Challenge page](https://conic-challenge.grand-challenge.org/).
2.  Register or log in to a Grand Challenge account.
3.  Navigate to the [Data section](https://conic-challenge.grand-challenge.org/Data/).
4.  Accept the terms and use the provided links to download the training and validation sets.
* **Storage:** Save to `<raw_data_directory>/CoNIC2022/`.

---

### 4. CRAG
Choose one of the two independent methods:
* **Option 1: OpenDataLab**
    1.  Visit [CRAG on OpenDataLab](https://opendatalab.com/OpenDataLab/CRAG/).
    2.  Download via the "Dataset File" tab or use the CLI/SDK tool.
* **Option 2: GitHub Shared Link**
    1.  Access the direct link provided by the author in this [GitHub Issue](https://github.com/XiaoyuZHK/CRAG-Dataset_Aug_ToCOCO/issues/1).
* **Storage:** Save to `<raw_data_directory>/CRAG/`.

---

### 5. DigestPath2019
Choose one of the two independent methods:
* **Option 1: Official Challenge Site**
    1.  Visit [DigestPath2019 Challenge](https://digestpath2019.grand-challenge.org/).
    2.  Log in and request access. Once approved, download the files from the **Dataset** section.
* **Option 2: Direct Google Drive**
    1.  Sign the **DATABASE USE AGREEMENT** as required by the author.
    2.  Access the [Google Drive Folder](https://drive.google.com/drive/folders/1hLd_OD4eGeyUrmb7UCWxxFfGSwwMI15V).
* **Storage:** Save to `<raw_data_directory>/DigestPath2019/`.

---

### 6. GlaS
1.  Visit the [GlaS Dataset Ninja page](https://datasetninja.com/gland-segmentation).
2.  Use the [direct download link](https://datasetninja.com/gland-segmentation#download).
* **Storage:** Unzip to `<raw_data_directory>/GlaS/`.

---

### 7. Janowczyk
1.  Review the tutorials on [Andrew Janowczyk's website](https://andrewjanowczyk.com/use-case-1-nuclei-segmentation/).
2.  **Recommended:** Download the compressed [Nuclei Data](https://andrewjanowczyk.com/wp-static/nuclei.tgz) (approx. 1.5GB).
* **Storage:** Unzip to `<raw_data_directory>/Janowczyk/`.

---

### 8. MIDOG
1.  Check the [MIDOG website](https://midog.deepmicroscopy.org/download-dataset/) for dataset details.
2.  Download directly from the [Google Drive link](https://drive.google.com/drive/folders/1P73g1xg8jw_JGLJaDFQDnxwQA7ROVykA?usp=sharing).
* **Storage:** Save to `<raw_data_directory>/MIDOG/`.

---

### 9. MoNuSAC
1.  Visit the [MoNuSAC Data page](https://monusac-2020.grand-challenge.org/Data/).
2.  Agree to the terms and download via the [Google Drive link](https://drive.google.com/file/d/1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq/view).
* **Storage:** Save to `<raw_data_directory>/MoNuSAC/`.

---

### 10. NuCLS
1.  Visit the [NuCLS project site](https://sites.google.com/view/nucls/home).
2.  Download directly from the [NuCLS Google Drive](https://drive.google.com/drive/folders/1gvjlC9lPYUyf03X2ddWKscqUcEMEsIFX?usp=sharing).
* **Storage:** Save to `<raw_data_directory>/NuCLS/`.

---

### 11. PanNuke
1.  Navigate to the [Kaggle - PanNuke page](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification).
2.  Download the dataset files.
* **Storage:** Unzip to `<raw_data_directory>/PanNuke/`.

---

### 12. RINGS
1.  Visit [RINGS on Mendeley Data](https://data.mendeley.com/datasets/h8bdwrtnr5/1).
2.  Download both the [Test Set](https://data.mendeley.com/public-files/datasets/h8bdwrtnr5/files/5846b131-09fa-44c5-afa8-8fbc52adbd88/file_downloaded) and the [Train Set](https://data.mendeley.com/public-files/datasets/h8bdwrtnr5/files/8416eb6b-d1c8-4a8c-96fd-2f36a2768e46/file_downloaded).
* **Storage:** Unzip both to `<raw_data_directory>/RINGS/`.

---

### 13. SegPath
1.  Visit the [SegPath website](https://dakomura.github.io/SegPath/).
2.  Download the required components from Zenodo:
    * [Epithelial (anti-panCK)](https://zenodo.org/record/7412731)
    * [Smooth muscle/αSMA](https://zenodo.org/record/7412732)
    * [Red blood cells (CD235a)](https://zenodo.org/record/7412580)
    * [Leukocytes (CD45RB)](https://zenodo.org/record/7412739)
    * [Lymphocytes (CD3/CD20)](https://zenodo.org/record/7412529)
    * [Endothelial (ERG)](https://zenodo.org/record/7412512)
    * [Plasma (MIST1)](https://zenodo.org/record/7412500)
    * [Myeloid (MNDA)](https://zenodo.org/record/7412690)
* **Storage:** Extract all files into `<raw_data_directory>/SegPath/`, preserving the internal folder structure.

---

### 14. SICAPv2
1.  Go to the [Kaggle - SICAPv2 page](https://www.kaggle.com/datasets/shridharspol/sicapv2).
2.  Download the dataset files.
* **Storage:** Unzip to `<raw_data_directory>/SICAPv2/`.

---

### 15. TNBC
Choose one of the two independent methods:
* **Option 1: Zenodo Website**
    1.  Visit [TNBC on Zenodo](https://zenodo.org/records/2579118) and download `TNBC_NucleiSegmentation.zip`.
* **Option 2: Direct Download**
    1.  Use the [Direct Download Link](https://zenodo.org/records/2579118/files/TNBC_NucleiSegmentation.zip?download=1).
* **Storage:** Extract to `<raw_data_directory>/TNBC/`.

---

### 16. WSSS4LUAD
1.  Visit the [WSSS4LUAD Challenge page](https://wsss4luad.grand-challenge.org/).
2.  Register/Log in and download the data after accepting the data usage terms.
* **Storage:** Save to `<raw_data_directory>/WSSS4LUAD/`.

---

### 17. CoNSeP
1.  Visit [CoNSeP on OpenDataLab](https://opendatalab.com/OpenDataLab/CoNSeP).
2.  Download via the "Dataset Files" section or use the CLI/SDK tool.
* **Storage:** Save to `<raw_data_directory>/CoNSeP/`.

---

### 18. CPM15
1.  Visit the [CPM15 Google Drive folder](https://drive.google.com/drive/folders/11ko-GcDsPpA9GBHuCtl_jNzWQl6qY_-I).
2.  Select the required files and download.
* **Storage:** Extract to `<raw_data_directory>/CPM15/`.

---

### 19. CPM17
1.  Visit the [CPM17 Google Drive folder](https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA).
2.  Select the required files and download.
* **Storage:** Extract to `<raw_data_directory>/CPM17/`.

---

### 20. Kumar
1.  Visit the [Kumar Google Drive folder](https://drive.google.com/drive/folders/1bI3RyshWej9c4YoRW-_q7lh7FOFDFUrJ).
2.  Select the required files and download.
* **Storage:** Extract to `<raw_data_directory>/Kumar/`.

---

### 21. Lizard
Choose one of the two independent methods:
* **Option 1: Kaggle CLI**
    1.  Install the CLI: `pip install kaggle`.
    2.  Place your `kaggle.json` in `~/.kaggle/`.
    3.  Run: `kaggle datasets download -p <raw_data_directory>/Lizard aadimator/lizard-dataset --unzip`.
* **Option 2: Kaggle Website**
    1.  Visit the [Lizard Dataset on Kaggle](https://www.kaggle.com/datasets/aadimator/lizard-dataset).
    2.  Log in and click "Download."
* **Storage:** Unzip to `<raw_data_directory>/Lizard/`.

---

Follow the steps above to acquire the datasets locally. If any links expire or change, please refer to the official instructions on the respective challenge or host sites.