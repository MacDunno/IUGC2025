# Unlabeled Data-Driven Fetal Landmark Detection in Intrapartum Ultrasound

This repository contains the implementation of our **MICCAI 2025 IUGC Challenge winning solution** for **fetal landmark detection in intrapartum ultrasound** and **automatic Angle of Progression (AoP) estimation**.

Our method is built on a **modified TransUNet** with a **TinyViT backbone**, and further improves performance through:

- **MAE-assisted knowledge distillation** from an ultrasound foundation model
- **Semi-supervised learning with pseudo-labeling**
- **Label perturbation for device-domain adaptation**

On the **IUGC2025 Challenge test set**, our method achieved:

- **Mean Radial Error (MRE): 11.6749 px**
- **Mean Absolute AoP Error: 3.8061°**

---

## 🧩 Overview

The goal of this work is to automatically detect three anatomical landmarks in intrapartum ultrasound images:

- **PS1**
- **PS2**
- **FH1**

These landmarks are then used to compute the **Angle of Progression (AoP)**, an important clinical parameter for assessing fetal head descent during labor.

Our framework consists of two stages:

1. **Pretraining**: domain-specific representation learning with MAE-assisted knowledge distillation
2. **Main training**: heatmap-based landmark detection with TransUNet, enhanced by pseudo-labeling and device-domain adaptation


---

## ⚙️ Usage

### 1. Pretraining

Run the code in `pretrain/` to pretrain the TinyViT backbone with MAE-assisted knowledge distillation.

### 2. Training

Run the code in `train/` to train the landmark detection model based on the pretrained backbone.

### 3. Evaluation

Use the trained model to predict landmark heatmaps and compute the final AoP from the detected coordinates.

> Detailed commands, environment setup, and dataset preparation instructions will be released soon.

---

 ## 📚 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{ma2026unlabeled,
  title     = {Unlabeled Data-Driven Fetal Landmark Detection in Intrapartum Ultrasound},
  author    = {Ma, Chen and Li, Yunshu and Guo, Bowen and Jiao, Jing and Huang, Yi and Wang, Yuanyuan and Guo, Yi},
  booktitle = {IUGC 2025},
  series    = {Lecture Notes in Computer Science},
  volume    = {16317},
  pages     = {14--23},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-11616-1_2}
}

@article{tinyusfm,
  author={Ma, Chen and Jiao, Jing and Liang, Shuyu and Fu, Junhu and Wang, Qin and Li, Zeju and Wang, Yuanyuan and Guo, Yi},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models}, 
  year={2026},
  pages={1-14},
  doi={10.1109/JBHI.2026.3678309}
}
```

---

## 📝 License

This project is released for academic research only.
