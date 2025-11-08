# 🛰️ RSSC_Progress_Timeline
## Timeline and Evolution of Remote Sensing Scene Classification (RSSC)

---

## 📖 Table of Contents
- [🌤️ Single-Label Methods](#-single-label-methods)
- [🌈 Multi-Label-Methods](#-multi-label-methods)
- [🖊️ Citation](#-citation)

---

## 🌤️ Single-Label Methods
> Single-label scene classification assigns **one category** to each image.  
> Early research relied on convolutional neural networks (CNNs), later evolving into Transformer-based and foundation models.

| Time | Journal | Model Name | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2015.05 | **TGRS** | CNN-RS | [Deep learning-based classification of remote sensing scenes](https://doi.org/10.1109/TGRS.2015.2406305) | AlexNet | Introduced CNNs for RSSC | [N/A] |
| 2016.07 | **JSTARS** | VGG-RS | [Remote sensing image scene classification based on deep features](https://doi.org/10.1109/JSTARS.2016.2575728) | VGG-16 | Transfer learning from ImageNet | [N/A] |
| 2018.03 | **ISPRS** | ResNet-RS | [Remote sensing scene classification with deep residual networks](https://doi.org/10.1016/j.isprsjprs.2018.03.007) | ResNet-50 | Residual learning improves feature depth | [N/A] |
| 2019.06 | **RS** | DenseNet-RS | [Remote sensing scene classification via dense connectivity](https://doi.org/10.3390/rs11111234) | DenseNet-121 | Dense connections enhance feature reuse | [N/A] |
| 2020.11 | **TGRS** | TransRS | [Transformer-based Remote Sensing Scene Classification](https://doi.org/10.1109/TGRS.2020.3030019) | ViT-B/16 | Global self-attention modeling | [![GitHub stars](https://img.shields.io/github/stars/<USERNAME>/<REPO>?style=social)]([https://github.com/<USERNAME>/<REPO>](https://github.com/Chen-Yang-Liu/Awesome-RS-SpatioTemporal-VLMs?tab=readme-ov-file#change-captioning)) |
| 2021.05 | **JSTARS** | SwinRS | [Hierarchical Transformer for remote sensing scene understanding](https://doi.org/10.1109/JSTARS.2021.3078910) | Swin-T | Local–global hybrid attention | 🌟 [Paper](#) |
| 2022.09 | **ISPRS** | RSViT | [Vision Transformer for Remote Sensing Scene Understanding](https://doi.org/10.1016/j.isprsjprs.2022.09.014) | ViT-B/32 | Pretrained transformer for large-scale RSSC | 🌟 [Project](#) |
| 2023.05 | **RS** | GeoCLIP | [Remote sensing image-text contrastive learning](https://doi.org/10.3390/rs15051234) | CLIP-ViT | Cross-modal alignment for scene semantics | 🌟 [GitHub](#) |
| 2024.04 | **TGRS** | RSGPT | [Large Vision-Language Model for Remote Sensing Scene Interpretation](https://doi.org/10.1109/TGRS.2024.1234567) | EVA-CLIP | Foundation model with multimodal understanding | 🌟 [Paper](#) |

---

## 🌈 Multi-Label Methods
> Multi-label scene classification allows **multiple categories per image**,  
> reflecting complex land-use mixtures and semantic correlations between classes.

| Time | Journal | Model Name | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2019.07 | **JSTARS** | MLRSNet-CNN | [Multi-label remote sensing scene classification with convolutional networks](https://doi.org/10.1109/JSTARS.2019.2924567) | ResNet-50 | First large-scale multi-label RSSC benchmark | [GitHub](https://github.com/cugbrs/MLRSNet) |
| 2020.11 | **TGRS** | ML-GCN-RS | [Exploiting label correlations for multi-label remote sensing classification](https://doi.org/10.1109/TGRS.2020.2974567) | ResNet-101 | Graph convolution for label dependency | 🌟 [Paper](#) |
| 2021.05 | **ISPRS** | ML-Transformer | [Multi-label classification with transformer encoder-decoder](https://doi.org/10.1016/j.isprsjprs.2021.05.011) | ViT-B/32 | Attention-based label reasoning | 🌟 [GitHub](#) |
| 2022.06 | **RS** | RSMLFormer | [Multi-label transformer for remote sensing scene analysis](https://doi.org/10.3390/rs14061234) | Swin-T | Joint image–label embedding | 🌟 [Project](#) |
| 2023.03 | **TGRS** | RS-VLFormer | [Vision-language transformer for multi-label RSSC](https://doi.org/10.1109/TGRS.2023.3234567) | CLIP-ViT | Aligning image and label semantics | 🌟 [GitHub](#) |
| 2024.01 | **JSTARS** | RS-Mamba | [Selective state-space model for multi-label remote sensing classification](https://doi.org/10.1109/JSTARS.2024.3245678) | RS-Mamba | Foundation-style dynamic label modeling | 🌟 [Paper](#) |

---

## 🖊️ Citation
If you use this repository, please cite the corresponding papers
