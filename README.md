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

| Time | Model Name | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|-------------|--------------|----------------|-----------|---------------|
| 2015.05 | CNN-RS | [Deep learning-based classification of remote sensing scenes](https://doi.org/10.1109/TGRS.2015.2406305) | AlexNet | Introduced CNNs for RSSC | [N/A] |
| 2016.07 | VGG-RS | Remote sensing image scene classification based on deep features | VGG-16 | Transfer learning from ImageNet | [N/A] |
| 2018.03 | ResNet-RS | Remote sensing scene classification with deep residual networks | ResNet-50 | Residual learning improves feature depth | [N/A] |
| 2019.06 | DenseNet-RS | Remote sensing scene classification via dense connectivity | DenseNet-121 | Dense connections enhance feature reuse | [N/A] |
| 2020.11 | TransRS | Transformer-based Remote Sensing Scene Classification | ViT-B/16 | Global self-attention modeling | 🌟 [GitHub](#) |
| 2021.05 | SwinRS | Hierarchical Transformer for remote sensing scene understanding | Swin-T | Local–global hybrid attention | 🌟 [Paper](#) |
| 2022.09 | RSViT | Vision Transformer for Remote Sensing Scene Understanding | ViT-B/32 | Pretrained transformer for large-scale RSSC | 🌟 [Project](#) |
| 2023.05 | GeoCLIP | Remote sensing image-text contrastive learning | CLIP-ViT | Cross-modal alignment for scene semantics | 🌟 [GitHub](#) |
| 2024.04 | RSGPT | Large Vision-Language Model for Remote Sensing Scene Interpretation | EVA-CLIP | Foundation model with multimodal understanding | 🌟 [Paper](#) |

---

## 🌈 Multi-Label Methods
> Multi-label scene classification allows **multiple categories per image**,  
> reflecting complex land-use mixtures and semantic correlations between classes.

| Time | Model Name | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|-------------|--------------|----------------|-----------|---------------|
| 2019.07 | MLRSNet-CNN | Multi-label remote sensing scene classification with convolutional networks | ResNet-50 | First large-scale multi-label RSSC benchmark | [GitHub](https://github.com/cugbrs/MLRSNet) |
| 2020.11 | ML-GCN-RS | Exploiting label correlations for multi-label remote sensing classification | ResNet-101 | Graph convolution for label dependency | 🌟 [Paper](#) |
| 2021.05 | ML-Transformer | Multi-label classification with transformer encoder-decoder | ViT-B/32 | Attention-based label reasoning | 🌟 [GitHub](#) |
| 2022.06 | RSMLFormer | Multi-label transformer for remote sensing scene analysis | Swin-T | Joint image–label embedding | 🌟 [Project](#) |
| 2023.03 | RS-VLFormer | Vision-language transformer for multi-label RSSC | CLIP-ViT | Aligning image and label semantics | 🌟 [GitHub](#) |
| 2024.01 | RS-Mamba | Selective state-space model for multi-label remote sensing classification | RS-Mamba | Foundation-style dynamic label modeling | 🌟 [Paper](#) |

---

## 🖊️ Citation
If you use this repository, please cite the corresponding papers
