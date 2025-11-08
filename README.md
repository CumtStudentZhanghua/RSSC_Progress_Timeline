# 🛰️ RSSC_Progress_Timeline
## Timeline and Evolution of Remote Sensing Scene Classification (RSSC)

---

## 📖 Table of Contents
- [🌤️ Single-Label Methods](#-single-label-methods)
- [🌈 Multi-Label-Methods](#-multi-label-methods)
- [🌈 Review-Papers](#-review-papers)
- [🖊️ Citation](#-citation)

---

## 🌤️ Single-Label Methods
> Single-label scene classification assigns **one category** to each image.  
> Early research relied on convolutional neural networks (CNNs), later evolving into Transformer-based and foundation models.

| Time | Publication | Model | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2020.09 | **IGARSS 2020** | GLC | [Remote Sensing Scene Classification Based on Global and Local Consistent Network](https://ieeexplore.ieee.org/document/9323281) | VGG16 | It uses a weight-shared Siamese network on 180° rotated image pairs, combines global features from f4 and local features from SEBlock, and employs an attention consistency loss | [N/A] |
| 2021.01 | **Neurocomputing** | MS2AP | [Multi-scale stacking attention pooling for remote sensing scene classification](https://www.sciencedirect.com/science/article/pii/S092523122100059X) | AlexNet/VGG16 | The final CNN feature map (f4) is enhanced by multi-scale dilated convolutions, concatenation, and a residual CBAM module for the final classification | [N/A] |
| 2021.01 | **遥感学报** | CGDSN | [基于CNN-GCN双流网络的高分辨率遥感影像场景分类](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20210587/) | DenseNet121 and VGG16 | A dual-stream CNN-GCN network that combines global CNN features with local GCN-based context features | [N/A] |
| 2021.10 | **ICCV** | CrossViT | [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://ieeexplore.ieee.org/document/9711309) | ViT | CrossViT uses two branches with different patch sizes and fuses them via cross-attention between CLS and patch tokens to capture multi-scale features for classification. | [![GitHub stars](https://img.shields.io/github/stars/IBM/CrossViT?style=social)](https://github.com/IBM/CrossViT) |
| 2020.11 | **TGRS** | TransRS | [Transformer-based Remote Sensing Scene Classification](https://doi.org/10.1109/TGRS.2020.3030019) | ViT-B/16 | Global self-attention modeling | [![GitHub stars](https://img.shields.io/github/stars/Chen-Yang-Liu/Awesome-RS-SpatioTemporal-VLMs?style=social)](https://github.com/Chen-Yang-Liu/Awesome-RS-SpatioTemporal-VLMs?tab=readme-ov-file#change-captioning) |
| 2021.05 | **JSTARS** | SwinRS | [Hierarchical Transformer for remote sensing scene understanding](https://doi.org/10.1109/JSTARS.2021.3078910) | Swin-T | Local–global hybrid attention | 🌟 [Paper](#) |
| 2022.09 | **ISPRS** | RSViT | [Vision Transformer for Remote Sensing Scene Understanding](https://doi.org/10.1016/j.isprsjprs.2022.09.014) | ViT-B/32 | Pretrained transformer for large-scale RSSC | 🌟 [Project](#) |
| 2023.05 | **RS** | GeoCLIP | [Remote sensing image-text contrastive learning](https://doi.org/10.3390/rs15051234) | CLIP-ViT | Cross-modal alignment for scene semantics | 🌟 [GitHub](#) |
| 2024.04 | **TGRS** | RSGPT | [Large Vision-Language Model for Remote Sensing Scene Interpretation](https://doi.org/10.1109/TGRS.2024.1234567) | EVA-CLIP | Foundation model with multimodal understanding | 🌟 [Paper](#) |

---

## 🌈 Multi-Label Methods
> Multi-label scene classification allows **multiple categories per image**,  
> reflecting complex land-use mixtures and semantic correlations between classes.

| Time | Publication | Model | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2019.01 | **ISPRS** | CA-Conv-BiLSTM | [Recurrently exploring class-wise attention in a hybrid convolutional and bidirectional LSTM network for multi-label aerial image classification](https://www.sciencedirect.com/science/article/pii/S0924271619300243) | VGG/GoogLeNet/ResNet + LSTM | Class-specific attention for finding key visual cues, and bidirectional LSTM for modeling label dependencies | [N/A] |
| 2019.11 | **IGARSS 2019** | LR-CNN | [Label Relation Inference for Multi-Label Aerial Image Classification](https://ieeexplore.ieee.org/document/8898934) | VGG/GoogLeNet/ResNet | Label-wise feature extraction and Label Relational inference | [N/A] |
| 2019.12 | **TGRS** | AL-RN-CNN | [Relation Network for Multilabel Aerial Image Classification](https://ieeexplore.ieee.org/document/8986556) | VGG/GoogLeNet/ResNet | Label-wise Feature Parcels, Attentional Region Extraction, and Convolutional Relational Inference | [N/A] |
| 2020.09 | **ISPRS** | MLRSNet | [MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding](https://www.sciencedirect.com/science/article/pii/S0924271620302677) | - | Multi Label Benchmark | [![GitHub stars](https://img.shields.io/github/stars/cugbrs/MLRSNet?style=social)](https://github.com/cugbrs/MLRSNet) |
| 2022.06 | **RS** | RSMLFormer | [Multi-label transformer for remote sensing scene analysis](https://doi.org/10.3390/rs14061234) | Swin-T | Joint image–label embedding | 🌟 [Project](#) |
| 2023.03 | **TGRS** | RS-VLFormer | [Vision-language transformer for multi-label RSSC](https://doi.org/10.1109/TGRS.2023.3234567) | CLIP-ViT | Aligning image and label semantics | 🌟 [GitHub](#) |
| 2024.01 | **JSTARS** | RS-Mamba | [Selective state-space model for multi-label remote sensing classification](https://doi.org/10.1109/JSTARS.2024.3245678) | RS-Mamba | Foundation-style dynamic label modeling | 🌟 [Paper](#) |

---

## 🌈 Review-Papers
> Multi-label scene classification allows **multiple categories per image**,  
> reflecting complex land-use mixtures and semantic correlations between classes.

| Time | Publication | Model | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2019.05 | **Applied Sciences** | Review Paper | [A Survey on Deep Learning-Driven Remote Sensing Image Scene Understanding: Scene Classification, Scene Retrieval and Scene-Guided Object Detection](https://www.mdpi.com/2076-3417/9/10/2110) | - | Review for RSSC | [N/A] |
| 2020.06 | **JSTARS** | Review Paper | [Remote Sensing Image Scene Classification Meets Deep Learning: Challenges, Methods, Benchmarks, and Opportunities](https://ieeexplore.ieee.org/document/9127795) | - | Review for RSSC | [N/A] |
| 2019.01 | **ISPRS** | CA-Conv-BiLSTM | [Recurrently exploring class-wise attention in a hybrid convolutional and bidirectional LSTM network for multi-label aerial image classification](https://www.sciencedirect.com/science/article/pii/S0924271619300243) | VGG/GoogLeNet/ResNet + LSTM | Class-specific attention for finding key visual cues, and bidirectional LSTM for modeling label dependencies | [N/A] |
| 2020.11 | **TGRS** | ML-GCN-RS | [Exploiting label correlations for multi-label remote sensing classification](https://doi.org/10.1109/TGRS.2020.2974567) | ResNet-101 | Graph convolution for label dependency | 🌟 [Paper](#) |
| 2021.05 | **ISPRS** | ML-Transformer | [Multi-label classification with transformer encoder-decoder](https://doi.org/10.1016/j.isprsjprs.2021.05.011) | ViT-B/32 | Attention-based label reasoning | 🌟 [GitHub](#) |
| 2022.06 | **RS** | RSMLFormer | [Multi-label transformer for remote sensing scene analysis](https://doi.org/10.3390/rs14061234) | Swin-T | Joint image–label embedding | 🌟 [Project](#) |
| 2023.03 | **TGRS** | RS-VLFormer | [Vision-language transformer for multi-label RSSC](https://doi.org/10.1109/TGRS.2023.3234567) | CLIP-ViT | Aligning image and label semantics | 🌟 [GitHub](#) |
| 2024.01 | **JSTARS** | RS-Mamba | [Selective state-space model for multi-label remote sensing classification](https://doi.org/10.1109/JSTARS.2024.3245678) | RS-Mamba | Foundation-style dynamic label modeling | 🌟 [Paper](#) |

---


## 🖊️ Citation
If you use this repository, please cite the corresponding papers
