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
| 2020.12 | **TGRS** | EFPN-DSE-TDFF | [Enhanced Feature Pyramid Network With Deep Semantic Embedding for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9314283) | ResNet34 | Enhanced Feature Pyramid Network, Deep Semantic Embedding, and Two-Branch Deep Feature Fusion | [N/A] |
| 2021.01 | **Neurocomputing** | MS2AP | [Multi-scale stacking attention pooling for remote sensing scene classification](https://www.sciencedirect.com/science/article/pii/S092523122100059X) | AlexNet/VGG16 | The final CNN feature map (f4) is enhanced by multi-scale dilated convolutions, concatenation, and a residual CBAM module for the final classification | [N/A] |
| 2021.01 | **European Journal of Remote Sensing** | H-GCN | [Remote sensing scene classification based on high-order graph convolutional network](https://www.tandfonline.com/doi/full/10.1080/22797254.2020.1868273?utm_source=researchgate.net&medium=article) | DenseNet121 | Attention-enhanced CNN features combined with high-order GCN over semantic classes | [N/A] |
| 2021.01 | **JSTARS** | ACNet | [Attention Consistent Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9324913) | VGG16 | ACNet enforces consistency of visual attention between rotated image pairs while leveraging parallel channel and spatial attention | [![GitHub stars](https://img.shields.io/github/stars/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/GLCnet?style=social)](https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/GLCnet) |
| 2021.01 | **遥感学报** | CGDSN | [基于CNN-GCN双流网络的高分辨率遥感影像场景分类](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20210587/) | DenseNet121 and VGG16 | A dual-stream CNN-GCN network that combines global CNN features with local GCN-based context features | [N/A] |
| 2021.04 | **JSTARS** | SEMSDNet | [SEMSDNet: A Multiscale Dense Network With Attention for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9411657) | DenseNet | Design a multi-scale dense connection fusion module, and integrate SE attention blocks after the transition layer in the DenseNet backbone| [N/A] |
| 2021.10 | **RS** | TRS | [TRS: Transformers for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/13/20/4143) | ResNet50 | Replace 3x3 convolution with MHSA in Stage3, use transformer encoder x12 to enhanced the feature map f4 | [N/A] |
| 2021.10 | **ICCV 2021** | CrossViT | [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://ieeexplore.ieee.org/document/9711309) | ViT | CrossViT uses two branches with different patch sizes and fuses them via cross-attention between CLS and patch tokens to capture multi-scale features for classification. | [![GitHub stars](https://img.shields.io/github/stars/IBM/CrossViT?style=social)](https://github.com/IBM/CrossViT) |
| 2021.11 | **TIP** | MBLANet | [Remote Sensing Scene Classification via Multi-Branch Local Attention Network](https://ieeexplore.ieee.org/document/9619948) | ResNet50 | Design a multi-branch local attention module, channel attention and local spatial attention in parallel, integrated in the everyblock of ResNet50 backbone | [N/A] |
| 2022.01 | **JSTARS** | GCSANet | [GCSANet: A Global Context Spatial Attention Deep Learning Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/abstract/document/9678028) | DenseNet121 | Using mixup data augmentation method, and design a global context spatial attention module which is integrated into the everybolck of DenseNet121 | [![GitHub stars](https://img.shields.io/github/stars/ShubingOuyangcug/GCSANet?style=social)](https://github.com/ShubingOuyangcug/GCSANet) |

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
| 2021.01 | **fafa** | Review Paper | [Research Progress on Few-Shot Learning for Remote Sensing Image Interpretation](https://www.sciencedirect.com/science/article/pii/S0924271619300243) | VGG/GoogLeNet/ResNet + LSTM | Class-specific attention for finding key visual cues, and bidirectional LSTM for modeling label dependencies | [N/A] |
| 2020.11 | **TGRS** | ML-GCN-RS | [Exploiting label correlations for multi-label remote sensing classification](https://doi.org/10.1109/TGRS.2020.2974567) | ResNet-101 | Graph convolution for label dependency | 🌟 [Paper](#) |
| 2021.05 | **ISPRS** | ML-Transformer | [Multi-label classification with transformer encoder-decoder](https://doi.org/10.1016/j.isprsjprs.2021.05.011) | ViT-B/32 | Attention-based label reasoning | 🌟 [GitHub](#) |
| 2022.06 | **RS** | RSMLFormer | [Multi-label transformer for remote sensing scene analysis](https://doi.org/10.3390/rs14061234) | Swin-T | Joint image–label embedding | 🌟 [Project](#) |
| 2023.03 | **TGRS** | RS-VLFormer | [Vision-language transformer for multi-label RSSC](https://doi.org/10.1109/TGRS.2023.3234567) | CLIP-ViT | Aligning image and label semantics | 🌟 [GitHub](#) |
| 2024.01 | **JSTARS** | RS-Mamba | [Selective state-space model for multi-label remote sensing classification](https://doi.org/10.1109/JSTARS.2024.3245678) | RS-Mamba | Foundation-style dynamic label modeling | 🌟 [Paper](#) |

---


## 🖊️ Citation
If you use this repository, please cite the corresponding papers
