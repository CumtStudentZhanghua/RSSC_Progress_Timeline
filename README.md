# 🛰️ RSSC_Progress_Timeline
## Timeline and Evolution of Remote Sensing Scene Classification (RSSC)

---

## 📖 Table of Contents
- [1.Single-Label Methods](#-1.-single-label-methods)
- [2.Multi-Label-Methods](#-2.-multi-label-methods)
- [3.Review-Papers](#-3.-review-papers)
- [4.Citation](#-4.-citation)

---

## 1.Single-Label Methods
> Single-label scene classification assigns **one category** to each image.  
> Early research relied on convolutional neural networks (CNNs), later evolving into Transformer-based and foundation models.

| Time | Publication | Model | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2020.09 | **IGARSS 2020** | GLC | [Remote Sensing Scene Classification Based on Global and Local Consistent Network](https://ieeexplore.ieee.org/document/9323281) | VGG16 | It uses a weight-shared Siamese network on 180° rotated image pairs, combines global features from f4 and local features from SEBlock, and employs an attention consistency loss | [N/A] |
| 2020.12 | **TGRS** | EFPN-DSE-TDFF | [Enhanced Feature Pyramid Network With Deep Semantic Embedding for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9314283) | ResNet34 | Enhanced Feature Pyramid Network, Deep Semantic Embedding, and Two-Branch Deep Feature Fusion | [N/A] |
| 2020.12 | **GRSL** | MSNet | [MSNet: A Multiple Supervision Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9302612) | ResNet50 | xxx | [N/A] |
| 2021.01 | **Neurocomputing** | MS2AP | [Multi-scale stacking attention pooling for remote sensing scene classification](https://www.sciencedirect.com/science/article/pii/S092523122100059X) | AlexNet/VGG16 | The final CNN feature map (f4) is enhanced by multi-scale dilated convolutions, concatenation, and a residual CBAM module for the final classification | [N/A] |
| 2021.01 | **European Journal of Remote Sensing** | H-GCN | [Remote sensing scene classification based on high-order graph convolutional network](https://www.tandfonline.com/doi/full/10.1080/22797254.2020.1868273?utm_source=researchgate.net&medium=article) | DenseNet121 | Attention-enhanced CNN features combined with high-order GCN over semantic classes | [N/A] |
| 2021.01 | **JSTARS** | ACNet | [Attention Consistent Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9324913) | VGG16 | ACNet enforces consistency of visual attention between rotated image pairs while leveraging parallel channel and spatial attention | [![GitHub stars](https://img.shields.io/github/stars/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/GLCnet?style=social)](https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/GLCnet) |
| 2021.01 | **遥感学报** | CGDSN | [基于CNN-GCN双流网络的高分辨率遥感影像场景分类](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20210587/) | DenseNet121 and VGG16 | A dual-stream CNN-GCN network that combines global CNN features with local GCN-based context features | [N/A] |
| 2021.03 | **TNNLS** | DFAGCN | [Deep Feature Aggregation Framework Driven by Graph Convolutional Network for Scene Classification in Remote Sensing](https://ieeexplore.ieee.org/document/9405447) | XXX | XXX | [N/A] |
| 2021.04 | **JSTARS** | SEMSDNet | [SEMSDNet: A Multiscale Dense Network With Attention for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9411657) | DenseNet | Design a multi-scale dense connection fusion module, and integrate SE attention blocks after the transition layer in the DenseNet backbone| [N/A] |
| 2021.10 | **RS** | TRS | [TRS: Transformers for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/13/20/4143) | ResNet50 | Replace 3x3 convolution with MHSA in Stage3, use transformer encoder x12 to enhanced the feature map f4 | [N/A] |
| 2021.10 | **ICCV 2021** | CrossViT | [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://ieeexplore.ieee.org/document/9711309) | ViT | CrossViT uses two branches with different patch sizes and fuses them via cross-attention between CLS and patch tokens to capture multi-scale features for classification. | [![GitHub stars](https://img.shields.io/github/stars/IBM/CrossViT?style=social)](https://github.com/IBM/CrossViT) |
| 2021.11 | **TIP** | MBLANet | [Remote Sensing Scene Classification via Multi-Branch Local Attention Network](https://ieeexplore.ieee.org/document/9619948) | ResNet50 | Design a multi-branch local attention module, channel attention and local spatial attention in parallel, integrated in the everyblock of ResNet50 backbone | [N/A] |
| 2021.12 | **JSTARS** | RANet | [Relation-Attention Networks for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9652121) | xxx | xxx | [N/A] |
| 2022.01 | **JSTARS** | GCSANet | [GCSANet: A Global Context Spatial Attention Deep Learning Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/abstract/document/9678028) | DenseNet121 | Using mixup data augmentation method, and design a global context spatial attention module which is integrated into the everybolck of DenseNet121 | [![GitHub stars](https://img.shields.io/github/stars/ShubingOuyangcug/GCSANet?style=social)](https://github.com/ShubingOuyangcug/GCSANet) |
| 2022.01 | **GRSL** | LML | [Remote Sensing Scene Classification by Local–Global Mutual Learning](https://ieeexplore.ieee.org/abstract/document/9709818) | ResNet/GoogleNet | Generate local regions from the original image using a heat map derived from deep CNN features, and train a two-branch network where global and local branches mutually learn to enhance complementary feature representation | [N/A] |
| 2022.03 | **TGRS** | SCViT | [SCViT: A Spatial-Channel Feature Preserving Vision Transformer for Remote Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/9729845) | xxx | xxx | [N/A] |
| 2022.03 | **RS** | MopNet-GCN-ResNet50 | [Multi-Output Network Combining GNN and CNN for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/14/6/1478) | xxx | xxx | [N/A] |
| 2022.04 | **GRSL** | MLF2Net_SAGM | [Multilayer Feature Fusion Network With Spatial Attention and Gated Mechanism for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9770786) | xxx | xxx | [N/A] |
| 2022.04 | **RS** | ACGLNet | [An Attention Cascade Global–Local Network for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/14/9/2042) | xxx | xxx | [N/A] |
| 2022.05 | **GRSL** | MITformer | [MITformer: A Multiinstance Vision Transformer for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/abstract/document/9777975) | xxx | xxx | [N/A] |
| 2022.07 | **TGRS** | EMTCAL | [EMTCAL: Efficient Multiscale Transformer and Cross-Level Attention Learning for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9844016) | XXX | XXX | [EMTCAL](https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/EMTCAL) |
| 2022.07 | **TGRS** | T-CNN | [Transferring CNN With Adaptive Learning for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9829875) | xxx | xxx | [N/A] |
| 2022.09 | **GRSL** | MFST | [MFST: A Multi-Level Fusion Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9882107) | xxx | xxx | [N/A] |
| 2023.01 | **遥感学报** | THViT | [基于双阶段高阶Transformer的遥感图像场景分类](https://www.ygxb.ac.cn/thesisDetails?columnId=45880183&lang=zh) | xxx | xxx | [N/A] |
| 2023.01 | **TIP** | SAGN | [SAGN: Semantic-Aware Graph Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10025702) | xxx | xxx | [SAGN](https://github.com/TangXu-Group/SAGN) |
| 2023.03 | **TGRS** | L2RCF | [Local and Long-Range Collaborative Learning for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10093899) | xxx | xxx | [N/A] |
| 2023.04 | **TGRS** | SF-MSFormer | [An Explainable Spatial–Frequency Multiscale Transformer for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10097579) | xxx | xxx | [SF-MSFormer]([https://github.com/TangXu-Group/SAGN](https://github.com/yutinyang/SF-MSFormer)) |
| 2023.05 | **TGRS** | EFPF | [Energy-Based CNN Pruning for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10128858) | xxx | xxx | [N/A] |
| 2023.06 | **JSTARS** | IBSwin-CR | [Inductive Biased Swin-Transformer With Cyclic Regressor for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10186881) | xxx | xxx | [N/A] |
| 2023.07 | **RS** | LTNet | [Faster and Better: A Lightweight Transformer Network for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/15/14/3645) | xxx | xxx | [N/A] |
| 2023.07 | **TGRS** | MGSNet | [Remote-Sensing Scene Classification via Multistage Self-Guided Separation Network](https://ieeexplore.ieee.org/document/10184498) | xxx | xxx | [N/A] |
| 2023.08 | **GRSL** | CSCANet | [Contextual Spatial-Channel Attention Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10216315) | xxx | xxx | [N/A] |
| 2023.11 | **TGRS** | HFFT-PD | [Hierarchical Feature Fusion of Transformer With Patch Dilating for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/abstract/document/10314557) | xxx | xxx | [N/A] |
| 2023.11 | **TGRS** | HFAM | [A Hyperparameter-Free Attention Module Based on Feature Map Mathematical Calculation for Remote-Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/10330628) | xxx | xxx | [N/A] |
| 2023.11 | **TGRS** | CDLNet | [CDLNet: Collaborative Dictionary Learning Network for Remote Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/10330735) | xxx | xxx | [![GitHub stars](https://img.shields.io/github/liuofficial/CDLNet/CrossViT?style=social)](https://github.com/liuofficial/CDLNet) |
| 2024.01 | **JSTARS** | PFFGCN | [Progressive Feature Fusion Framework Based on Graph Convolutional Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10381852) | xxx | xxx | [N/A] |
| 2024.01 | **JSTARS** | CASD | [Class-Aware Self-Distillation for Remote Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/10361538) | xxx | xxx | [N/A] |
| 2024.01 | **JSTARS** | LSMNet | [Large Kernel Separable Mixed ConvNet for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10399773) | xxx | xxx | [N/A] |
| 2024.01 | **GRSL** | PSCLI-TF | [PSCLI-TF: Position-Sensitive Cross-Layer Interactive Transformer Model for Remote Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/10415469) | xxx | xxx | [N/A] |
| 2024.05 | **GRSL** | CLENet | [Label Embedding Based on Interclass Correlation Mining for Remote Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/10526291) | xxx | xxx | [N/A] |
| 2024.05 | **TGRS** | SDT2Net | [Remote Sensing Scene Classification via Second-Order Differentiable Token Transformer Network](https://ieeexplore.ieee.org/document/10542965) | xxx | xxx | [![GitHub stars](https://img.shields.io/github/stars/RSIP-NJUPT/SDT2Net?style=social)](https://github.com/RSIP-NJUPT/SDT2Net) |
| 2024.07 | **IGARSS 2024** | MST | [Multi-Scale Sparse Transformer for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10642862) | xxx | xxx | [N/A] |
| 2024.07 | **IGARSS 2024** | MCRNet | [Remote Sensing Image Scene Classification With Multi-View Collaborative Representation Network](https://ieeexplore.ieee.org/document/10640773) | xxx | xxx | [N/A] |
| 2024.09 | **GRSL** | SAF-Net | [Remote Sensing Scene Classification Based on Semantic-Aware Fusion Network](https://ieeexplore.ieee.org/document/10700843) | xxx | xxx | [N/A] |
| 2024.10 | **JSTARS** | HGTNet | [A Hierarchical Graph-Enhanced Transformer Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10742489) | xxx | xxx | [N/A] |
| 2024.10 | **遥感学报** | ADC-CPANet | [ADC-CPANet：一种局部—全局特征融合的遥感图像分类方法](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20232658/) | xxx | xxx | [N/A] |
| 2024.12 | **RS** | MSCT-HCST | [Multiple Hierarchical Cross-Scale Transformer for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/17/1/42) | xxx | xxx | [N/A] |
| 2025.01 | **TGRS** | MSCN | [Multiscale Sparse Cross-Attention Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10820553) | xxx | xxx | [MSCN](https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/MSCN) |
| 2025.01 | **TGRS** | XXX | [Remote Sensing Scene Classification via Pseudo-Category-Relationand Orthogonal Feature Learning](https://ieeexplore.ieee.org/document/10879025) | xxx | xxx | [N/A] |
| 2025.01 | **JSTARS** | IRCHKD | [An Inverted Residual Cross Head Knowledge Distillation Network for Remote Sensing Scene Image Classification](https://ieeexplore.ieee.org/document/10870144) | xxx | xxx | [N/A] |
| 2025.02 | **TGRS** | LSDGNet | [Label Semantic Dynamic Guidance Network for Remote Sensing Image Scene Classification](https://ieeexplore.ieee.org/document/10879025) | xxx | xxx | [N/A] |
| 2025.02 | **RS** | STMSF | [STMSF: Swin Transformer with Multi-Scale Fusion for Remote Sensing Scene Classification](https://www.mdpi.com/2072-4292/17/4/668) | xxx | xxx | [N/A] |
| 2025.02 | **RS** | ATMformer | [ATMformer: An Adaptive Token Merging Vision Transformer for Remote Sensing Image Scene Classification](https://www.mdpi.com/2072-4292/17/4/660) | XXX | XXX | [N/A] |
| 2025.03 | **TGRS** | SceneFormer | [SceneFormer: Neural Architecture Search of Transformers for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10942436) | XXX | XXX | [N/A] |
| 2025.04 | **中国图象图形学报** |  | [用于遥感场景分类的全局—局部特征耦合网络](https://ieeexplore.ieee.org/document/10942436) | XXX | XXX | [N/A] |
| 2025.07 | **JSTARS** | ACTFormer | [ACTFormer: A Transformer Network With Attention and Convolutional Synergy for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/11080299) | XXX | XXX | [N/A] |



MSNet: A Multiple Supervision Network for Remote Sensing Scene Classification
---

## 2.Multi-Label Methods
> Multi-label scene classification allows **multiple categories per image**,  
> reflecting complex land-use mixtures and semantic correlations between classes.

| Time | Publication | Model | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2019.01 | **ISPRS** | CA-Conv-BiLSTM | [Recurrently exploring class-wise attention in a hybrid convolutional and bidirectional LSTM network for multi-label aerial image classification](https://www.sciencedirect.com/science/article/pii/S0924271619300243) | VGG/GoogLeNet/ResNet + LSTM | Class-specific attention for finding key visual cues, and bidirectional LSTM for modeling label dependencies | [N/A] |
| 2019.11 | **IGARSS 2019** | LR-CNN | [Label Relation Inference for Multi-Label Aerial Image Classification](https://ieeexplore.ieee.org/document/8898934) | VGG/GoogLeNet/ResNet | Label-wise feature extraction and Label Relational inference | [N/A] |
| 2019.12 | **TGRS** | AL-RN-CNN | [Relation Network for Multilabel Aerial Image Classification](https://ieeexplore.ieee.org/document/8986556) | VGG/GoogLeNet/ResNet | Label-wise Feature Parcels, Attentional Region Extraction, and Convolutional Relational Inference | [N/A] |
| 2020.09 | **ISPRS** | MLRSNet | [MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding](https://www.sciencedirect.com/science/article/pii/S0924271620302677) | - | Multi Label Benchmark | [![GitHub stars](https://img.shields.io/github/stars/cugbrs/MLRSNet?style=social)](https://github.com/cugbrs/MLRSNet) |
| 2021.08 | **TGRS** | MultiScene | [MultiScene: A Large-Scale Dataset and Benchmark for Multiscene Recognition in Single Aerial Images](https://ieeexplore.ieee.org/document/9537917) | - | Multi Label Benchmark | [MultiScene](https://gitlab.lrz.de/ai4eo/reasoning/multiscene) |
| 2022.03 | **AAAI 2024** | MulSupCon | [Multi-Label Supervised Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29619) | XXX | XXX | [N/A] |
| 2022.04 | **TGRS** | SCIDA | [SCIDA: Self-Correction Integrated Domain Adaptation From Single- to Multi-Label Aerial Images](https://ieeexplore.ieee.org/document/9762917) | XXX | XXX | [![GitHub stars](https://img.shields.io/github/stars/Ryan315/Single2multi-DA?style=social)](https://github.com/Ryan315/Single2multi-DA) |
| 2024.06 | **TGRS** | GMFANet | [Gradient-Guided Multiscale Focal Attention Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10599200) | XXX | XXX | [![GitHub stars](https://img.shields.io/github/stars/bling2beyond/GMFANet?style=social)](https://github.com/bling2beyond/GMFANet) |
| 2024.07 | **TGRS** | Multitask Fine-Grained Feature Mining | [Multitask Fine-Grained Feature Mining for Multilabel Remote Sensing Image Classification](https://ieeexplore.ieee.org/document/10595130) | XXX | ML-GCN + Graph Transformer Layer + Class-Specific Feature Extraction + Multi-scale Fusion Strategy and Visual Attention Mechanism | [N/A] |
| 2024.10 | **TGRS** | ML-HKG | [Hierarchical Knowledge Graph for Multilabel Classification of Remote Sensing Images](https://ieeexplore.ieee.org/document/10741275) | XXX | XXX | [N/A] |
| 2024.10 | **TGRS** | CAGRN | [Cross-Attention-Driven Adaptive Graph Relational Network for Multilabel Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10707603) | XXX | XXX | [N/A] |
| 2025.01 | **TGRS** | SFIN | [Semantic-Assisted Feature Integration Network for Multilabel Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10802953) | XXX | XXX | [SFIN](https://github.com/TangXu-Group/multilabelRSSC/tree/main/SFIN) |
| 2025.01 | **IEEE Transactions on Circuits and Systems for Video Technology（IEEE TCSVT）** | MLMamba | [MLMamba: A Mamba-Based Efficient Network for Multi-Label Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/10857393) | XXX | XXX | [MLMamba](https://github.com/TangXu-Group/multilabelRSSC/tree/main/MLMamba) |
| 2025.01 | **JSTARS** | CMCL | [Cross-Modal Compositional Learning for Multilabel Remote Sensing Image Classification](https://ieeexplore.ieee.org/document/10872829) | XXX | XXX | [N/A] |
| 2025.01 | **CVPR 2025** | SR-based MLRSSC | [Multi-Label Scene Classification in Remote Sensing Benefits from Image Super-Resolution](https://arxiv.org/abs/2501.06720) | XXX | XXX | [N/A] |


---

## 3.Review-Papers
> Multi-label scene classification allows **multiple categories per image**,  
> reflecting complex land-use mixtures and semantic correlations between classes.

| Time | Publication | Model | Paper Title | Visual Encoder | Key Idea | Code/Project |
|------|----------|-------------|--------------|----------------|-----------|---------------|
| 2019.05 | **Applied Sciences** | Review Paper | [A Survey on Deep Learning-Driven Remote Sensing Image Scene Understanding: Scene Classification, Scene Retrieval and Scene-Guided Object Detection](https://www.mdpi.com/2076-3417/9/10/2110) | - | Review for RSSC | [N/A] |
| 2020.06 | **JSTARS** | Review Paper | [Remote Sensing Image Scene Classification Meets Deep Learning: Challenges, Methods, Benchmarks, and Opportunities](https://ieeexplore.ieee.org/document/9127795) | - | Review for RSSC | [N/A] |
| 2021.01 | **fafa** | Benchmark | [Research Progress on Few-Shot Learning for Remote Sensing Image Interpretation](https://www.sciencedirect.com/science/article/pii/S0924271619300243) | VGG/GoogLeNet/ResNet + LSTM | Class-specific attention for finding key visual cues, and bidirectional LSTM for modeling label dependencies | [N/A] |
| 2023.09 | **中国图象图形学报** | Review Paper | [Transformer驱动的图像分类研究进展](https://www.cjig.cn/en/article/doi/10.11834/jig.220799/) | - | Review for RSSC | [N/A] |
| 2023.09 | **RS** | Review Paper | [Deep Learning for Remote Sensing Image Scene Classification: A Review and Meta-Analysis](https://www.mdpi.com/2072-4292/15/19/4804) | - | Review for RSSC | [N/A] |
| 2024.10 | **航天返回与遥感** | Review Paper | [遥感影像场景分类研究进展](https://htfhyyg.spacejournal.cn/article/doi/10.3969/j.issn.1009-8518.2024.04.013) | - | Review for RSSC | [N/A] |
| 2024.11 | **遥感学报** | Review Paper | [高分辨率遥感图像场景分类研究进展](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20243519/) | - | Review for RSSC | [N/A] |
| 2024.12 | **Engineering Applications of Artificial Intelligence** | Review Paper | [Lightweight deep learning models for aerial scene classification: A comprehensive survey](https://www.sciencedirect.com/science/article/pii/S0952197624020189?via%3Dihub) | - | Review for RSSC | [N/A] |


---


## 4.Citation
If you use this repository, please cite the corresponding papers
