# 🧠 ViT–MiT Brain Tumor Classification (IEEE Research Implementation)

This repository presents an implementation of a **hybrid Vision Transformer (ViT) and Mix Transformer (MiT) ensemble** for **multiclass brain tumor classification using MRI images**, developed as part of an IEEE-style research study.

The model classifies MRI scans into:
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

**Performance:**
- **Test Accuracy:** 99.01%  
- **Macro F1-score:** 0.99  

---

## 🔍 Key Features

- Hybrid **ViT + MiT ensemble**
- Logit-level fusion (**α = 0.6**)
- Progressive transfer learning
- Cosine LR scheduling + warmup
- Automatic Mixed Precision (AMP)
- Test-Time Augmentation (TTA)
- Attention-based explainability

---

## 📊 Dataset

- **Total Images:** 7,200  
- **Balanced across 4 classes**

| Split | Images per Class | Total |
|------|----------------|------|
| Train | 1,400 | 5,600 |
| Test | 400 | 1,600 |

Source:  
J. Cheng, *Brain Tumor Dataset*, figshare (2017)  
https://doi.org/10.6084/m9.figshare.1512427.v5  

---

## 🧱 Model Overview

### Vision Transformer (ViT)
Captures **global spatial dependencies** using self-attention.

### Mix Transformer (MiT - SegFormer)
Extracts **hierarchical multi-scale features**.

### Ensemble Fusion

\[
z_{ens} = \alpha z_{ViT} + (1 - \alpha) z_{MiT}, \quad \alpha = 0.6
\]

---

## ⚙️ Training Strategy

- **Optimizer:** AdamW  
- **Epochs:** 50  
- **Batch Size:** 32  
- **Hardware:** NVIDIA T4  

### Key Techniques
- Progressive backbone unfreezing  
- Cosine LR schedule with warmup  
- Label smoothing  
- AMP acceleration  

---

## 🧪 Preprocessing

- Resize: 224 × 224  
- ImageNet normalization  
- Augmentations:
  - Horizontal flip  
  - Rotation (±10°)  

---

# 📈 Results & Visualizations

## Dataset Distribution

![Dataset Distribution](dataset distribution.png)

*Balanced class-wise distribution across training and testing splits.*

---

## Training Curves

![Training Curves](Training Curves.png)

*Training loss and validation accuracy trends across 50 epochs with progressive unfreezing.*

---

## Model Comparison

![Model Comparison](Model Comparison.png)

*Performance comparison between ViT, MiT, and the proposed ensemble.*

---

## Confusion Matrix

![Confusion Matrix](Confusion Matrix.png)

*High concentration along diagonal indicates strong classification performance across all classes.*

---

## ROC Curve

![ROC Curve](ROC curve.png)

*Strong class separability with high AUC across all tumor categories.*

---

## Attention Maps

![Attention Map](Attention map.png)

*Model focuses on clinically relevant tumor regions, improving interpretability.*

---

## 📄 Research Context

This work corresponds to an IEEE-format research study on:

**Hybrid Vision Transformer and Mix Transformer Ensemble for Multiclass Brain Tumor Classification Using MRI Images**

Includes:
- Independent test evaluation  
- Ablation study  
- Class-wise metrics  
- Explainability analysis  

---

## ⚠️ Limitations

- Based on 2D MRI slices  
- Limited dataset diversity  
- Meningioma vs glioma overlap  

---

## 🚀 Future Work

- 3D MRI extension  
- Multi-institution validation  
- Clinical deployment testing  

---

## 📜 License

This project is intended for **academic and research use**.

- Code: Open for research and educational purposes  
- Dataset: Subject to original figshare license  
- Research aligned with **IEEE publication standards**

---

## 🙏 Acknowledgment

- SRM Institute of Science and Technology  
- Public MRI dataset contributors  

---

## ⭐ Citation

```bibtex
@article{vit_mit_brain_tumor,
  title={Hybrid Vision Transformer and Mix Transformer Ensemble for Multiclass Brain Tumor Classification Using MRI Images},
  author={Grover, Hardik and others},
  year={2026}
}
