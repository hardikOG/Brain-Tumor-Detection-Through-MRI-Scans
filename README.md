# 🧠 ViT–MiT Brain Tumor Classification (IEEE Research Implementation)

This repository presents an implementation of a **hybrid Vision Transformer (ViT) and Mix Transformer (MiT) ensemble** for **multiclass brain tumor classification using MRI images**, developed as part of an **IEEE-style research study**.

The proposed framework performs **4-class classification**:
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

The model is evaluated on a **balanced public MRI dataset (7,200 images)** and achieves:

- **Test Accuracy:** 99.01%  
- **Macro F1-score:** 0.99  

The system integrates **transformer-based architectures, logit-level ensembling, progressive transfer learning, and attention-based explainability**.

---

## 🔍 Key Contributions

- **Hybrid Transformer Ensemble**
  - Vision Transformer (ViT) for **global contextual understanding**
  - Mix Transformer (MiT - SegFormer encoder) for **hierarchical multi-scale feature extraction**
  - Logit-level fusion with validation-optimized weighting (**α = 0.6**)

- **High-Performance Classification**
  - Robust 4-class classification on balanced MRI dataset
  - Strong generalization with independent test evaluation

- **Training Optimization**
  - Progressive backbone unfreezing
  - Cosine learning-rate scheduling with warmup
  - Automatic Mixed Precision (AMP)
  - Test-Time Augmentation (TTA)

- **Explainability**
  - Attention map visualization for interpretability
  - Class-wise evaluation and confusion matrix analysis

---

## 📊 Dataset

The model is trained and evaluated on a **public brain MRI dataset**:

- **Total Images:** 7,200  
- **Classes:** 4 (balanced)

| Split  | Images per Class | Total |
|--------|-----------------|-------|
| Train  | 1,400           | 5,600 |
| Test   | 400             | 1,600 |

**Source:**
- J. Cheng, *Brain Tumor Dataset*, figshare (2017)  
  DOI: https://doi.org/10.6084/m9.figshare.1512427.v5  

---

## 🧱 Model Architecture

### Vision Transformer (ViT)
- ViT-Base (16×16 patches)
- 224×224 input resolution
- 12 layers, 12 heads
- Embedding dimension: 768

Captures **global spatial relationships** in MRI scans.

---

### Mix Transformer (MiT - SegFormer Encoder)
- MiT-B2 backbone
- Multi-scale feature hierarchy (strides 4, 8, 16, 32)
- Channel sizes: 64, 128, 320, 512

Encodes **local + hierarchical tumor morphology**.

---

### Logit-Level Fusion

\[
z_{ens} = \alpha z_{ViT} + (1 - \alpha) z_{MiT}, \quad \alpha = 0.6
\]

- Fusion performed **before softmax**
- Weight optimized using **validation set**

---

## ⚙️ Training Strategy

- **Hardware:** NVIDIA T4 (Google Colab)
- **Optimizer:** AdamW  
- **Learning Rate:** 3 × 10⁻⁵  
- **Batch Size:** 32  
- **Epochs:** 50  

### Progressive Unfreezing
- Epochs 1–5: backbone frozen  
- Epochs 6–15: partial unfreezing  
- Epochs 16–50: full fine-tuning  

### Learning Rate Schedule
- Linear warmup → cosine decay  

### Regularization
- Label smoothing (ε = 0.1)  
- Automatic Mixed Precision (AMP)

---

## 🧪 Preprocessing & Augmentation

- Resize: 224 × 224  
- ImageNet normalization  
- Augmentations:
  - Horizontal flip  
  - Rotation (±10°)

### Test-Time Augmentation (TTA)
- Original + flipped inference  
- Logits averaged before softmax  

---

## 📈 Results

### Overall Performance

- **Accuracy:** 99.01%  
- **Macro F1-score:** 0.99  

### Per-Class Metrics

| Class       | Precision | Recall | F1  |
|------------|----------|--------|-----|
| Glioma     | 1.00     | 0.98   | 0.99 |
| Meningioma | 0.98     | 0.98   | 0.98 |
| No Tumor   | 1.00     | 1.00   | 1.00 |
| Pituitary  | 0.99     | 1.00   | 0.99 |

---

### Ablation Study

| Configuration | Accuracy (%) |
|--------------|-------------|
| ViT only | 98.86 |
| MiT only | 98.78 |
| Ensemble (α=0.6) | 98.93 |
| Ensemble + TTA | **99.01** |

---

## 🖼 Visualizations

Add your figures here:

```markdown
- Dataset distribution  
- Training curves  
- Model comparison  
- Confusion matrix  
- ROC curves  
- Attention maps  
