# Brain-Tumor-Detector

Building a detection model using a **Convolutional Neural Network (CNN)** in **TensorFlow & Keras**.

This project focuses on classifying **MRI brain images** as *tumorous* or *non-tumorous*. The dataset is sourced from Kaggle and serves as a compact yet effective dataset for practicing medical image classification using deep learning.

---

## 🧠 About the Data

The dataset contains **two folders** — `yes` and `no` — which include a total of **253 brain MRI images**:

* `yes/` → 155 images containing **tumors**
* `no/` → 98 images that are **non-tumorous**

**Dataset Source:** [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets)

---

## 🚀 Getting Started

> ⚠️ Note: Sometimes, IPython notebooks do not render properly on GitHub. If that happens, view them using [nbviewer](https://nbviewer.jupyter.org/).

### Prerequisites

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python
```

Clone the repository and navigate into it:

```bash
git clone https://github.com/yourusername/Brain-Tumor-Detector.git
cd Brain-Tumor-Detector
```

---

## 📈 Data Augmentation

**Why Data Augmentation?**

Because the dataset is small and imbalanced (155 positive vs 98 negative), augmentation was necessary to:

* Increase data diversity and prevent overfitting.
* Address the imbalance between tumor and non-tumor samples.

**Before augmentation:**

* Positive (tumor): 155 images
* Negative (no tumor): 98 images
* **Total:** 253 images

**After augmentation:**

* Positive (tumor): 1085 images
* Negative (no tumor): 980 images
* **Total:** 2065 images

All augmented images (including the original ones) are stored in a folder named `augmented_data/`.

---

## ⚙️ Data Preprocessing

Each MRI image undergoes the following preprocessing steps:

1. **Cropping:** Retain only the brain region (removing unnecessary background).
2. **Resizing:** Uniformly resize all images to `(240, 240, 3)`.
3. **Normalization:** Scale pixel values to the `[0, 1]` range.
4. **Splitting:** Divide the dataset into:

   * 70% → Training
   * 15% → Validation
   * 15% → Testing

---

## 🧩 Neural Network Architecture

The CNN architecture is designed to be simple yet efficient for small datasets.

### Architecture Overview

1. Input layer: (240, 240, 3)
2. Zero Padding layer (2, 2)
3. Conv2D → 32 filters, (7×7 kernel), stride = 1
4. Batch Normalization
5. ReLU activation
6. MaxPooling (4×4)
7. Another MaxPooling (4×4)
8. Flatten layer
9. Dense → 1 neuron with **sigmoid** activation (binary output)

### Why this architecture?

Initially, transfer learning with **ResNet50** and **VGG16** was attempted, but due to limited data and computational constraints (Intel i7 CPU, 8 GB RAM), those models overfit. A custom lightweight CNN was built and trained from scratch — yielding stable and strong performance.

---

## 🏋️ Training the Model

The model was trained for **24 epochs**.

Training plots:

* **Loss Curve:** Demonstrates steady convergence.
* **Accuracy Curve:** Shows validation accuracy peaking at epoch 23.

**Best validation accuracy:** achieved at epoch 23.

---

## 🎯 Results

The model achieved:

* **Accuracy:** 88.7% on the test set
* **F1 Score:** 0.88 on the test set

| Metric   | Validation Set | Test Set |
| -------- | -------------- | -------- |
| Accuracy | 91%            | 89%      |
| F1 Score | 0.91           | 0.88     |

Considering the dataset size and balance, these results are highly promising.

---

## 🧾 Files and Structure

```
Brain-Tumor-Detector/
├── data/
│   ├── yes/
│   ├── no/
│   ├── augmented_data/
├── notebooks/
│   ├── Data_Augmentation.ipynb
│   ├── Model_Training.ipynb
├── models/
│   ├── cnn-parameters-improvement-23-0.91.model
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
├── README.md
```

---

## 💾 Loading the Trained Model

You can restore and reuse the trained model as follows:

```python
from tensorflow.keras.models import load_model
model = load_model('models/cnn-parameters-improvement-23-0.91.model')
```

Then use it for inference:

```python
prediction = model.predict(image_array)
```

---

## 🧠 Key Learnings

* Data augmentation can drastically improve small medical datasets.
* Simple CNNs can outperform heavy transfer models in low-resource environments.
* Proper preprocessing (cropping, normalization) is crucial for medical imaging.

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!

To contribute:

1. Fork the repository.
2. Create a new branch (`feature-name`).
3. Commit your changes.
4. Open a pull request.

---

## 🙏 Acknowledgements

* Kaggle for providing the dataset.
* TensorFlow & Keras for easy deep learning workflows.

**Author:** Hardik Grover
**Email:** [reach.hardikgrover@gmail.com](mailto:reach.hardikgrover@gmail.com)

> *Thank you for exploring Brain Tumor Detector!* 🧠✨
