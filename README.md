# Osteosarcoma Tumor Classification using GoogLeNet and Grad-CAM (Explainable AI)

## Project Overview  
Osteosarcoma is a rare but highly aggressive bone cancer, most common in adolescents and young adults. Due to the complexity and diversity of tumor tissue structures, **early and accurate classification** of histopathological images is crucial for proper diagnosis and treatment.  

This project applies **GoogLeNet**, a deep convolutional neural network architecture with *inception modules*, to classify osteosarcoma tissue images.  
To make the model’s decisions interpretable, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is integrated to visualize the image regions influencing predictions — enabling an **explainable AI approach** for medical image analysis.

---

## Objectives  
- Develop a deep learning model using GoogLeNet for osteosarcoma classification.  
- Apply **Grad-CAM** for visual interpretability of model predictions.  
- Demonstrate the role of explainable AI in supporting digital pathology.  

---

## Dataset  
**Source:** PKG – Osteosarcoma Tumor Assessment Dataset (Training Dataset-2)  

This dataset contains histopathological images of osteosarcoma tissue.  

### Dataset Organization  
The original dataset did **not** contain separate folders for tumor types.  
A Python preprocessing step was performed to organize the images into three classes:  
 1) Viable Tumor
 2) Non-Viable Tumor
 3) Non Tumor

---

## Data Preparation  

### Preprocessing Steps:
- **Resizing:** All images were resized to **224×224 pixels** (compatible with GoogLeNet input).  
- **Normalization:** Adjusts pixel intensity range for faster convergence.  
- **Label Conversion:** Converts class names to integer labels.  

---

## Model Architecture – GoogLeNet  

**Framework:** PyTorch  
**Model Base:** Pretrained GoogLeNet (from `torchvision.models`)  

### Key Features:
- **Inception Modules:** Capture features at multiple spatial scales.  
- **1×1 Convolutions:** Reduce dimensionality, improving efficiency.  
- **Auxiliary Classifiers:** Mitigate vanishing gradients during training.  
- **Global Average Pooling:** Reduces parameters before final FC layer.  

The final fully-connected layer was modified to match **3 output classes** (*viable tumor*, *non-viable tumor*, *non-tumor*).

---

## Training Details  

- **Optimizer:** Adam (`lr = 0.0001`)  
- **Loss Function:** Cross-Entropy Loss  
- **Epochs:** 10  
- **Batch Size:** 16  
- **Hardware:** GPU (CUDA)  

### Training Metrics  
| Dataset  | Accuracy | Loss |
|-----------|-----------|------|
| Training  | 86.59%    | 1.83 |
| Testing   | 85.71%    | 1.83 |

---

## Explainability – Grad-CAM  

Grad-CAM was implemented to visualize **important regions** that influenced the model’s classification decisions.  
These heatmaps help validate that the network focuses on tumor regions rather than irrelevant background areas.  

---

## Results & Discussion  
- Achieved **85.71% accuracy** on the test dataset.  
- Grad-CAM visualizations confirmed that the model’s focus aligned with critical tumor regions.  
- The fine-tuned GoogLeNet model shows strong potential for digital pathology and could assist pathologists by providing **fast, consistent, and explainable diagnostic insights**.

---

## Tools & Frameworks  

| Category | Tools Used |
|-----------|-------------|
| Programming | Python 3.10+ |
| Deep Learning | PyTorch, Torchvision |
| Data Processing | NumPy, OpenCV, PIL, os |
| Environment | Jupyter Notebook |

---

## Acknowledgements

Special thanks to the open-source community for providing the datasets and tools necessary for this project.
