# 🧠 Fake Image Detection (Deepfake Classifier)

A lightweight desktop application for detecting **AI-generated or manipulated images** using two deep learning models — a **custom-trained CNN** and the **Meso4 architecture**.  
Built with **TensorFlow** and a clean **PyQt5 GUI** interface.

---

## 🚀 Features
- Dual-model evaluation for more reliable results  
- Displays both individual model outputs and averaged decision  
- Confidence toggle to show/hide detailed probabilities  
- Simple, responsive PyQt5 interface with progress indicator  
- Works entirely offline once models are loaded

---

## 🧩 Models Used
- **New Model (`new_model.h5`)** — a fine-tuned CNN trained on fake vs. real image dataset.  
- **Meso4** — a proven architecture for deepfake detection, loaded via pre-trained weights (`Meso4_DF.h5`).

The final prediction is based on the **average confidence** from both models.

---

## 🖥️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-image-detector.git
cd fake-image-detector
