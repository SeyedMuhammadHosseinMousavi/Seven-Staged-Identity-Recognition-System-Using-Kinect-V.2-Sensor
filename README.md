# Seven Staged Identity Recognition System Using Kinect V2 Sensor

This repository presents the **Seven Staged Identity Recognition System Using Kinect V2 Sensor**, which implements a robust, multi-stage identity recognition framework. The system leverages color and depth image processing, signal processing, machine learning, and evolutionary algorithms for high-accuracy recognition in various security applications.

---

## 📜 **Motivation**
Most existing identity recognition systems operate in only three stages, which may lead to security vulnerabilities. This paper proposes a more robust seven-stage system to achieve nearly **99% accuracy** by combining cutting-edge technologies with a Kinect V2 sensor and additional hardware enhancements like macro lenses.

### Link to the paper:
- https://ieeexplore.ieee.org/abstract/document/9756435
- DOI: https://doi.org/10.1109/CFIS54774.2022.9756435
### Please cite:
## 📝 **Citation**
- Mousavi, Seyed Muhammad Hossein, and Atiye Ilanloo. "Seven staged identity recognition system using Kinect V. 2 sensor." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.
---

## 🛠️ **System Architecture**
The proposed system consists of **seven recognition and estimation stages**:
1. **Face Recognition**
2. **Voice Recognition**
3. **Fingerprint Recognition**
4. **Iris Recognition**
5. **Gesture Recognition**
6. **Sex Detection**
7. **Age Estimation**

Each stage utilizes a combination of **machine learning algorithms**, **fuzzy systems**, and advanced image and signal processing techniques to improve accuracy and reliability.
![Seven Staged Identity Recognition System Using Kinect V 2 Sensor_page-0003](https://github.com/user-attachments/assets/5a5ba7dc-7244-4de8-8a67-88589460e9ac)

---

## 🚀 **Methodology**

### **1. Data Acquisition**
The Kinect V2 sensor captures both **color** and **depth data**, enabling robust recognition even in challenging environments, such as pure darkness (via infrared support).

### **2. Preprocessing**
- **Image Data**:
  - **Face and Iris Recognition**: Background removal using the Viola-Jones algorithm, edge detection with Canny, and morphological operations.
  - **Gesture Recognition**: Body segmentation from depth data.
  - **Fingerprint Recognition**: Captured using the color sensor and enhanced by a macro lens.
- **Audio Data**:
  - Noise removal and normalization using a median filter.
  - Feature extraction via **Mel-Frequency Cepstral Coefficients (MFCCs)**.

### **3. Feature Extraction**
- **HOG (Histogram of Oriented Gradients)** for images.
- **MFCCs** for voice data.
- **Local Phase Quantization (LPQ)** and **Speeded-Up Robust Features (SURF)** for age and sex estimation.
- **Histogram of Oriented Gradients (HOG)** for fingerprint and iris recognition.

### **4. Dimensionality Reduction**
- Features are refined using **Lasso regularization**, which removes outliers and accelerates classification.

### **5. Classification**
- **Support Vector Machine (SVM)** for face, voice, iris, and fingerprint classification.
- **Differential Evolution Adaptive Neuro-Fuzzy Inference System (DE-ANFIS)** for gesture and sex classification.

### **6. Decision Integration**
- After all stages are processed, the system checks for consistency in the recognition results.
- If all stages match, the identity is confirmed and the system halts with success; otherwise, it loops back for re-identification.
![2](https://github.com/user-attachments/assets/63de1523-1ba8-49d7-8d42-dde8b65b5c94)

---

## 📊 **Validation and Results**

### **Dataset**
The system was validated on a dataset consisting of:
- **6 Subjects**
- Data collected for **face**, **voice**, **fingerprint**, **iris**, and **gesture** recognition.

### **Performance Metrics**
The system achieved the following average accuracy across all stages:
- **CNN**: 99.41%
- **SVM**: 98.50%
- **DE-ANFIS**: 97.91%

#### **Confusion Matrices**:
Detailed confusion matrices for each classification stage demonstrate the system's robustness and minimal misclassifications.

---

## ⚙️ **System Requirements**
- **Hardware**:
  - Kinect V2 Sensor
  - Macro lens for improved iris and fingerprint recognition.
- **Software**:
  - Python 3.x
  - OpenCV
  - SciPy, NumPy
  - Scikit-learn
  - Librosa for audio processing

---

## 🔍 **Applications**
The system is designed for:
- **Security and Surveillance**: High-security zones, museums, or bank vaults.
- **Criminology**: Law enforcement and forensic applications.
- **Medical Research**: Containment of sensitive materials or viruses.
- **Defense and Intelligence**: Confidential document handling.

---

## 🧩 **Future Work**
- Expand the database to include more subjects.
- Incorporate advanced algorithms like **3D face recognition**.
- Experiment with **Scale-Invariant Feature Transform (SIFT)** and **Local Binary Patterns (LBP)** for improved feature extraction.
- Integrate additional hardware for faster real-time processing.

---



![Seven Staged Identity Recognition System](https://user-images.githubusercontent.com/11339420/166145010-d6a4abba-1d2e-4cbc-a7e7-7a8ac92f0870.JPG)
