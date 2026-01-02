# MNIST-image-classification-using-CNN

# MNIST Image Classification using CNN (TensorFlow)

This project demonstrates a **simple image classification application** using a **Convolutional Neural Network (CNN)** built with **TensorFlow**.  
The model is trained on the **MNIST handwritten digits dataset** to classify digits from **0 to 9**.

---

## 📌 Project Features
- Uses the MNIST handwritten digit dataset
- Image normalization and reshaping
- CNN-based image classification model
- Predicts the class of a sample image
- Displays prediction result visually

---

## 🛠 Tools & Technologies
- Python
- NumPy
- TensorFlow
- Matplotlib
- OpenCV (optional)

---
## Simple Explanation

- This program works with images of handwritten numbers.
- It uses many images to learn how each number looks.
- Before training, the images are cleaned and resized so the computer can understand them.
- The program trains a model to recognize numbers from 0 to 9.
- After training, it checks how well the model works.
- Finally, it selects one image and displays both the predicted number and the actual number.


## ⚠️ Python Version Requirement
TensorFlow is compatible with **Python 3.10 (64-bit)**.

✅ **Required Python version:**  
## 🧪 How to Create a Virtual Environment (venv)

### Step 1: Create venv
```bash
py -3.10 -m venv venv

**Activate venv (Windows)**
venv\Scripts\activate

Step 1: Install from requirements.txt
python image_classification.py

Step 2: Run the project
python main.py
