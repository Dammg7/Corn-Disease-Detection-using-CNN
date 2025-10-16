# 🌽 Corn Leaf Disease Classifier (CNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-brightgreen.svg)

A deep learning project that classifies corn leaf images into four categories using a **Convolutional Neural Network (CNN)**. The model is trained on a custom image dataset and achieves **96% validation accuracy**.

A **Streamlit Web App** is included for easy user interaction and real-time prediction.

---

## ✅ Features

* 🧠 **Train CNN model from scratch:** A custom, sequential CNN is implemented for image classification.
* 📊 **High Accuracy:** Achieves **96% validation accuracy** on the test dataset.
* 🔍 **Predict Disease:** Quickly predict the disease class from an uploaded image.
* 🌐 **Streamlit Web App:** Provides a clean, interactive interface for predictions.
* 💾 **Save and Reuse Model:** The trained weights are saved as `net.keras`.
* 📂 **Organized Dataset Structure:** Supports a standard image folder structure for easy data management.
* ✔ **Environment Versatility:** Code runs smoothly in Google Colab or locally.

---

## 🏷 Disease Classes

The model is trained to classify images into four distinct categories:

1.  **Healthy**
2.  **Rust**
3.  **Blight**
4.  **Leaf Spot Gray**

---

## 📂 Dataset Structure

The training notebooks expect the data to be organized in the following directory structure (or equivalent paths configured for Google Drive):

## 🛠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib**
* **PIL (Pillow)**
* **Scikit-learn**
* **Streamlit**
* **Google Colab** (for training environment)

---

## 🚀 How to Train the Model

The training process is contained within the `corn training.ipynb` notebook.

| Metric | Result |
| :--- | :--- |
| **Validation Accuracy** | **0.96 (96%)** |

### Steps:

1.  **Open Notebook:** Load the **`corn training.ipynb`** file in Google Colab.
2.  **Mount Drive:** Ensure your Google Drive is mounted and the dataset paths are correct (e.g., `/content/drive/MyDrive/training/...`).
3.  **Run Cells:** Execute all cells sequentially. The final cell will train the model for 50 epochs and save the trained weights.

```python
model.save("net.keras")
```
## 
🌐 Streamlit Web App

The model is deployed via a lightweight web application for user-friendly inference.

app.py
Create a file named app.py with the following content:
Run the App
Ensure you have the net.keras file in the same directory as app.py, then run this command in your terminal:
streamlit run app.py

## Future Improvements

Data Augmentation: Implement real-time data augmentation (e.g., using ImageDataGenerator) to improve generalization and robustness.

Transfer Learning: Experiment with pre-trained models like MobileNet or EfficientNet for higher accuracy and faster convergence.

API Deployment: Wrap the model in a REST API (e.g., using Flask or FastAPI) for scalable deployment.

Mobile App Version: Develop a mobile application (e.g., with TensorFlow Lite) for field-based diagnosis.

Cloud Hosting: Deploy the Streamlit application to a cloud service (e.g., Streamlit Cloud, Heroku).
