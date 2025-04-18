# 🔬 Skin Cancer Classification Web App

This is an end-to-end web app for classifying **7 types of skin cancer** using a deep learning model (ResNet50), with explainability via **Grad-CAM visualization**.

🌐 **Live Demo**: [Click to Try it on Streamlit](https://skincancer-detection.streamlit.app/)  
📁 **Dataset Used**: [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## 🧠 Features

- Upload and classify dermoscopic images
- 7-class classification output
- Grad-CAM visualization for interpretability
- Download button for Grad-CAM result
- Clean, interactive UI built with Streamlit
- Sidebar with project info and class descriptions

---

## 📸 Skin Cancer Classes

| Label | Name                              |
|-------|-----------------------------------|
| 0     | Actinic Keratoses (AKIEC)         |
| 1     | Basal Cell Carcinoma (BCC)        |
| 2     | Benign Keratosis-Lesions (BKL)    |
| 3     | Dermatofibroma (DF)               |
| 4     | Melanoma (MEL)                    |
| 5     | Melanocytic Nevi (NV)             |
| 6     | Vascular Lesions (VASC)           |

---

## 🧰 Tech Stack

- Python
- PyTorch
- OpenCV, PIL, NumPy, Matplotlib
- Streamlit (for the frontend)
- Git LFS (for large model file upload)

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/shawray-sinha/skin-cancer-classifier-app.git
cd skin-cancer-classifier-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
