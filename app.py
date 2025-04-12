import streamlit as st
from PIL import Image
import torch
import os
import numpy as np
from utils.model_utils import load_model, predict_image
from utils.gradcam_utils import generate_gradcam


st.set_page_config(page_title="Skin Cancer Classifier", layout="wide")


st.sidebar.title("Project Info")
st.sidebar.subheader("🔬 Skin Cancer Classification")
st.sidebar.write("""
This project uses a deep learning model (ResNet50) to classify dermoscopic images of skin lesions into one of 7 categories. 
The model is fine-tuned and optimized for accurate prediction using state-of-the-art techniques like Grad-CAM for visualization.
""")
st.sidebar.write("### Class Descriptions")
st.sidebar.write("""
1. **Actinic keratoses (akiec)**: A pre-cancerous area of thick, scaly, or crusty skin.
2. **Basal cell carcinoma (bcc)**: A type of skin cancer that begins in the basal cells.
3. **Dermatofibroma (df)**: A benign skin tumor.
4. **Melanoma (mel)**: A serious type of skin cancer that begins in melanocytes.
5. **Nevus (nv)**: A mole or birthmark.
6. **Pigmented benign keratosis (bkl)**: A non-cancerous growth on the skin.
7. **Vascular lesion (vasc)**: Lesions caused by abnormal blood vessels.

For detailed explanations, consult the dermatology reference materials.
""")
st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/skin-cancer-classifier)")


st.title(" Skin Cancer Classifier")
st.write("Upload a dermoscopic image to classify it into one of the 7 skin cancer types.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

   
    with st.spinner("Loading model and making prediction..."):
        model = load_model("model/resnet_model.pth")
        prediction, class_name = predict_image(model, image)

    st.success(f"**Predicted Class:** {class_name}")

    
    if st.button("Explain with Grad-CAM"):
        st.subheader(" Grad-CAM Visualization")
        heatmap = generate_gradcam(model, image, prediction)

        
        heatmap_image = Image.fromarray(heatmap)

        st.image(heatmap_image, caption="Grad-CAM Heatmap", use_column_width=True)

      
        heatmap_image.save("gradcam_output.jpg")

        
        st.download_button(
            label="Download Grad-CAM Result",
            data=open("gradcam_output.jpg", "rb"),
            file_name="gradcam_output.jpg",
            mime="image/jpeg"
        )
