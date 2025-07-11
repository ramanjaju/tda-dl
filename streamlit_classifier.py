# streamlit_app.py

import streamlit as st
from PIL import Image
from torchvision import transforms
import torch

from image_classification import load_alexnet_model  # 👈 importing from File 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model only once
@st.cache_resource
def get_model():
    return load_alexnet_model()

model, class_names = get_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🧠 Brainrot Meme Classifier (AlexNet)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    st.success(f"✅ Predicted Class: **{prediction}**")
