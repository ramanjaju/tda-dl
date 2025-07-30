import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime

from detector import detect_face
from embedder import get_embedding
from recognizer import save_user_embedding, knn_match

# Page configuration
st.set_page_config(page_title="Facial Scanner", layout="centered")
st.title("🔐 Facial Recognition Unlock System")

# Camera frame capture
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# Brightness correction
def auto_brighten(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    brightened = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return brightened

# Registration process
def register(name):
    st.info("📸 Please look into the camera. Capturing multiple frames...")

    cap = cv2.VideoCapture(0)
    embeddings = []
    frame_count = 0
    max_frames = 10

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access camera.")
            break

        frame = auto_brighten(frame)
        face = detect_face(frame)

        if face is not None:
            embedding = get_embedding(face)
            if embedding is not None:
                embeddings.append(embedding)
                frame_count += 1
                st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f"Captured Face {frame_count}")
        else:
            st.warning("No face detected. Try again.")

    cap.release()

    if embeddings:
        save_user_embedding(name, embeddings)
        st.success(f"✅ {name} registered successfully!")
    else:
        st.error("No face captured. Please retry.")

# Unlock logic
def unlock():
    st.header("🔓 Face Unlock")
    if st.button("Scan Face"):
        frame = capture_frame()
        if frame is None:
            st.error("Camera error.")
            return

        frame = auto_brighten(frame)
        face = detect_face(frame)
        if face is not None:
            embedding = get_embedding(face)
            matched_user = knn_match(embedding, k=1, threshold=0.7)
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Scanned Face")

            if matched_user:
                st.success(f"✅ Access Granted: Welcome, {matched_user}!")
            else:
                st.error("❌ Access Denied: Face not recognized.")
        else:
            st.warning("No face detected.")

# Sidebar mode selection
mode = st.sidebar.radio("Select Mode", ["Register", "Unlock"])

if mode == "Register":
    st.subheader("👤 Register Face")
    name = st.text_input("Enter your name")
    if st.button("Scan Face"):
        if name.strip() == "":
            st.warning("Please enter a name before scanning.")
        else:
            register(name)
else:
    unlock()
