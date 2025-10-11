
import streamlit as st
import cv2
import numpy as np
from datetime import datetime

from detector import detect_face
from embedder import get_embedding
from recognizer import save_user_embedding, knn_match, svm_match

# Page configuration
st.set_page_config(page_title="Facial Scanner", layout="centered")
st.title("🔐 Facial Recognition Unlock System")

# --- Utility functions ---

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def auto_brighten(img):
    """Automatically enhance brightness using histogram equalization."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    brightened = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return brightened

# --- Registration Process ---
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
            # Optional: resize face to standard size (improves embedding consistency)
            face_resized = cv2.resize(face, (160, 160))
            embedding = get_embedding(face_resized)

            if embedding is not None:
                embeddings.append(embedding)
                frame_count += 1
                st.image(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB), caption=f"Captured Face {frame_count}")
        else:
            st.warning("No face detected. Try again...")

    cap.release()

    if embeddings:
        save_user_embedding(name, embeddings)
        st.success(f"✅ {name} registered successfully!")
    else:
        st.error("⚠️ No face captured. Please retry.")


# --- Unlock Logic ---
def unlock(algorithm):
    st.header("🔓 Face Unlock")

    if st.button("Scan Face"):
        st.info("📷 Scanning for your face... please stay still for 3 seconds.")
        cap = cv2.VideoCapture(0)
        found_face = False
        best_face = None
        embedding = None

        for attempt in range(10):  # Capture multiple frames
            ret, frame = cap.read()
            if not ret:
                continue

            frame = auto_brighten(frame)
            face = detect_face(frame)

            # Skip small or unclear detections
            if face is not None and face.shape[0] > 60 and face.shape[1] > 60:
                found_face = True
                best_face = face
                face_resized = cv2.resize(face, (160, 160))
                embedding = get_embedding(face_resized)
                break

        cap.release()

        if not found_face:
            st.warning("⚠️ No face detected. Please ensure proper lighting and face position.")
            return

        if embedding is None:
            st.error("❌ Could not extract face embedding. Try again.")
            return

        st.image(cv2.cvtColor(best_face, cv2.COLOR_BGR2RGB), caption="Scanned Face")

        if algorithm == "k-NN":
            matched_user, score = knn_match(embedding)
            label = "Similarity"
        else:
            matched_user, score = svm_match(embedding)
            label = "Probability"

        if matched_user:
            st.success(f"✅ Access Granted: Welcome, {matched_user}! ({label}: {score:.2f})")
        else:
            st.error(f"🚫 Access Denied: Face not recognized. ({label}: {score:.2f})")

# # --- Sidebar ---
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
    algorithm = st.sidebar.radio("Select Algorithm", ["k-NN", "SVM"])
    unlock(algorithm)
