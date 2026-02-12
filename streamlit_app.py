import streamlit as st
import cv2
import json
import numpy as np
import random
import time
from PIL import Image

# Page Config
st.set_page_config(page_title="Smart Aquarium AI HUD", page_icon="🐠", layout="wide")

# Custom CSS for high-tech look
st.markdown("""
<style>
    .stApp {
        background-color: #050a0f;
        color: #00ffff;
    }
    .main-title {
        font-family: 'Courier New', Courier, monospace;
        color: #00ffff;
        text-align: center;
        text-shadow: 0 0 10px #00ffff;
        border-bottom: 2px solid #00ffff;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🐠 SMART AQUARIUM AI - BIO-SCANNER</h1>", unsafe_allow_html=True)

# Load fish info
try:
    with open("fish_data.json") as f:
        fish_info = json.load(f)
except:
    fish_info = {
        "Fish1": {"species": "Clownfish", "fact": "Loves hiding in anemones"},
        "Fish2": {"species": "Goldfish", "fact": "Recognizes its owner"}
    }

# Video Setup
video_path = "fish_video.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("❌ Could not load aquarium data.")
    st.stop()

# Helper Functions
def draw_hud_box(img, pt1, pt2, color, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    l = min(w, h) // 4
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

# Sidebar UI
st.sidebar.title("🛠 Control Center")
selected_target = st.sidebar.selectbox("🎯 Select Scan Target", ["None"] + list(fish_info.keys()))

st.sidebar.markdown("---")
st.sidebar.info("This AI system uses Computer Vision to detect and track aquatic life in real-time.")

# Display Info Panel in Sidebar if selected
if selected_target != "None":
    info = fish_info[selected_target]
    st.sidebar.subheader(f"📊 Analyzing: {selected_target}")
    st.sidebar.markdown(f"**Species:** {info['species']}")
    st.sidebar.markdown(f"**Bio-Fact:** {info['fact']}")
    st.sidebar.progress(100)
    st.sidebar.caption("Scan Status: Complete")

# Image Placeholder
frame_placeholder = st.empty()

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
fish_id_map = {}

# Stream processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (800, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = fgbg.apply(cv2.GaussianBlur(gray, (5, 5), 0))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_fish = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            fish_name = list(fish_info.keys())[i % len(fish_info)]
            current_fish.append((fish_name, x, y, w, h))
            
            is_selected = (fish_name == selected_target)
            color = (0, 255, 255) if is_selected else (0, 180, 255)
            
            draw_hud_box(frame, (x, y), (x + w, y + h), color, 2 if is_selected else 1)
            
            # Label
            cv2.putText(frame, f"ID: {fish_name}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Convert back to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    time.sleep(0.01) # Smoothness

cap.release()
