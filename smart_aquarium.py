import cv2
import json
import random
import numpy as np
import time

# Load fish info
try:
    with open("fish_data.json") as f:
        fish_info = json.load(f)
except Exception:
    fish_info = {
        "Fish1": {"species": "Clownfish", "fact": "Loves hiding in anemones"},
        "Fish2": {"species": "Goldfish", "fact": "Recognizes its owner"},
        "Fish3": {"species": "Mini Shark", "fact": "Fast swimmer, but friendly!"}
    }

fish_names = list(fish_info.keys())
fish_id_map = {}
selected_fish = None
info_panel_alpha = 0  # For fade-in animation
fish_boxes = []
pending_click = None
pulse_val = 0
pulse_dir = 1

# Create gradient overlay
def create_gradient_overlay(width, height, color1, color2, alpha=0.8):
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        ratio = i / height
        b = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        r = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        overlay[i, :] = [b, g, r]
    return overlay

# Draw modern HUD-style rounded rectangle
def draw_hud_box(img, pt1, pt2, color, thickness=2, corners_only=True):
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    l = min(w, h) // 4  # Length of corner lines
    
    if corners_only:
        # Top-left
        cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)
        
        # Add a faint full box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

# Draw rounded rectangle (filled or border)
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=15):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main bars
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    
    if thickness < 0:  # Fill
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

# Handle mouse click
click_visual = None # For click feedback
def on_click(event, x, y, flags, param):
    global pending_click, click_visual
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_click = (x, y)
        click_visual = [x, y, 15] # x, y, frames to show

# Simple Tracker
tracker_data = {} # {id: (x, y, name)}
next_id = 0

# Load video
cap = cv2.VideoCapture("fish_video.mov")
if not cap.isOpened():
    print("❌ Could not open video file.")
    exit()

# Setup
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
cv2.namedWindow("Smart Aquarium HUD", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Smart Aquarium HUD", on_click)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
        continue

    # UI Pulse animation
    pulse_val += pulse_dir * 5
    if pulse_val > 150 or pulse_val < 0: pulse_dir *= -1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = fgbg.apply(blurred)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            current_detections.append((x, y, w, h))

    # Match detections to existing tracked fish (Simple distance tracker)
    new_tracker_data = {}
    fish_boxes = []
    
    for (x, y, w, h) in current_detections:
        cx, cy = x + w//2, y + h//2
        best_match = None
        min_dist = 50 # Max distance to match
        
        for tid, (tx, ty, tname) in tracker_data.items():
            dist = np.sqrt((cx-tx)**2 + (cy-ty)**2)
            if dist < min_dist:
                min_dist = dist
                best_match = tid
        
        if best_match is not None:
            name = tracker_data[best_match][2]
            new_tracker_data[best_match] = (cx, cy, name)
            del tracker_data[best_match]
        else:
            name = random.choice(fish_names)
            new_tracker_data[next_id] = (cx, cy, name)
            next_id += 1
        
        fish_boxes.append((name, x, y, x + w, y + h))

    tracker_data = new_tracker_data
    overlay = frame.copy()

    # Handle Click logic
    if pending_click:
        cx, cy = pending_click
        found = False
        for name, x1, y1, x2, y2 in fish_boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                selected_fish = name
                info_panel_alpha = 0
                found = True
                break
        if not found:
            selected_fish = None
        pending_click = None

    for fish_name, x1, y1, x2, y2 in fish_boxes:
        is_selected = (fish_name == selected_fish)
        color = (0, 255, 255) if is_selected else (0, 180, 255)
        
        draw_hud_box(frame, (x1, y1), (x2, y2), color, 2 if is_selected else 1)
        
        if is_selected:
            alpha = (pulse_val / 255.0) * 0.5
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        label = f"SCAN: {fish_name}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        points = np.array([[x1, y1-5], [x1+label_w+20, y1-5], [x1+label_w+10, y1+15], [x1, y1+15]], np.int32)
        cv2.fillPoly(frame, [points], (20, 20, 20))
        cv2.putText(frame, label, (x1+5, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # Info Panel
    if selected_fish:
        if info_panel_alpha < 1.0: info_panel_alpha = min(1.0, info_panel_alpha + 0.1)
        info = fish_info.get(selected_fish, {"species": "Unknown", "fact": "Information gathering..."})
        
        px, py = 20, 80
        pw, ph = 450, 180
        sub_overlay = frame.copy()
        draw_rounded_rectangle(sub_overlay, (px, py), (px + pw, py + ph), (40, 20, 10), -1, 15)
        cv2.addWeighted(sub_overlay, info_panel_alpha * 0.8, frame, 1 - info_panel_alpha * 0.8, 0, frame)
        draw_rounded_rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2, 15)
        
        cv2.putText(frame, f"TARGET: {selected_fish}", (px+20, py+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Type: {info['species']}", (px+20, py+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        fact = info['fact']
        y_text = py + 120
        words = fact.split()
        line = ""
        for word in words:
            test = line + " " + word if line else word
            (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if tw < pw - 60: line = test
            else:
                cv2.putText(frame, line, (px+20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y_text += 25
                line = word
        cv2.putText(frame, line, (px+20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Click Feedback
    if click_visual:
        cv2.circle(frame, (click_visual[0], click_visual[1]), (20 - click_visual[2]), (0, 255, 255), 2)
        click_visual[2] -= 1
        if click_visual[2] <= 0: click_visual = None

    # Top Header
    header_h = 60
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (0, 0), (frame.shape[1], header_h), (30, 10, 5), -1)
    cv2.addWeighted(header_overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, header_h), (frame.shape[1], header_h), (0, 255, 255), 1)
    
    cv2.putText(frame, "SMART AQUARIUM AI - BIO-SCANNER v2.0", (30, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Time/Status
    status_text = f"STREAM: ACTIVE | FISH DETECTED: {len(fish_boxes)}"
    (sw, sh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, status_text, (frame.shape[1] - sw - 30, 38), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Smart Aquarium HUD", frame)
    
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):
        break
    elif key == 27: # ESC to deselect
        selected_fish = None

cap.release()
cv2.destroyAllWindows()

