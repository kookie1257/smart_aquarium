import cv2
import json
import random

# Load fish data
with open("fish_data.json") as f:
    fish_info = json.load(f)

# Open video
cap = cv2.VideoCapture("fish_video.mov")


if not cap.isOpened():
    print("❌ Error: Video not found or cannot be opened.")
else:
    print("✅ Video loaded successfully.")


# Motion detector
fgbg = cv2.createBackgroundSubtractorMOG2()
fish_boxes = []
selected_fish = None

# Randomly assign fish names
fish_names = list(fish_info.keys())
fish_id_map = {}

# Click event
def on_click(event, x, y, flags, param):
    global selected_fish
    if event == cv2.EVENT_LBUTTONDOWN:
        for fish_id, (x1, y1, x2, y2) in fish_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_fish = fish_id
                print(f"\n🐟 You clicked on: {fish_id}")
                print(f"Species: {fish_info[fish_id]['species']}")
                print(f"Fact: {fish_info[fish_id]['fact']}")
                print("🎣 Catching...\n")
                break

cv2.namedWindow("Fish Tank")
cv2.setMouseCallback("Fish Tank", on_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = fgbg.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fish_boxes = []
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            fish_name = fish_id_map.get(count, random.choice(fish_names))
            fish_id_map[count] = fish_name
            fish_boxes.append((fish_name, x, y, x+w, y+h))
            color = (0, 255, 0) if fish_name == selected_fish else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, fish_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            count += 1

    if selected_fish:
        cv2.putText(frame, f"Catching {selected_fish}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    cv2.imshow("Fish Tank", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
