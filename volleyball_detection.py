
import cv2
import numpy as np

video_path = '/Users/raman/Downloads/volleyball_match.mp4'

# ======================
# 🎯 PHASE 1: Show Original Video
# ======================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

print("▶️ Playing Original Video...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of original video.")
        break

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Original Video", frame)

    # ⏱️ Show at 30 FPS approx. Quit with 'q'
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ======================
# 🎯 PHASE 2: Motion Mask Only
# ======================
cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

print("▶️ Playing Motion Mask...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of motion mask video.")
        break

    frame = cv2.resize(frame, (640, 480))
    fgmask = fgbg.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Motion Mask", fgmask)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ======================
# 🎯 PHASE 3: Enhanced Ball Detection (Color + Motion)
# ======================
cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

print("🎯 Starting Enhanced Volleyball Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of detection.")
        break

    frame = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 💡 Background Subtraction (motion mask)
    fgmask = fgbg.apply(blurred)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 💡 Convert frame to HSV for color detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 🎯 Yellow Range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 🎯 Red Range (split into 2 ranges)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # ✅ Combine motion with each color separately
    moving_yellow = cv2.bitwise_and(yellow_mask, fgmask)
    moving_red = cv2.bitwise_and(red_mask, fgmask)

    # ✅ Final mask = moving + red/yellow
    combined_mask = cv2.bitwise_or(moving_yellow, moving_red)

    # 🔍 Find contours in combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 150:  # 🎯 size condition
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5) 

            if 6.5 < radius < 10 and circularity > 0.75:  # 🎯 radius and shape condition
                ball_center = (int(x), int(y))
                cv2.circle(frame, ball_center, int(radius), (0, 255, 0), 2)
                break  # Detect only one ball at a time

    # 🖼️ Display outputs
    cv2.imshow("Detected Volleyball", frame)
    # cv2.imshow("Combined Mask", combined_mask)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()