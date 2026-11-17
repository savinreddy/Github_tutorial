"""
Hand Gesture Air-Writing & App Switcher (mediapipe Tasks API version)
-----------------------------------------------------------------------
Uses your camera feed (your phone acting as a webcam) to:
  1. Pinch (thumb+index together) = "air write" - draws a trail on screen
  2. Closed fist = clear the note
  3. Open hand + fast horizontal swipe = switch apps/windows on your COMPUTER
  4. Manual/legend shown at the bottom of the screen at all times

This version works with newer mediapipe releases (1.x) that use the
"Tasks" API instead of the older mp.solutions.hands interface.

INSTALL (run once):
    pip install opencv-python mediapipe pyautogui numpy

RUN:
    python gesture_control.py

The FIRST time you run it, it will automatically download a small
hand-tracking model file (a few MB) - this needs an internet connection
once. After that it's saved locally and works offline.

Press 'q' to quit.
"""

import cv2
import numpy as np
import time
import platform
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
CAMERA_INDEX = 0
PINCH_THRESHOLD = 40
SWIPE_MIN_DISTANCE = 150
SWIPE_MAX_TIME = 0.6
DRAW_COLOR = (0, 255, 255)
DRAW_THICKNESS = 4

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

IS_MAC = platform.system() == "Darwin"

# ----------------------------------------------------------------------
# Download the hand-tracking model if we don't have it yet
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("First run: downloading hand-tracking model (a few MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
canvas = None
prev_point = None
wrist_history = []
last_action_time = 0
last_action_text = ""


def fingers_up(lm):
    """lm is a list of 21 landmarks, each with .x and .y. Returns 5 booleans."""
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = []
    # Thumb: sideways comparison
    fingers.append(
        lm[tips[0]].x < lm[pips[0]].x if lm[tips[0]].x < lm[0].x else lm[tips[0]].x > lm[pips[0]].x
    )
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def switch_app(direction):
    import pyautogui
    if IS_MAC:
        if direction == "right":
            pyautogui.keyDown("command")
            pyautogui.press("tab")
            pyautogui.keyUp("command")
        else:
            pyautogui.keyDown("command")
            pyautogui.keyDown("shift")
            pyautogui.press("tab")
            pyautogui.keyUp("shift")
            pyautogui.keyUp("command")
    else:
        pyautogui.keyDown("alt")
        pyautogui.press("tab")
        pyautogui.keyUp("alt")


def draw_manual(frame):
    h, w, _ = frame.shape
    bar_height = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    lines = [
        "MANUAL:  Pinch thumb+index = write/draw   |   Open hand + fast swipe = switch app",
        "Closed fist = clear note   |   'q' key = quit",
    ]
    y = h - bar_height + 30
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def main():
    global canvas, prev_point, last_action_time, last_action_text

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}. Try changing CAMERA_INDEX at the top of the script.")
        return

    print("Camera opened. Press 'q' in the video window to quit.")
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]  # list of 21 landmarks for the first hand

            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
            wrist = (int(lm[0].x * w), int(lm[0].y * h))

            up = fingers_up(lm)
            pinch_dist = distance(index_tip, thumb_tip)

            # Draw simple markers for key points (no full skeleton in this API)
            cv2.circle(frame, index_tip, 6, (0, 255, 0), -1)
            cv2.circle(frame, thumb_tip, 6, (255, 0, 0), -1)

            # Gesture 1: Pinch = draw
            if pinch_dist < PINCH_THRESHOLD:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, index_tip, DRAW_COLOR, DRAW_THICKNESS)
                prev_point = index_tip
                cv2.circle(frame, index_tip, 8, DRAW_COLOR, -1)
            else:
                prev_point = None

            # Gesture 2: Fist = clear
            if not any(up):
                canvas = np.zeros_like(frame)
                last_action_text = "Note cleared"
                last_action_time = time.time()

            # Gesture 3: Open hand + swipe = switch app
            if all(up):
                now = time.time()
                wrist_history.append((wrist[0], now))
                wrist_history[:] = [pt for pt in wrist_history if now - pt[1] < SWIPE_MAX_TIME]

                if len(wrist_history) >= 2:
                    dx = wrist_history[-1][0] - wrist_history[0][0]
                    dt = wrist_history[-1][1] - wrist_history[0][1]
                    if abs(dx) > SWIPE_MIN_DISTANCE and dt < SWIPE_MAX_TIME:
                        if now - last_action_time > 1.0:
                            direction = "right" if dx > 0 else "left"
                            switch_app(direction)
                            last_action_text = f"Switched app ({direction} swipe)"
                            last_action_time = now
                            wrist_history.clear()
            else:
                wrist_history.clear()

        # Merge drawing canvas onto frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame_bg, canvas)

        if last_action_text and time.time() - last_action_time < 1.5:
            cv2.putText(frame, last_action_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        draw_manual(frame)
        cv2.imshow("Gesture Control - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()