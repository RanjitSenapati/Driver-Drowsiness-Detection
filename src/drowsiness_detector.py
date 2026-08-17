"""
Driver Drowsiness Detection System
------------------------------------
Detects driver drowsiness in real time using a webcam by monitoring the
Eye Aspect Ratio (EAR) from facial landmarks, and triggers an audible
alarm when the driver's eyes stay closed for too long.

Author: Ranjit Senapati 
Tech stack: Python, OpenCV, dlib, NumPy, Pandas, Matplotlib
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
EAR_THRESHOLD = 0.25        # Eye Aspect Ratio below this = eyes considered closed
EAR_CONSEC_FRAMES = 20      # Number of consecutive frames eyes must stay closed
                             # before the alarm is triggered (~0.6-0.8 sec at 30fps)
YAWN_THRESHOLD = 20         # Mouth aspect distance threshold for yawn detection
SHAPE_PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "shape_predictor_68_face_landmarks.dat"
)
ALARM_SOUND_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "alarm.wav"
)
LOG_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "drowsiness_log.csv"
)

# Facial landmark indexes for left/right eye and mouth (68-point dlib model)
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# --------------------------------------------------------------------------
# CORE MATH: Eye Aspect Ratio (EAR)
# --------------------------------------------------------------------------
def eye_aspect_ratio(eye):
    """
    Computes the Eye Aspect Ratio (EAR) given 6 (x, y) eye landmark points.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A falling EAR indicates a closing eye.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    """Vertical mouth opening distance — used as a simple yawn indicator."""
    A = dist.euclidean(mouth[13], mouth[19])   # inner top/bottom lip
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    return (A + B + C) / 3.0


# --------------------------------------------------------------------------
# ALARM HANDLING
# --------------------------------------------------------------------------
class AlarmPlayer:
    """Wraps pygame.mixer so the alarm loops without blocking the video loop."""

    def __init__(self, sound_path):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound(sound_path) if os.path.exists(sound_path) else None
        self.playing = False

    def start(self):
        if self.sound and not self.playing:
            self.sound.play(loops=-1)
            self.playing = True

    def stop(self):
        if self.sound and self.playing:
            self.sound.stop()
            self.playing = False


# --------------------------------------------------------------------------
# EVENT LOGGER (for later analysis in Pandas / Matplotlib)
# --------------------------------------------------------------------------
class EventLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.records = []

    def log(self, event_type, ear_value, mar_value):
        self.records.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            "ear": round(ear_value, 3),
            "mar": round(mar_value, 3),
        })

    def save(self):
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=header, index=False)
        print(f"[INFO] Logged {len(self.records)} events to {self.csv_path}")


# --------------------------------------------------------------------------
# MAIN DETECTION LOOP
# --------------------------------------------------------------------------
def run(camera_index=0):
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()  # type: ignore[attr-defined]
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)  # type: ignore[attr-defined]

    alarm = AlarmPlayer(ALARM_SOUND_PATH)
    logger = EventLogger(LOG_CSV_PATH)

    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(camera_index)
    time.sleep(1.0)

    closed_frame_counter = 0
    alarm_on = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[L_START:L_END]
                right_eye = shape[R_START:R_END]
                mouth = shape[M_START:M_END]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth)

                # Draw eye/mouth contours for visual feedback
                for pts in (left_eye, right_eye, mouth):
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

                # --- Drowsiness logic ---
                if ear < EAR_THRESHOLD:
                    closed_frame_counter += 1
                    if closed_frame_counter >= EAR_CONSEC_FRAMES:
                        if not alarm_on:
                            alarm_on = True
                            alarm.start()
                            logger.log("DROWSINESS_ALERT", ear, mar)
                        cv2.putText(frame, "DROWSINESS ALERT!", (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    closed_frame_counter = 0
                    if alarm_on:
                        alarm_on = False
                        alarm.stop()

                # --- Yawn detection (secondary indicator) ---
                if mar > YAWN_THRESHOLD:
                    cv2.putText(frame, "YAWN DETECTED", (60, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    logger.log("YAWN", ear, mar)

                cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"MAR: {mar:.2f}", (480, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        alarm.stop()
        logger.save()
        print("[INFO] Session ended. Stay safe!")


if __name__ == "__main__":
    run()
