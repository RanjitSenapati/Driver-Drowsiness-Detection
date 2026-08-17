# 🚗 Driver Drowsiness Detection System

Real-time driver drowsiness detection using computer vision. The system watches
the driver's eyes through a webcam, computes the **Eye Aspect Ratio (EAR)**
from 68 facial landmarks, and sounds a **wake-up alarm** the moment it detects
prolonged eye closure — helping prevent micro-sleep accidents.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

| Eyes Open (Normal) | Eyes Closed (Alert Triggered) |
|---|---|
| EAR ≈ 0.30 | EAR < 0.25 → 🔔 Alarm |

*(Add your own screenshots/GIF to `assets/demo.gif` and link them here)*

---

## 🧠 How It Works

1. **Face Detection** — dlib's HOG-based frontal face detector locates the driver's face in each webcam frame.
2. **Landmark Detection** — a 68-point facial landmark predictor maps the eyes and mouth.
3. **Eye Aspect Ratio (EAR)** — computed from 6 eye landmark points:

   ```
   EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2‖p1-p4‖)
   ```

   EAR stays roughly constant with eyes open and drops sharply when eyes close.
4. **Drowsiness Logic** — if EAR stays below `0.25` for `20` consecutive frames (~0.6–0.8s), the system flags drowsiness.
5. **Alarm** — a looping buzzer sound plays via `pygame` until the driver's eyes reopen.
6. **Yawn Detection** — Mouth Aspect Ratio (MAR) is also monitored as a secondary drowsiness cue.
7. **Logging & Analysis** — every alert is timestamped and saved to `data/drowsiness_log.csv`, then visualized in the Jupyter notebook with Pandas + Matplotlib.

---

## 📁 Project Structure

```
drowsiness-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   └── drowsiness_detector.py     # Main real-time detection script
├── notebooks/
│   └── Driver_Drowsiness_Detection.ipynb   # EDA, EAR walkthrough, result analysis
├── models/
│   └── shape_predictor_68_face_landmarks.dat   # (download separately, see below)
├── assets/
│   └── alarm.wav                  # Alarm sound
└── data/
    └── drowsiness_log.csv         # Auto-generated session logs
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the facial landmark model

The 68-point predictor file (~99 MB) isn't included in this repo (GitHub file
size limits). Download it and place it in `models/`:

```bash
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 models/shape_predictor_68_face_landmarks.dat.bz2
```

---

## ▶️ Usage

**Run the live detector:**

```bash
python src/drowsiness_detector.py
```

- Press **`q`** to quit.
- The webcam window shows live EAR/MAR values and contour overlays on the eyes and mouth.
- When drowsiness is detected, a red **"DROWSINESS ALERT!"** banner appears and the alarm sounds.

**Explore the notebook** (EAR theory, single-image test, and post-session analysis):

```bash
jupyter notebook notebooks/Driver_Drowsiness_Detection.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **OpenCV** | Video capture, image processing, drawing overlays |
| **dlib** | Face detection + 68-point facial landmark prediction |
| **NumPy** | Array/vector math for landmark coordinates |
| **Pandas** | Logging and analyzing drowsiness events |
| **Matplotlib / Seaborn** | Visualizing EAR trends and alert statistics |
| **Jupyter Notebook** | Interactive development and result presentation |
| **pygame** | Alarm sound playback |

---

## 📊 Results

- Real-time performance: **~25–30 FPS** on a standard laptop webcam (CPU only).
- EAR threshold of **0.25** with **20 consecutive frames** reliably distinguishes blinks from drowsy eye closure.
- Session analysis (see notebook) plots EAR over time and highlights drowsy segments.

---

## 🚀 Future Improvements

- [ ] Replace geometric EAR with a CNN eye-state classifier (open/closed) trained on the MRL Eye Dataset for higher accuracy in poor lighting
- [ ] Add head-pose estimation to catch head-nodding drowsiness
- [ ] Port to Raspberry Pi / Jetson Nano for real in-vehicle deployment
- [ ] Integrate with a mobile app for remote alerts
- [ ] Add infrared camera support for night driving

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Author

**Your Name**
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

---

⭐ If you found this project useful, consider giving it a star!
# Driver-Drowsiness-Detection
