import os
import cv2
import time
import math
import smtplib
import threading
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from email.mime.text import MIMEText
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SENDER_EMAIL = "manjunathgs172006@gmail.com"
SENDER_PASSWORD = "aqctszremquclabj" 
EMERGENCY_CONTACT = "immanjunath008@gmail.com"

app = Flask(__name__)

# --- GLOBAL STATE ---
state = {
    "seatbelt": False,
    "drowsy": False,
    "yawn_count": 0,
    "suggestion": False,
    "engine_started": False,
    "alert_message": None,
    "distracted": False
    # Phone_use is REMOVED to save speed
}

timers = {"eye": None, "distraction": None, "mail": 0}
yawn_active = False
prev_nose_x = None

# --- EMAIL LOGIC ---
def send_sos_email(reason="Emergency"):
    if time.time() - timers["mail"] < 60: return
    state["alert_message"] = f"🚨 the alert {reason}! Email Sent!"
    threading.Timer(5.0, lambda: state.update({"alert_message": None})).start()

    def _send():
        try:
            msg = MIMEText(f"CRITICAL ALERT    YOUR DRIVER IS IN DANGER SO GO THROUGH HIM FASTLY OR CALL  HIM QUICKLY!!!  : {reason}\nLocation: https://www.google.co.in/maps/place/Kuppam+Engineering+College/@12.7217372,78.3589143,17z/data=!4m6!3m5!1s0x3badb767a1848c87:0xdaead3774cbe234e!8m2!3d12.721662!4d78.3603111!16s%2Fg%2F1tdrcthl?entry=ttu&g_ep=EgoyMDI2MDIxMC4wIKXMDSoASAFQAw%3D%3D")
            msg['Subject'] = f"KAVACH AI - {reason}"
            msg['From'], msg['To'] = SENDER_EMAIL, EMERGENCY_CONTACT
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, EMERGENCY_CONTACT, msg.as_string())
            server.quit()
        except Exception as e: print(f"Email Error: {e}")
    
    threading.Thread(target=_send).start()
    timers["mail"] = time.time()

# --- MATH HELPER ---
def get_ear(landmarks, indices):
    try:
        p = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))
    except: return 0.3

# --- 🟢 LOAD ONLY FACE MODEL (FAST) ---
print("⏳ LOADING FACE MODEL ONLY...")
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("❌ ERROR: 'face_landmarker.task' MISSING")
    face_detector = None
else:
    base_options = python.BaseOptions(model_asset_path=model_path)
    # Using VIDEO mode here because we have plenty of speed now
    face_options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
print("✅ FACE MODEL LOADED.")


def gen_frames():
    global yawn_active, prev_nose_x
    
    cap = cv2.VideoCapture(0)
    # Standard resolution
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ ERROR: Camera not found.")
        return

    while True:
        success, frame = cap.read()
        if not success: break

        # 1. Blocked Camera Check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 10 and state["engine_started"]:
             send_sos_email("Camera defect detected any accident occurred")

        # 2. Prepare Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. RUN AI (Face Only)
        if face_detector:
            try:
                # We use timestamp in ms for VIDEO mode
                face_result = face_detector.detect_for_video(mp_image, int(time.time() * 1000))

                if face_result.face_landmarks:
                    state["seatbelt"] = True
                    m = face_result.face_landmarks[0]
                    h, w, _ = frame.shape

                    # A. DROWSY (EAR)
                    ear = (get_ear(m, [33, 160, 158, 133, 153, 144]) + get_ear(m, [362, 385, 387, 263, 373, 380])) / 2.0
                    if ear < 0.20:
                        if timers["eye"] is None: timers["eye"] = time.time()
                        elif time.time() - timers["eye"] > 2.0: state["drowsy"] = True
                    else: timers["eye"], state["drowsy"] = None, False

                    # B. DISTRACTION (Head Turn)
                    nose_x = m[1].x
                    if prev_nose_x and abs(nose_x - prev_nose_x) > 0.08:
                        if timers["distraction"] is None: timers["distraction"] = time.time()
                        elif time.time() - timers["distraction"] > 2.0: state["distracted"] = True
                    else: timers["distraction"], state["distracted"] = None, False
                    prev_nose_x = nose_x

                    # C. YAWN (Mouth Open)
                    mar = np.linalg.norm(np.array([m[13].x, m[13].y]) - np.array([m[14].x, m[14].y]))
                    if mar > 0.08:
                        if not yawn_active:
                            state["yawn_count"] += 1
                            yawn_active = True
                            if state["yawn_count"] >= 3: state["suggestion"] = True
                    else: yawn_active = False

                    # Draw Visuals (Face Mesh Only)
                    for idx in [33, 263, 1, 13, 14]:
                        cv2.circle(frame, (int(m[idx].x * w), int(m[idx].y * h)), 2, (0, 255, 0), -1)

                else:
                    state["seatbelt"] = False

            except Exception as e: pass

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index(): return render_template('index.html')
@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_status')
def get_status(): return jsonify(state)
@app.route('/start_engine', methods=['POST'])
def start_engine():
    if state["seatbelt"]: state["engine_started"] = True; return jsonify({"success": True})
    return jsonify({"success": False})
@app.route('/stop_engine', methods=['POST'])
def stop_engine(): state["engine_started"] = False; return jsonify({"success": True})
@app.route('/emergency_alert', methods=['POST'])
def emergency_alert(): send_sos_email("Panic Button"); return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)