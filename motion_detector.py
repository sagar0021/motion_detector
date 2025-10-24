import cv2
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from playsound import playsound
import threading

# --- CONFIGURATION ---
MIN_CONTOUR_AREA = 500
THRESHOLD_SENSITIVITY = 30
ALERT_COOLDOWN_SECONDS = 30.0
RECORD_SECONDS_AFTER_MOTION = 5
GRACE_PERIOD_SECONDS = 4.0
RECOGNITION_THRESHOLD = 0.8 # Cosine similarity threshold for a match (higher is stricter)

# --- EMAIL CONFIGURATION ---
SENDER_EMAIL = "sagarsag212000@gmail.com"
SENDER_PASSWORD = "rqie uruy qrbq zquz"
RECEIVER_EMAIL = "sagarsag212000@gmail.com"

# --- YOLO/SFACE MODEL CONFIGURATION ---
DETECTOR_PATH = 'models/face_detection_yunet_2023mar.onnx'
RECOGNIZER_PATH = 'models/face_recognition_sface_2021dec.onnx'

try:
    detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (0, 0))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
    print("✅ YOLO face models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# --- LOAD AUTHORIZED FACE FEATURES (PLURAL) ---
try:
    authorized_features = np.load("authorized_face_feature.npy")
    print("✅ Authorized face features loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: No 'authorized_face_feature.npy' file found.")
    print("Please run the 'enroll_face_yolo.py' script first.")
    exit()

# --- HELPER FUNCTIONS ---
def cosine_similarity(features1, feature2):
    # features1 can be a list of authorized features
    # feature2 is the single feature from the detected face
    return np.dot(features1, feature2.T)

def send_alert_email():
    """Connects to the SMTP server and sends an alert email."""
    try:
        msg = MIMEText("Unauthorized motion detected in the monitored area.")
        msg['Subject'] = 'SECURITY ALERT!'
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ Alert email sent successfully at {time.ctime()}!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

roi_points = []
roi_selected = False
def select_roi(event, x, y, flags, param):
    """Mouse callback function to select ROI."""
    global roi_points, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        roi_selected = False
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        roi_selected = True

# --- MAIN APPLICATION ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('Smart Security System (YOLO)')
cv2.setMouseCallback('Smart Security System (YOLO)', select_roi)

print("Instructions:")
print("1. Click and drag to select a Region of Interest (ROI).")
print("2. Press 'r' to reset the background reference.")
print("3. Press 'q' to quit.")

first_frame_roi = None
last_alert_time = 0
is_recording = False
video_writer = None
motion_stop_time = None
first_unauthorized_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    motion_status = "No Movement"
    
    if roi_selected and len(roi_points) == 2:
        p1, p2 = roi_points
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

        roi = frame[y1:y2, x1:x2]

        if not roi.any() or roi.shape[0] < 1 or roi.shape[1] < 1:
            continue

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)

        if first_frame_roi is None:
            first_frame_roi = gray_roi
            continue

        frame_delta = cv2.absdiff(first_frame_roi, gray_roi)
        thresh = cv2.threshold(frame_delta, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_found = False
        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            motion_found = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        if motion_found:
            unauthorized_event = False
            
            h_roi, w_roi, _ = roi.shape
            detector.setInputSize((w_roi, h_roi))
            faces = detector.detect(roi)
            
            if faces[1] is None:
                unauthorized_event = True
                motion_status = "Unknown Motion Detected!"
            else:
                is_authorized_person_present = False
                for face_info in faces[1]:
                    try:
                        aligned_face = recognizer.alignCrop(roi, face_info)
                        detected_feature = recognizer.feature(aligned_face)
                        
                        # UPDATED to check against all saved face angles
                        scores = cosine_similarity(authorized_features, detected_feature)
                        max_score = np.max(scores)
                        
                        if max_score >= RECOGNITION_THRESHOLD:
                            is_authorized_person_present = True
                            break
                    except Exception:
                        continue
                
                if is_authorized_person_present:
                    motion_status = "Authorized Person"
                else:
                    unauthorized_event = True
                    motion_status = "UNAUTHORIZED PERSON!"
            
            if unauthorized_event:
                if first_unauthorized_time is None:
                    first_unauthorized_time = time.time()
                
                if (time.time() - first_unauthorized_time) > GRACE_PERIOD_SECONDS:
                    motion_stop_time = None
                    if not is_recording:
                        is_recording = True
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"output/{timestamp}.avi"
                        height, width, _ = frame.shape
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
                        print(f"🔴 Started recording to {filename}")
                    current_time = time.time()
                    if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                        last_alert_time = current_time
                        print("🚨 UNAUTHORIZED EVENT! Triggering alerts...")
                        send_alert_email()
                        siren_thread = threading.Thread(target=playsound, args=('siren.mp3',))
                        siren_thread.start()
            else:
                first_unauthorized_time = None

        elif is_recording:
            first_unauthorized_time = None
            if motion_stop_time is None:
                motion_stop_time = time.time()
            if (time.time() - motion_stop_time) >= RECORD_SECONDS_AFTER_MOTION:
                is_recording = False
                if video_writer:
                    video_writer.release()
                print("⚪ Stopped recording.")
                motion_stop_time = None
    
    if is_recording and video_writer:
        video_writer.write(frame)

    if len(roi_points) > 0:
        cv2.rectangle(frame, roi_points[0], roi_points[-1], (0, 255, 0), 2)

    text_color = (0, 0, 255) if "UNAUTHORIZED" in motion_status or "Unknown" in motion_status else (0, 255, 0)
    cv2.putText(frame, f"Status: {motion_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.imshow('Smart Security System (YOLO)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        first_frame_roi = None
        print("🔄 Background reset!")

if is_recording and video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()