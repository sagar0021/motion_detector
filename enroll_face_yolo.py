import cv2
import numpy as np
import os

# --- MODEL CONFIGURATION ---
DETECTOR_PATH = 'models/face_detection_yunet_2023mar.onnx'
RECOGNIZER_PATH = 'models/face_recognition_sface_2021dec.onnx'
SAVE_PATH = "authorized_face_feature.npy"

# --- LOAD MODELS ---
try:
    detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (0, 0))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# --- MAIN SCRIPT ---
cap = cv2.VideoCapture(0)
authorized_features = []
print("\n--- Photo Booth Enrollment ---")
print("Instructions:")
print("1. Look at the camera. Turn your head slightly for each shot.")
print("2. Press 's' to save the current face angle.")
print("3. Press 'q' to finish and save all captured faces.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    detector.setInputSize((w, h))
    
    faces = detector.detect(frame)
    
    display_frame = frame.copy()

    if faces[1] is not None:
        face_info = faces[1][0]
        coords = face_info[0:4].astype(np.int32)
        cv2.rectangle(display_frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
    
    # Display the number of saved faces
    cv2.putText(display_frame, f"Saved Angles: {len(authorized_features)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Enrollment - Press 's' to save angle, 'q' to finish", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        if len(authorized_features) > 0:
            # Save the list of feature vectors
            np.save(SAVE_PATH, authorized_features)
            print(f"\n✅ Successfully saved {len(authorized_features)} face angles to '{SAVE_PATH}'")
        else:
            print("\n⚠️ No faces were saved.")
        break
        
    elif key == ord('s'):
        if faces[1] is not None:
            try:
                face_aligned = recognizer.alignCrop(frame, faces[1][0])
                feature_vector = recognizer.feature(face_aligned)
                authorized_features.append(feature_vector)
                print(f"✅ Face angle #{len(authorized_features)} captured!")
            except Exception as e:
                print(f"❌ Error processing face: {e}")
        else:
            print("⚠️ No face detected. Please try again.")

cap.release()
cv2.destroyAllWindows()