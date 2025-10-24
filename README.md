# Smart Security System with Motion and Face Detection

This project uses OpenCV and a YOLO-based model to detect motion in a specified region of interest. If motion is detected, it uses a deep learning model to recognize an authorized face. If an unauthorized person or object is detected, it triggers an alarm, sends an email alert, and records video.

## Setup

1.  Clone the repository.
2.  Create a Python 3.9 virtual environment: `py -3.9 -m venv venv`
3.  Activate it: `.\venv\Scripts\activate`
4.  Install the required packages: `pip install -r requirements.txt`

## Usage

1.  Run the enrollment script to save your face features: `python enroll_face_yolo.py`
2.  Run the main application: `python motion_detector.py`